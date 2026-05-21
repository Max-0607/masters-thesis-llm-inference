import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM

GPTQ_ROOT = Path(__file__).resolve().parent
if str(GPTQ_ROOT) not in sys.path:
    sys.path.insert(0, str(GPTQ_ROOT))

from gptq import GPTQ
from modelutils import find_layers
from quant import Quantizer

DEV = torch.device("cuda:0")


def get_olmo(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    model.seqlen = 2048
    return model


@torch.no_grad()
def olmo_sequential(model, dataloader, dev, args):
    print("Starting ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None:
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args["nsamples"], model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=dev,
    )

    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            cache["position_ids"] = kwargs.get("position_ids", None)
            raise ValueError

    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            batch_inp = batch[0].to(dev)
            position_ids = torch.arange(
                batch_inp.shape[1],
                device=dev,
                dtype=torch.long,
            ).unsqueeze(0)
            model(batch_inp, position_ids=position_ids)
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None:
        model.model.rotary_emb = model.model.rotary_emb.cpu()

    if model.model.norm is not None:
        model.model.norm = model.model.norm.cpu()

    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    if attention_mask is not None:
        attention_mask = attention_mask.to(dev)
    if position_ids is not None:
        position_ids = position_ids.to(dev)

    position_embeddings = None
    if (
        hasattr(model.model, "rotary_emb")
        and model.model.rotary_emb is not None
        and position_ids is not None
    ):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
        position_ids = position_ids.to(dev)
        position_embeddings = model.model.rotary_emb(
            inps[0].unsqueeze(0).to(dev),
            position_ids,
        )

    print("Ready.")

    quantizers = {}

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        sequential = [
            ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            ["self_attn.o_proj"],
            ["mlp.up_proj", "mlp.gate_proj"],
            ["mlp.down_proj"],
        ]

        for names in sequential:
            subset = {n: full[n] for n in names if n in full}
            if not subset:
                continue

            gptq = {}

            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args["wbits"],
                    perchannel=True,
                    sym=False,
                    mse=False,
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(args["nsamples"]):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0]

            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print("Quantizing ...")
                gptq[name].fasterquant(
                    percdamp=args["percdamp"],
                    groupsize=args["groupsize"],
                    actorder=args["act_order"],
                    static_groups=False,
                )
                quantizers[f"model.layers.{i}.{name}"] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args["nsamples"]):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return quantizers


def get_wikitext2_olmo(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []

    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    return trainloader, TokenizerWrapper(testenc.input_ids)
