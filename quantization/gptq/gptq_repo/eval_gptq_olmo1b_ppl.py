import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import random
from datasets import load_dataset
from gptq import GPTQ
from modelutils import find_layers
from quant import Quantizer, Quant3Linear, make_quant3


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
                batch_inp.shape[1], device=dev, dtype=torch.long
            ).unsqueeze(0)
            model(batch_inp, position_ids=position_ids)
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
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
                    args["wbits"], perchannel=True, sym=False, mse=False
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
            )[0]

        layers[i] = layer.cpu()
        del layer
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return quantizers



def apply_quantizers(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])

    print("Packing quantized layers ...")
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print("Done.")
    return model


@torch.no_grad()
def olmo_eval_ppl(model, testenc, dev):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size),
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

    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            position_ids = torch.arange(
                batch.shape[1], device=dev, dtype=torch.long
            ).unsqueeze(0)
            model(batch, position_ids=position_ids)
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    if attention_mask is not None:
        attention_mask = attention_mask.to(dev)
    if position_ids is not None:
        position_ids = position_ids.to(dev)

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]

        layers[i] = layer.cpu()
        del layer
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    ppl_value = ppl.item()
    print(ppl_value)

    model.config.use_cache = use_cache
    return ppl_value

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

if __name__ == "__main__":
    model_id = "allenai/OLMo-1B-0724-hf"
    output_json = Path("gptq_olmo1b_wikitext2.json")

    args = {
        "nsamples": 4,
        "wbits": 4,
        "groupsize": 128,
        "percdamp": 0.01,
        "act_order": True,
    }

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading model: {model_id}")
    model = get_olmo(model_id)
    model.eval()

    print("Loading calibration/test data ...")
    dataloader, testloader = get_wikitext2_olmo(
        nsamples=args["nsamples"],
        seed=0,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )

    print("Running GPTQ quantization ...")
    tick = time.time()
    quantizers = olmo_sequential(model, dataloader, DEV, args)
    print(f"Quantization done in {time.time() - tick:.2f}s")

    print("Running WikiText-2 evaluation on quantized model ...")
    ppl = olmo_eval_ppl(model, testloader, DEV)

    result = {
        "model": "olmo-1b",
        "model_id": model_id,
        "method": "gptq_runtime_probe",
        "bits": args["wbits"],
        "groupsize": args["groupsize"],
        "nsamples": args["nsamples"],
        "dataset": "wikitext2",
        "perplexity": ppl,
    }

    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved result to {output_json}")