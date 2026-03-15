import argparse
import torch
import torch.nn as nn

from quant import Quantizer
from gptq import GPTQ
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelutils import find_layers


def get_model(model_name: str, dtype=torch.float16):
    def skip(*args, **kwargs):
        pass

    # Avoid useless re-init when loading checkpoints
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    if hasattr(model.config, "max_position_embeddings"):
        model.seqlen = model.config.max_position_embeddings
    else:
        model.seqlen = 2048

    return model, tokenizer


def get_transformer_layers(model):
    # LLaMA / Mistral / Phi-3 / many HF causal LMs
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    # OPT-style
    if (
        hasattr(model, "model")
        and hasattr(model.model, "decoder")
        and hasattr(model.model.decoder, "layers")
    ):
        return model.model.decoder.layers

    raise ValueError(f"Unsupported model structure: {type(model)}")


def move_embeddings_to_device(model, dev):
    # OPT-style
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        dec = model.model.decoder

        if hasattr(dec, "embed_tokens") and dec.embed_tokens is not None:
            dec.embed_tokens = dec.embed_tokens.to(dev)

        if hasattr(dec, "embed_positions") and dec.embed_positions is not None:
            dec.embed_positions = dec.embed_positions.to(dev)

        if hasattr(dec, "project_out") and dec.project_out is not None:
            dec.project_out = dec.project_out.to(dev)

        if hasattr(dec, "project_in") and dec.project_in is not None:
            dec.project_in = dec.project_in.to(dev)

        if hasattr(dec, "final_layer_norm") and dec.final_layer_norm is not None:
            dec.final_layer_norm = dec.final_layer_norm.to(dev)

        return

    # LLaMA / Mistral / Phi-3 / similar
    if hasattr(model, "model"):
        core = model.model

        if hasattr(core, "embed_tokens") and core.embed_tokens is not None:
            core.embed_tokens = core.embed_tokens.to(dev)

        if hasattr(core, "norm") and core.norm is not None:
            core.norm = core.norm.to(dev)

        if hasattr(core, "rotary_emb") and core.rotary_emb is not None:
            core.rotary_emb = core.rotary_emb.to(dev)

        return

    raise ValueError(f"Unsupported model structure for embeddings: {type(model)}")


def move_embeddings_to_cpu(model):
    # OPT-style
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        dec = model.model.decoder

        if hasattr(dec, "embed_tokens") and dec.embed_tokens is not None:
            dec.embed_tokens = dec.embed_tokens.cpu()

        if hasattr(dec, "embed_positions") and dec.embed_positions is not None:
            dec.embed_positions = dec.embed_positions.cpu()

        if hasattr(dec, "project_out") and dec.project_out is not None:
            dec.project_out = dec.project_out.cpu()

        if hasattr(dec, "project_in") and dec.project_in is not None:
            dec.project_in = dec.project_in.cpu()

        if hasattr(dec, "final_layer_norm") and dec.final_layer_norm is not None:
            dec.final_layer_norm = dec.final_layer_norm.cpu()

        return

    # LLaMA / Mistral / Phi-3 / similar
    if hasattr(model, "model"):
        core = model.model

        if hasattr(core, "embed_tokens") and core.embed_tokens is not None:
            core.embed_tokens = core.embed_tokens.cpu()

        if hasattr(core, "norm") and core.norm is not None:
            core.norm = core.norm.cpu()

        if hasattr(core, "rotary_emb") and core.rotary_emb is not None:
            core.rotary_emb = core.rotary_emb.cpu()

        return

    raise ValueError(f"Unsupported model structure for embeddings: {type(model)}")


def inspect_first_layer(model):
    layers = get_transformer_layers(model)
    first = layers[0]

    print(f"Model type: {type(model)}")
    print(f"Number of transformer layers: {len(layers)}")
    print(f"First layer type: {type(first)}")
    print("\nLinear layers in first transformer block:\n")

    for name, module in first.named_modules():
        if isinstance(module, nn.Linear):
            print(f"{name}: in={module.in_features}, out={module.out_features}")


def inspect_all_linear_layers(model):
    print("\nAll linear layers in model:\n")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"{name}: in={module.in_features}, out={module.out_features}")


@torch.no_grad()
def dry_run_structure(model_name: str, device: str):
    model, tokenizer = get_model(model_name)
    dev = torch.device(device)

    move_embeddings_to_device(model, dev)
    layers = get_transformer_layers(model)
    layers[0] = layers[0].to(dev)

    print(f"\nLoaded model: {model_name}")
    print(f"Tokenizer type: {type(tokenizer)}")
    print(f"Sequence length: {model.seqlen}")
    print(f"First transformer layer moved to: {dev}")

    subset = find_layers(layers[0], layers=[nn.Linear])
    print("\nLinear layers found in first block via find_layers():\n")
    for name, module in subset.items():
        print(f"{name}: in={module.in_features}, out={module.out_features}")

    layers[0] = layers[0].cpu()
    move_embeddings_to_cpu(model)
    torch.cuda.empty_cache()


def get_calibration_dataloader(tokenizer, model_name, nsamples=32, seqlen=2048):
    texts = [
        """Large language models have become a central building block of modern NLP systems.
        They can solve a wide range of tasks, including reasoning, translation, summarization,
        and coding. However, their deployment is often limited by high memory and compute costs.
        Quantization is one of the most common approaches to reduce these costs while trying to
        preserve model quality as much as possible.""",

        """Post-training quantization methods aim to compress models without additional retraining.
        Among these methods, GPTQ is a widely used baseline that performs layer-wise weight
        quantization based on second-order information. More recently, researchers have argued
        that a very small number of weights may dominate model behavior, suggesting that these
        weights should be preserved with higher precision.""",

        """In practice, evaluating quantized language models requires care. Different methods may
        use different calibration datasets, group sizes, or quantization schemes. For a fair
        comparison, it is important to keep the evaluation task, the model, and the reporting
        metric constant. In this project, HellaSwag is used as a common benchmark and acc_norm
        is used as the main evaluation metric.""",

        """Reasoning benchmarks are often sensitive to quantization noise because they depend on
        subtle token probability differences. If important channels or projection matrices are
        distorted too strongly, multiple-choice performance can degrade substantially. This makes
        such benchmarks useful when studying the robustness of quantization methods.""",

        """A robust quantization pipeline usually consists of three parts: collecting calibration
        activations, estimating quantization parameters, and evaluating the quantized model under
        a fixed benchmark setup. If the calibration data is too small or unrepresentative, the
        resulting quantizer may become numerically unstable or fail to preserve model behavior."""
    ]

    samples = []
    i = 0
    max_len = min(seqlen, 1024)

    while len(samples) < nsamples:
        text = texts[i % len(texts)]
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )
        samples.append((enc["input_ids"],))
        i += 1

    return samples


@torch.no_grad()
def model_sequential(model, dataloader, dev, args):
    print("Starting quantization ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = get_transformer_layers(model)

    move_embeddings_to_device(model, dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype

    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("model.config.hidden_size not found")

    calib_seqlen = dataloader[0][0].shape[1]
    print(f"Calibration sequence length: {calib_seqlen}")

    inps = torch.zeros(
        (args.nsamples, calib_seqlen, hidden_size),
        dtype=dtype,
        device=dev,
    )
    cache = {
        "i": 0,
        "attention_mask": None,
        "position_ids": None,
    }

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            hidden_states = inp[0] if isinstance(inp, tuple) else inp
            inps[cache["i"]] = hidden_states
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            cache["position_ids"] = kwargs.get("position_ids", None)
            raise ValueError

    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    move_embeddings_to_cpu(model)
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    print("Collected calibration activations.")

    quantizers = {}
    max_layers = min(len(layers), args.max_layers)

    for i in range(max_layers):
        print(f"\nLayer {i}")
        layer = layers[i].to(dev)

        subset = find_layers(layer, layers=[nn.Linear])

        skip_patterns = []
        if args.skip_o_proj:
            skip_patterns.append("o_proj")
        if args.skip_mlp:
            skip_patterns.extend(["gate_up_proj", "down_proj"])

        subset = {
            name: module
            for name, module in subset.items()
            if not any(p in name for p in skip_patterns)
        }

        if len(subset) == 0:
            print(f"[WARN] No layers left to quantize in layer {i}, skipping.")
            for j in range(args.nsamples):
                layer_kwargs = {}
                if attention_mask is not None:
                    layer_kwargs["attention_mask"] = attention_mask
                if position_ids is not None:
                    layer_kwargs["position_ids"] = position_ids

                out = layer(inps[j].unsqueeze(0), **layer_kwargs)
                outs[j] = out[0] if isinstance(out, tuple) else out

            layers[i] = layer.cpu()
            del layer
            torch.cuda.empty_cache()
            inps, outs = outs, inps
            continue

        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(
                args.wbits,
                perchannel=True,
                sym=args.sym,
                mse=False,
            )

        def add_batch(name):
            def tmp(_, inp, out):
                x = inp[0] if isinstance(inp, tuple) else inp
                gptq[name].add_batch(x.data, out.data)
            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            layer_kwargs = {}
            if attention_mask is not None:
                layer_kwargs["attention_mask"] = attention_mask
            if position_ids is not None:
                layer_kwargs["position_ids"] = position_ids

            out = layer(inps[j].unsqueeze(0), **layer_kwargs)
            outs[j] = out[0] if isinstance(out, tuple) else out

        for h in handles:
            h.remove()

        for name in subset:
            print(f"Quantizing layer {i} / {name}")
            try:
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=args.groupsize,
                    actorder=args.act_order,
                    static_groups=args.static_groups,
                )
                quantizers[f"model.layers.{i}.{name}"] = gptq[name].quantizer
            except Exception as e:
                print(f"[WARN] Skipping {i}:{name} because of error: {e}")
            finally:
                gptq[name].free()

        for j in range(args.nsamples):
            layer_kwargs = {}
            if attention_mask is not None:
                layer_kwargs["attention_mask"] = attention_mask
            if position_ids is not None:
                layer_kwargs["position_ids"] = position_ids

            out = layer(inps[j].unsqueeze(0), **layer_kwargs)
            outs[j] = out[0] if isinstance(out, tuple) else out

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return quantizers


@torch.no_grad()
def evaluate_hellaswag_lm_eval(model, tokenizer, device="cuda:0", batch_size=1, eval_limit=None):
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    print("\nStarting HellaSwag evaluation with lm_eval ...")

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=["hellaswag"],
        num_fewshot=0,
        batch_size=batch_size,
        limit=eval_limit,
    )

    print("\nHellaSwag raw result dict:")
    print(results["results"]["hellaswag"])

    hellaswag = results["results"]["hellaswag"]

    if "acc_norm,none" in hellaswag:
        print(f"\nacc_norm = {hellaswag['acc_norm,none']}")
    elif "acc_norm" in hellaswag:
        print(f"\nacc_norm = {hellaswag['acc_norm']}")
    else:
        print("\n[WARN] acc_norm key not found in HellaSwag results.")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="HF model name or local path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for dry-run / quantization / evaluation",
    )
    parser.add_argument(
        "--all-linears",
        action="store_true",
        help="Print all linear layers in the full model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Move embeddings + first transformer block to device and inspect find_layers()",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Run GPTQ quantization",
    )
    parser.add_argument(
        "--eval-hellaswag",
        action="store_true",
        help="Evaluate the current in-memory model on HellaSwag (0-shot) using lm_eval",
    )
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--groupsize", type=int, default=-1)
    parser.add_argument("--sym", action="store_true")
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--act-order", action="store_true")
    parser.add_argument("--static-groups", action="store_true")
    parser.add_argument("--max-layers", type=int, default=1)
    parser.add_argument("--skip-o-proj", action="store_true")
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--eval-limit", type=int, default=None)
    args = parser.parse_args()

    model, tokenizer = get_model(args.model)

    inspect_first_layer(model)

    if args.all_linears:
        inspect_all_linear_layers(model)

    if args.dry_run:
        dry_run_structure(args.model, args.device)

    if args.quantize:
        dataloader = get_calibration_dataloader(
            tokenizer,
            args.model,
            nsamples=args.nsamples,
            seqlen=model.seqlen,
        )
        quantizers = model_sequential(
            model,
            dataloader,
            torch.device(args.device),
            args,
        )
        print("\nFinished quantization. Quantized modules:\n")
        for k in quantizers.keys():
            print(k)

    if args.eval_hellaswag:
        evaluate_hellaswag_lm_eval(
            model,
            tokenizer,
            device=args.device,
            batch_size=args.eval_batch_size,
            eval_limit=args.eval_limit,
        )


if __name__ == "__main__":
    main()