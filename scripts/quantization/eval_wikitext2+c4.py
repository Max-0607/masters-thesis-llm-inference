import argparse
import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.models import MODEL_CONFIGS
from configs.superweights import SUPERWEIGHTS
from src.hooks import get_nested_attr


def resolve_torch_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def make_json_safe(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    return obj


@torch.no_grad()
def quantize_activation(x: torch.Tensor, n_bits: int):
    if not torch.is_floating_point(x):
        return x

    orig_dtype = x.dtype
    x = x.float()

    qmin = -(2 ** (n_bits - 1))
    qmax = (2 ** (n_bits - 1)) - 1

    max_abs = x.abs().amax(dim=-1, keepdim=True)
    scale = max_abs / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    q = torch.clamp(torch.round(x / scale), qmin, qmax)
    return (q * scale).to(orig_dtype)


def add_activation_hooks(model, n_bits: int):
    handles = []

    def hook(module, inputs):
        if not inputs:
            return inputs
        x = inputs[0]
        if not torch.is_tensor(x):
            return inputs
        return (quantize_activation(x, n_bits),) + tuple(inputs[1:])

    for module in model.modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_pre_hook(hook))

    return handles


@torch.no_grad()
def quantize_weight_tensor(w: torch.Tensor, n_bits: int):
    orig_dtype = w.dtype
    w = w.float()

    qmin = -(2 ** (n_bits - 1))
    qmax = (2 ** (n_bits - 1)) - 1

    max_abs = w.abs().amax()

    if max_abs.item() == 0:
        return w.to(orig_dtype)

    scale = max_abs / qmax
    q = torch.clamp(torch.round(w / scale), qmin, qmax)

    return (q * scale).to(orig_dtype)


@torch.no_grad()
def apply_weight_quantization(model, n_bits: int):
    for name, param in model.named_parameters():
        if "weight" not in name:
            continue
        if param.ndim < 2:
            continue

        param.data.copy_(quantize_weight_tensor(param.data, n_bits))

    return model


@torch.no_grad()
def collect_superweights(model, model_key: str, sw_scale: float = 1.0):
    if model_key not in SUPERWEIGHTS:
        raise ValueError(f"No superweights registered for model_key={model_key}")

    model_cfg = MODEL_CONFIGS[model_key]
    layers = get_nested_attr(model, model_cfg["layer_path"])
    down_proj_path = model_cfg["down_proj_path"]

    protected = []

    for entry in SUPERWEIGHTS[model_key]:
        layer_idx = int(entry["layer"])
        row = int(entry["row"])
        col = int(entry["col"])

        module = get_nested_attr(layers[layer_idx], down_proj_path)
        value = module.weight.data[row, col].detach().clone() * sw_scale

        protected.append(
            {
                "layer": layer_idx,
                "row": row,
                "col": col,
                "value": value,
            }
        )

    return protected


@torch.no_grad()
def restore_superweights(model, model_key: str, protected):
    model_cfg = MODEL_CONFIGS[model_key]
    layers = get_nested_attr(model, model_cfg["layer_path"])
    down_proj_path = model_cfg["down_proj_path"]

    for item in protected:
        module = get_nested_attr(layers[item["layer"]], down_proj_path)
        module.weight.data[item["row"], item["col"]] = item["value"]

    return model


def prepare_model(model, model_key: str, mode: str, bits: int, activation_bits: Optional[int], sw_scale: float):
    activation_handles = []
    protected = []

    if mode == "fp16":
        return model, protected, activation_handles

    if mode == "naive":
        apply_weight_quantization(model, bits)

        if activation_bits is not None and activation_bits > 0:
            activation_handles = add_activation_hooks(model, activation_bits)

        return model, protected, activation_handles

    if mode == "super":
        protected = collect_superweights(model, model_key, sw_scale=sw_scale)
        apply_weight_quantization(model, bits)
        restore_superweights(model, model_key, protected)

        if activation_bits is not None and activation_bits > 0:
            activation_handles = add_activation_hooks(model, activation_bits)

        return model, protected, activation_handles

    raise ValueError(f"Unsupported mode: {mode}")


def load_texts(dataset: str, split: str, limit: int):
    if dataset == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        texts = [x["text"] for x in ds if x["text"].strip()]

    elif dataset == "c4":
        ds = load_dataset("allenai/c4", "en", split=split, streaming=True)
        texts = []
        for x in ds:
            text = x.get("text", "")
            if text.strip():
                texts.append(text)
            if len(texts) >= limit:
                break

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return texts[:limit]


@torch.no_grad()
def evaluate_ppl(model, tokenizer, texts, max_length: int):
    device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0

    model.eval()

    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        input_ids = enc["input_ids"].to(device)

        if input_ids.shape[1] < 2:
            continue

        outputs = model(
            input_ids=input_ids,
            labels=input_ids,
            use_cache=False,
            return_dict=True,
        )

        num_tokens = input_ids.shape[1] - 1
        loss = outputs.loss.float().item()

        total_loss += loss * num_tokens
        total_tokens += num_tokens

    if total_tokens == 0:
        raise RuntimeError("No valid tokens evaluated.")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    return {
        "loss": avg_loss,
        "perplexity": ppl,
        "num_tokens": total_tokens,
        "num_texts": len(texts),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-key", required=True, choices=sorted(MODEL_CONFIGS.keys()))
    parser.add_argument("--mode", default="fp16", choices=["fp16", "naive", "super"])
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--activation-bits", type=int, default=None)
    parser.add_argument("--sw-scale", type=float, default=1.0)

    parser.add_argument("--dataset", required=True, choices=["wikitext2", "c4"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)

    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output-json", required=True)

    args = parser.parse_args()

    model_cfg = MODEL_CONFIGS[args.model_key]
    model_id = model_cfg["hf_name"]
    torch_dtype = resolve_torch_dtype(args.dtype)

    print(f"Loading tokenizer: {model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_id}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    print(
        f"Preparing model mode={args.mode}, bits={args.bits}, activation_bits={args.activation_bits}",
        flush=True,
    )

    model, protected, activation_handles = prepare_model(
        model=model,
        model_key=args.model_key,
        mode=args.mode,
        bits=args.bits,
        activation_bits=args.activation_bits,
        sw_scale=args.sw_scale,
    )

    print(f"Protected values: {len(protected)}", flush=True)
    print(f"Activation hooks: {len(activation_handles)}", flush=True)

    print(f"Loading dataset={args.dataset}, split={args.split}, limit={args.limit}", flush=True)
    texts = load_texts(args.dataset, args.split, args.limit)

    metrics = evaluate_ppl(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=args.max_length,
    )

    result = {
        "benchmark": "perplexity",
        "dataset": args.dataset,
        "model_key": args.model_key,
        "model_id": model_id,
        "mode": args.mode,
        "bits": args.bits,
        "activation_bits": args.activation_bits,
        "sw_scale": args.sw_scale,
        "num_protected_values": len(protected),
        "num_activation_hooks": len(activation_handles),
        "dtype": args.dtype,
        "split": args.split,
        "limit": args.limit,
        "max_length": args.max_length,
        **metrics,
    }

    result = make_json_safe(result)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print(f"Saved result to: {output_path}", flush=True)


if __name__ == "__main__":
    main()