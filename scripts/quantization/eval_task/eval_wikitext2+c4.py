import argparse
import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.models import MODEL_CONFIGS
from configs.superweights import SUPERWEIGHTS
from src.hooks import get_nested_attr


def resolve_torch_dtype(name: str):
    name = name.lower()

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


def get_superweight_restore_indices(row: int, col: int, shape, neighborhood: str = "scalar"):
    n_rows, n_cols = shape
    indices = set()

    def add(r, c):
        if 0 <= r < n_rows and 0 <= c < n_cols:
            indices.add((r, c))

    add(row, col)

    if neighborhood in ["row", "cross"]:
        add(row, col - 1)
        add(row, col + 1)

    if neighborhood in ["column", "cross"]:
        add(row - 1, col)
        add(row + 1, col)

    return sorted(indices)


@torch.no_grad()
def quantize_activation(x: torch.Tensor, n_bits: int):
    if not torch.is_floating_point(x):
        return x

    orig_dtype = x.dtype
    x_float = x.float()

    qmin = -(2 ** (n_bits - 1))
    qmax = (2 ** (n_bits - 1)) - 1

    max_abs = x_float.abs().amax(dim=-1, keepdim=True)
    scale = max_abs / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    q = torch.clamp(torch.round(x_float / scale), qmin, qmax)

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
def clip_weight_tensor_zscore(w: torch.Tensor, clip_z: Optional[float]):
    if clip_z is None or clip_z <= 0:
        return w

    orig_dtype = w.dtype
    w_float = w.float()

    mean = w_float.mean()
    std = w_float.std(unbiased=False)

    if std.item() == 0:
        return w

    lower = mean - clip_z * std
    upper = mean + clip_z * std

    return torch.clamp(w_float, lower, upper).to(orig_dtype)


@torch.no_grad()
def quantize_weight_tensor(w: torch.Tensor, n_bits: int):
    """
    Asymmetric min-max RTN quantization:
    Q(W) = round((W - min(W)) / delta)
    W_q = Q(W) * delta + min(W)
    """

    if n_bits <= 0:
        raise ValueError("n_bits must be positive")

    orig_dtype = w.dtype
    w_float = w.float()

    qmin = 0
    qmax = (2 ** n_bits) - 1

    w_min = w_float.min()
    w_max = w_float.max()

    if (w_max - w_min).item() == 0:
        return w_float.to(orig_dtype)

    delta = (w_max - w_min) / qmax

    q = torch.clamp(
        torch.round((w_float - w_min) / delta),
        qmin,
        qmax,
    )

    return (q * delta + w_min).to(orig_dtype)


@torch.no_grad()
def quantize_weight_tensor_blockwise_2d(
    w: torch.Tensor,
    n_bits: int,
    block_rows: int = 128,
    block_cols: int = 128,
):
    """
    Blockwise asymmetric min-max RTN quantization.
    Each 2D block gets its own min/max range.
    """

    if n_bits <= 0:
        raise ValueError("n_bits must be positive")

    if block_rows <= 0 or block_cols <= 0:
        raise ValueError("block_rows and block_cols must be positive")

    if w.ndim != 2:
        return quantize_weight_tensor(w, n_bits)

    orig_dtype = w.dtype
    w_float = w.float()
    out = torch.empty_like(w_float)

    qmin = 0
    qmax = (2 ** n_bits) - 1

    n_rows, n_cols = w_float.shape

    for r0 in range(0, n_rows, block_rows):
        r1 = min(r0 + block_rows, n_rows)

        for c0 in range(0, n_cols, block_cols):
            c1 = min(c0 + block_cols, n_cols)
            block = w_float[r0:r1, c0:c1]

            b_min = block.min()
            b_max = block.max()

            if (b_max - b_min).item() == 0:
                out[r0:r1, c0:c1] = block
                continue

            delta = (b_max - b_min) / qmax

            q = torch.clamp(
                torch.round((block - b_min) / delta),
                qmin,
                qmax,
            )

            out[r0:r1, c0:c1] = q * delta + b_min

    return out.to(orig_dtype)


@torch.no_grad()
def quantize_parameter(
    param: torch.Tensor,
    n_bits: int,
    quant_granularity: str = "tensor",
    block_rows: int = 128,
    block_cols: int = 128,
    clip_z: Optional[float] = None,
):
    w = clip_weight_tensor_zscore(param, clip_z)

    if quant_granularity == "tensor":
        return quantize_weight_tensor(w, n_bits)

    if quant_granularity == "block2d":
        return quantize_weight_tensor_blockwise_2d(
            w,
            n_bits=n_bits,
            block_rows=block_rows,
            block_cols=block_cols,
        )

    raise ValueError(f"Unsupported quant_granularity: {quant_granularity}")


@torch.no_grad()
def apply_weight_quantization(
    model,
    n_bits: int,
    quant_granularity: str = "tensor",
    block_rows: int = 128,
    block_cols: int = 128,
    clip_z: Optional[float] = None,
):
    """
    Quantize only nn.Linear layers.
    Skip lm_head and embeddings.
    """

    quantized_modules = 0
    skipped_modules = 0

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if "lm_head" in module_name:
            skipped_modules += 1
            continue

        if "embed" in module_name:
            skipped_modules += 1
            continue

        q_weight = quantize_parameter(
            module.weight.data,
            n_bits=n_bits,
            quant_granularity=quant_granularity,
            block_rows=block_rows,
            block_cols=block_cols,
            clip_z=clip_z,
        )

        module.weight.data.copy_(q_weight)
        quantized_modules += 1

    print(
        f"Weight quantization done: quantized Linear modules={quantized_modules}, "
        f"skipped Linear modules={skipped_modules}",
        flush=True,
    )

    return model


@torch.no_grad()
def collect_superweights(
    model,
    model_key: str,
    sw_scale: float = 1.0,
    restore_neighborhood: str = "scalar",
):
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
        weight = module.weight.data

        restore_indices = get_superweight_restore_indices(
            row=row,
            col=col,
            shape=weight.shape,
            neighborhood=restore_neighborhood,
        )

        for rr, cc in restore_indices:
            is_center = rr == row and cc == col
            value = weight[rr, cc].detach().clone()

            if is_center:
                value = value * sw_scale

            protected.append(
                {
                    "layer": layer_idx,
                    "center_row": row,
                    "center_col": col,
                    "row": int(rr),
                    "col": int(cc),
                    "value": value,
                    "is_center": bool(is_center),
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


def prepare_model(
    model,
    model_key: str,
    mode: str,
    bits: int,
    activation_bits: Optional[int],
    sw_scale: float,
    restore_neighborhood: str = "scalar",
    quant_granularity: str = "tensor",
    block_rows: int = 128,
    block_cols: int = 128,
    clip_z: Optional[float] = None,
):
    activation_handles = []
    protected = []

    if mode == "fp16":
        return model, protected, activation_handles

    if mode == "naive":
        apply_weight_quantization(
            model=model,
            n_bits=bits,
            quant_granularity=quant_granularity,
            block_rows=block_rows,
            block_cols=block_cols,
            clip_z=None,
        )

        if activation_bits is not None and activation_bits > 0:
            activation_handles = add_activation_hooks(model, activation_bits)

        return model, protected, activation_handles

    if mode == "super":
        protected = collect_superweights(
            model=model,
            model_key=model_key,
            sw_scale=sw_scale,
            restore_neighborhood=restore_neighborhood,
        )

        apply_weight_quantization(
            model=model,
            n_bits=bits,
            quant_granularity=quant_granularity,
            block_rows=block_rows,
            block_cols=block_cols,
            clip_z=clip_z,
        )

        restore_superweights(
            model=model,
            model_key=model_key,
            protected=protected,
        )

        if activation_bits is not None and activation_bits > 0:
            activation_handles = add_activation_hooks(model, activation_bits)

        return model, protected, activation_handles

    raise ValueError(f"Unsupported mode: {mode}")


def load_texts(dataset: str, split: str, limit: int):
    if dataset == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [x["text"] for x in ds if x["text"].strip()]
        return texts[:limit]

    if dataset == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)

        texts = []
        for x in ds:
            text = x.get("text", "")
            if text.strip():
                texts.append(text)

            if len(texts) >= limit:
                break

        return texts

    raise ValueError(f"Unsupported dataset: {dataset}")


@torch.no_grad()
def evaluate_ppl(model, tokenizer, texts, max_length: int):
    device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0

    model.eval()

    for i, text in enumerate(texts):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        input_ids = enc.input_ids.to(device)

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

        if (i + 1) % 25 == 0:
            current_ppl = math.exp(total_loss / total_tokens)
            print(f"{i+1}/{len(texts)} | current_ppl={current_ppl:.4f}", flush=True)

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

def build_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-key", required=True, choices=sorted(MODEL_CONFIGS.keys()))
    parser.add_argument("--mode", default="fp16", choices=["fp16", "naive", "super"])
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--activation-bits", type=int, default=None)
    parser.add_argument("--sw-scale", type=float, default=1.0)

    parser.add_argument(
        "--restore-neighborhood",
        type=str,
        default="scalar",
        choices=["scalar", "row", "column", "cross"],
    )

    parser.add_argument("--quant-granularity", type=str, default="tensor", choices=["tensor", "block2d"])
    parser.add_argument("--block-rows", type=int, default=128)
    parser.add_argument("--block-cols", type=int, default=128)
    parser.add_argument("--clip-z", type=float, default=None)

    parser.add_argument("--dataset", required=True, choices=["wikitext2", "c4"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)

    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output-json", required=True)

    return parser


def main():
    args = build_arg_parser().parse_args()

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
        f"Preparing model mode={args.mode}, bits={args.bits}, "
        f"activation_bits={args.activation_bits}, sw_scale={args.sw_scale}, "
        f"restore_neighborhood={args.restore_neighborhood}, "
        f"quant_granularity={args.quant_granularity}, "
        f"block_rows={args.block_rows}, block_cols={args.block_cols}, "
        f"clip_z={args.clip_z}",
        flush=True,
    )

    model, protected, activation_handles = prepare_model(
        model=model,
        model_key=args.model_key,
        mode=args.mode,
        bits=args.bits,
        activation_bits=args.activation_bits,
        sw_scale=args.sw_scale,
        restore_neighborhood=args.restore_neighborhood,
        quant_granularity=args.quant_granularity,
        block_rows=args.block_rows,
        block_cols=args.block_cols,
        clip_z=args.clip_z,
    )

    print(f"Protected values: {len(protected)}", flush=True)
    print(f"Activation hooks: {len(activation_handles)}", flush=True)

    print(f"Loading dataset={args.dataset}, split={args.split}, limit={args.limit}", flush=True)
    texts = load_texts(
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
    )

    metrics = evaluate_ppl(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=args.max_length,
    )

    protected_summary = [
        {
            "layer": int(x["layer"]),
            "center_row": int(x["center_row"]),
            "center_col": int(x["center_col"]),
            "row": int(x["row"]),
            "col": int(x["col"]),
            "is_center": bool(x["is_center"]),
        }
        for x in protected
    ]

    result = {
        "benchmark": "perplexity",
        "dataset": args.dataset,
        "model_key": args.model_key,
        "model_id": model_id,
        "mode": args.mode,
        "bits": args.bits,
        "activation_bits": args.activation_bits,
        "sw_scale": args.sw_scale,
        "restore_neighborhood": args.restore_neighborhood,
        "quant_granularity": args.quant_granularity,
        "block_rows": args.block_rows,
        "block_cols": args.block_cols,
        "clip_z": args.clip_z,
        "num_protected_values": len(protected),
        "protected_values": protected_summary,
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