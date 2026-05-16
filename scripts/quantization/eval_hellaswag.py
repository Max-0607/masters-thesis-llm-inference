import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
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
def uniform_quantize_activation_tensor(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")

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


def add_activation_quant_hooks(model, n_bits: int):
    handles = []

    def pre_hook(module, inputs):
        if not inputs:
            return inputs

        x = inputs[0]

        if not torch.is_tensor(x):
            return inputs

        x_q = uniform_quantize_activation_tensor(x, n_bits)
        return (x_q,) + tuple(inputs[1:])

    for module in model.modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_pre_hook(pre_hook))

    return handles


@torch.no_grad()
def clip_weight_tensor_zscore(w: torch.Tensor, clip_z: Optional[float]) -> torch.Tensor:
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
def uniform_quantize_weight_tensor(w: torch.Tensor, n_bits: int) -> torch.Tensor:
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")

    orig_dtype = w.dtype
    w_float = w.float()

    qmin = -(2 ** (n_bits - 1))
    qmax = (2 ** (n_bits - 1)) - 1

    max_abs = w_float.abs().amax()

    if max_abs.item() == 0:
        return w_float.to(orig_dtype)

    scale = max_abs / qmax
    q = torch.clamp(torch.round(w_float / scale), qmin, qmax)

    return (q * scale).to(orig_dtype)


@torch.no_grad()
def uniform_quantize_weight_tensor_blockwise_2d(
    w: torch.Tensor,
    n_bits: int,
    block_rows: int = 128,
    block_cols: int = 128,
) -> torch.Tensor:
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")

    if block_rows <= 0 or block_cols <= 0:
        raise ValueError("block_rows and block_cols must be positive")

    if w.ndim != 2:
        return uniform_quantize_weight_tensor(w, n_bits)

    orig_dtype = w.dtype
    w_float = w.float()
    out = torch.empty_like(w_float)

    qmin = -(2 ** (n_bits - 1))
    qmax = (2 ** (n_bits - 1)) - 1

    n_rows, n_cols = w_float.shape

    for r0 in range(0, n_rows, block_rows):
        r1 = min(r0 + block_rows, n_rows)

        for c0 in range(0, n_cols, block_cols):
            c1 = min(c0 + block_cols, n_cols)
            block = w_float[r0:r1, c0:c1]
            max_abs = block.abs().amax()

            if max_abs.item() == 0:
                out[r0:r1, c0:c1] = block
                continue

            scale = max_abs / qmax
            q = torch.clamp(torch.round(block / scale), qmin, qmax)
            out[r0:r1, c0:c1] = q * scale

    return out.to(orig_dtype)


@torch.no_grad()
def quantize_parameter(
    param: torch.Tensor,
    n_bits: int,
    quant_granularity: str = "tensor",
    block_rows: int = 128,
    block_cols: int = 128,
    clip_z: Optional[float] = None,
) -> torch.Tensor:
    w = clip_weight_tensor_zscore(param, clip_z)

    if quant_granularity == "tensor":
        return uniform_quantize_weight_tensor(w, n_bits)

    if quant_granularity == "block2d":
        return uniform_quantize_weight_tensor_blockwise_2d(
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
    for name, param in model.named_parameters():
        if "weight" not in name:
            continue

        if param.ndim < 2:
            continue

        q_weight = quantize_parameter(
            param.data,
            n_bits=n_bits,
            quant_granularity=quant_granularity,
            block_rows=block_rows,
            block_cols=block_cols,
            clip_z=clip_z,
        )

        param.data.copy_(q_weight)

    return model


@torch.no_grad()
def collect_protected_superweights(
    model,
    model_key: str,
    sw_scale: float = 1.0,
    restore_neighborhood: str = "scalar",
):
    if model_key not in SUPERWEIGHTS:
        raise ValueError(f"No superweights registered for model_key='{model_key}'")

    model_cfg = MODEL_CONFIGS[model_key]
    layers = get_nested_attr(model, model_cfg["layer_path"])
    down_proj_path = model_cfg["down_proj_path"]

    protected_values = []

    for entry in SUPERWEIGHTS[model_key]:
        layer_idx = int(entry["layer"])
        row_idx = int(entry["row"])
        col_idx = int(entry["col"])

        module = get_nested_attr(layers[layer_idx], down_proj_path)
        weight = module.weight.data

        restore_indices = get_superweight_restore_indices(
            row=row_idx,
            col=col_idx,
            shape=weight.shape,
            neighborhood=restore_neighborhood,
        )

        for rr, cc in restore_indices:
            is_center = rr == row_idx and cc == col_idx
            value = weight[rr, cc].detach().clone()

            if is_center:
                value = value * sw_scale

            protected_values.append(
                {
                    "layer": layer_idx,
                    "center_row": row_idx,
                    "center_col": col_idx,
                    "row": int(rr),
                    "col": int(cc),
                    "value": value,
                    "is_center": bool(is_center),
                }
            )

    return protected_values


@torch.no_grad()
def restore_protected_superweights(model, model_key: str, protected_values):
    model_cfg = MODEL_CONFIGS[model_key]
    layers = get_nested_attr(model, model_cfg["layer_path"])
    down_proj_path = model_cfg["down_proj_path"]

    for item in protected_values:
        module = get_nested_attr(layers[item["layer"]], down_proj_path)
        module.weight.data[item["row"], item["col"]] = item["value"]

    return model


@torch.no_grad()
def apply_superweight_quantization(
    model,
    model_key: str,
    n_bits: int,
    sw_scale: float = 1.0,
    restore_neighborhood: str = "scalar",
    quant_granularity: str = "tensor",
    block_rows: int = 128,
    block_cols: int = 128,
    clip_z: Optional[float] = None,
):
    protected_values = collect_protected_superweights(
        model=model,
        model_key=model_key,
        sw_scale=sw_scale,
        restore_neighborhood=restore_neighborhood,
    )

    apply_weight_quantization(
        model=model,
        n_bits=n_bits,
        quant_granularity=quant_granularity,
        block_rows=block_rows,
        block_cols=block_cols,
        clip_z=clip_z,
    )

    restore_protected_superweights(
        model=model,
        model_key=model_key,
        protected_values=protected_values,
    )

    return model, protected_values


def prepare_model(
    model,
    model_key: str,
    mode: str,
    bits: int,
    activation_bits: Optional[int] = None,
    sw_scale: float = 1.0,
    restore_neighborhood: str = "scalar",
    quant_granularity: str = "tensor",
    block_rows: int = 128,
    block_cols: int = 128,
    clip_z: Optional[float] = None,
):
    activation_handles = []

    if mode == "fp16":
        return model, [], activation_handles

    if mode == "naive":
        model = apply_weight_quantization(
            model=model,
            n_bits=bits,
            quant_granularity=quant_granularity,
            block_rows=block_rows,
            block_cols=block_cols,
            clip_z=None,
        )

        if activation_bits is not None and activation_bits > 0:
            activation_handles = add_activation_quant_hooks(model, activation_bits)

        return model, [], activation_handles

    if mode == "super":
        model, protected_values = apply_superweight_quantization(
            model=model,
            model_key=model_key,
            n_bits=bits,
            sw_scale=sw_scale,
            restore_neighborhood=restore_neighborhood,
            quant_granularity=quant_granularity,
            block_rows=block_rows,
            block_cols=block_cols,
            clip_z=clip_z,
        )

        if activation_bits is not None and activation_bits > 0:
            activation_handles = add_activation_quant_hooks(model, activation_bits)

        return model, protected_values, activation_handles

    raise ValueError(f"Unsupported mode: {mode}")


def build_prompt(example: Dict) -> str:
    ctx = example.get("ctx", "").strip()
    ctx_a = example.get("ctx_a", "").strip()
    ctx_b = example.get("ctx_b", "").strip()

    if ctx:
        return ctx

    return f"{ctx_a} {ctx_b}".strip()


@torch.no_grad()
def score_continuation(
    model,
    tokenizer,
    prompt: str,
    continuation: str,
    device: torch.device,
    max_length: int = 256,
    normalize_by_length: bool = True,
) -> float:
    prompt = prompt.strip()
    continuation = continuation.strip()
    full_text = prompt + " " + continuation

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    prompt_len = prompt_ids.shape[1]
    full_len = full_ids.shape[1]

    if full_len <= prompt_len:
        return float("-inf")

    input_ids = full_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )

    logits = outputs.logits[:, :-1, :].contiguous()
    target_ids = input_ids[:, 1:].contiguous()

    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    cont_start = max(prompt_len - 1, 0)
    cont_log_probs = token_log_probs[:, cont_start:]

    if cont_log_probs.numel() == 0:
        return float("-inf")

    score = cont_log_probs.sum().item()

    if normalize_by_length:
        score = score / cont_log_probs.numel()

    return float(score)


def load_hellaswag_examples(split: str, limit: Optional[int]) -> List[Dict]:
    candidate_loaders = [
        lambda: load_dataset("hellaswag", split=split),
        lambda: load_dataset("Rowan/hellaswag", split=split),
    ]

    last_error = None
    ds = None

    for loader in candidate_loaders:
        try:
            ds = loader()
            break
        except Exception as e:
            last_error = e

    if ds is None:
        raise RuntimeError(f"Could not load HellaSwag dataset. Last error: {last_error}")

    examples = []

    for i, row in enumerate(ds):
        label = row.get("label", None)

        try:
            label = int(label)
        except Exception:
            continue

        endings = row.get("endings", None)

        if endings is None or len(endings) != 4:
            continue

        if label not in [0, 1, 2, 3]:
            continue

        examples.append(
            {
                "id": row.get("ind", str(i)),
                "ctx": row.get("ctx", ""),
                "ctx_a": row.get("ctx_a", ""),
                "ctx_b": row.get("ctx_b", ""),
                "activity_label": row.get("activity_label", ""),
                "endings": list(endings),
                "label": label,
            }
        )

        if limit is not None and len(examples) >= limit:
            break

    if not examples:
        raise RuntimeError("No valid HellaSwag examples found.")

    return examples


def evaluate_hellaswag(
    model,
    tokenizer,
    examples: List[Dict],
    max_length: int,
    normalize_by_length: bool = True,
) -> Dict:
    device = next(model.parameters()).device

    num_correct = 0
    predictions = []

    model.eval()

    for ex in examples:
        prompt = build_prompt(ex)
        scores = []

        for ending in ex["endings"]:
            score = score_continuation(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                continuation=ending,
                device=device,
                max_length=max_length,
                normalize_by_length=normalize_by_length,
            )
            scores.append(float(score))

        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        gold = int(ex["label"])
        correct = int(pred == gold)

        num_correct += correct

        other_scores = scores[:pred] + scores[pred + 1:]
        margin = scores[pred] - max(other_scores) if other_scores else 0.0

        predictions.append(
            {
                "id": ex["id"],
                "prediction": pred,
                "gold": gold,
                "correct": bool(correct),
                "scores": scores,
                "margin": float(margin),
            }
        )

    total = len(examples)

    if total == 0:
        raise RuntimeError("No HellaSwag examples were scored.")

    return {
        "num_examples": total,
        "accuracy": num_correct / total,
        "num_correct": num_correct,
        "predictions": predictions,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-key", type=str, required=True, choices=sorted(MODEL_CONFIGS.keys()))
    parser.add_argument("--mode", type=str, default="fp16", choices=["fp16", "naive", "super"])
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--activation-bits", type=int, default=None)

    parser.add_argument("--sw-scale", type=float, default=1.0)
    parser.add_argument("--restore-neighborhood", type=str, default="scalar", choices=["scalar", "row", "column", "cross"])

    parser.add_argument("--quant-granularity", type=str, default="tensor", choices=["tensor", "block2d"])
    parser.add_argument("--block-rows", type=int, default=128)
    parser.add_argument("--block-cols", type=int, default=128)
    parser.add_argument("--clip-z", type=float, default=None)

    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--normalize-by-length", action="store_true")
    parser.add_argument("--output-json", type=str, required=True)

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

    print(f"Loading model: {model_id} ({args.dtype})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    print(
        f"Preparing model with mode={args.mode}, bits={args.bits}, "
        f"activation_bits={args.activation_bits}, sw_scale={args.sw_scale}, "
        f"restore_neighborhood={args.restore_neighborhood}, "
        f"quant_granularity={args.quant_granularity}, "
        f"block_rows={args.block_rows}, block_cols={args.block_cols}, "
        f"clip_z={args.clip_z}",
        flush=True,
    )

    model, protected_values, activation_handles = prepare_model(
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

    print(f"Protected/restored values: {len(protected_values)}", flush=True)
    print(f"Activation quant hooks: {len(activation_handles)}", flush=True)

    print(f"Loading HellaSwag split={args.split}, limit={args.limit}", flush=True)
    examples = load_hellaswag_examples(
        split=args.split,
        limit=args.limit,
    )

    print(f"Evaluating {len(examples)} examples...", flush=True)
    metrics = evaluate_hellaswag(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        max_length=args.max_length,
        normalize_by_length=args.normalize_by_length,
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
        for x in protected_values
    ]

    result = {
        "benchmark": "hellaswag",
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
        "normalize_by_length": args.normalize_by_length,
        "num_protected_values": len(protected_values),
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