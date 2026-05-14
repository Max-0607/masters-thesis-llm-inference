import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
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


def get_superweight_restore_indices(
    row: int,
    col: int,
    shape,
    neighborhood: str = "scalar",
):
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
def uniform_quantize_weight_tensor(
    w: torch.Tensor,
    n_bits: int,
) -> torch.Tensor:
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

    q = torch.clamp(
        torch.round(w_float / scale),
        qmin,
        qmax,
    )

    dq = q * scale

    return dq.to(orig_dtype)


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

            q = torch.clamp(
                torch.round(block / scale),
                qmin,
                qmax,
            )

            out[r0:r1, c0:c1] = q * scale

    return out.to(orig_dtype)


@torch.no_grad()
def quantize_parameter(
    param: torch.Tensor,
    n_bits: int,
    quant_granularity: str = "tensor",
    block_rows: int = 128,
    block_cols: int = 128,
) -> torch.Tensor:
    if quant_granularity == "tensor":
        return uniform_quantize_weight_tensor(param, n_bits)

    if quant_granularity == "block2d":
        return uniform_quantize_weight_tensor_blockwise_2d(
            param,
            n_bits=n_bits,
            block_rows=block_rows,
            block_cols=block_cols,
        )

    raise ValueError(
        f"Unsupported quant_granularity: {quant_granularity}"
    )


@torch.no_grad()
def apply_naive_weight_quantization(
    model,
    n_bits: int,
    quant_granularity: str = "tensor",
    block_rows: int = 128,
    block_cols: int = 128,
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
        raise ValueError(
            f"No superweights registered for model_key='{model_key}'"
        )

    model_cfg = MODEL_CONFIGS[model_key]

    layers = get_nested_attr(
        model,
        model_cfg["layer_path"],
    )

    down_proj_path = model_cfg["down_proj_path"]

    protected_values = []

    for entry in SUPERWEIGHTS[model_key]:

        layer_idx = int(entry["layer"])
        row_idx = int(entry["row"])
        col_idx = int(entry["col"])

        module = get_nested_attr(
            layers[layer_idx],
            down_proj_path,
        )

        weight = module.weight.data

        restore_indices = get_superweight_restore_indices(
            row=row_idx,
            col=col_idx,
            shape=weight.shape,
            neighborhood=restore_neighborhood,
        )

        for rr, cc in restore_indices:

            is_center = (
                rr == row_idx and cc == col_idx
            )

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
def restore_protected_superweights(
    model,
    model_key: str,
    protected_values,
):
    model_cfg = MODEL_CONFIGS[model_key]

    layers = get_nested_attr(
        model,
        model_cfg["layer_path"],
    )

    down_proj_path = model_cfg["down_proj_path"]

    for item in protected_values:

        module = get_nested_attr(
            layers[item["layer"]],
            down_proj_path,
        )

        module.weight.data[
            item["row"],
            item["col"],
        ] = item["value"]

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
):
    protected_values = collect_protected_superweights(
        model=model,
        model_key=model_key,
        sw_scale=sw_scale,
        restore_neighborhood=restore_neighborhood,
    )

    apply_naive_weight_quantization(
        model=model,
        n_bits=n_bits,
        quant_granularity=quant_granularity,
        block_rows=block_rows,
        block_cols=block_cols,
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
    sw_scale: float = 1.0,
    restore_neighborhood: str = "scalar",
    quant_granularity: str = "tensor",
    block_rows: int = 128,
    block_cols: int = 128,
):
    if mode == "fp16":
        return model, []

    if mode == "naive":

        model = apply_naive_weight_quantization(
            model=model,
            n_bits=bits,
            quant_granularity=quant_granularity,
            block_rows=block_rows,
            block_cols=block_cols,
        )

        return model, []

    if mode == "super":

        return apply_superweight_quantization(
            model=model,
            model_key=model_key,
            n_bits=bits,
            sw_scale=sw_scale,
            restore_neighborhood=restore_neighborhood,
            quant_granularity=quant_granularity,
            block_rows=block_rows,
            block_cols=block_cols,
        )

    raise ValueError(f"Unsupported mode: {mode}")


def build_prompt(example: Dict) -> str:
    passage = example["passage"].strip()
    question = example["question"].strip()

    return (
        f"{passage}\n"
        f"Question: {question}\n"
        f"Answer:"
    )


@torch.no_grad()
def score_answer(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    device: torch.device,
    max_length: int = 512,
    normalize_by_length: bool = False,
) -> float:
    prompt = prompt.strip()
    answer = answer.strip()

    full_text = prompt + " " + answer

    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    full_ids = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    prompt_len = prompt_ids.shape[1]
    full_len = full_ids.shape[1]

    if full_len <= prompt_len:
        return float("-inf")

    input_ids = full_ids.to(device)

    outputs = model(
        input_ids=input_ids,
        use_cache=False,
        return_dict=True,
    )

    logits = outputs.logits[:, :-1, :].contiguous()

    target_ids = input_ids[:, 1:].contiguous()

    log_probs = F.log_softmax(
        logits.float(),
        dim=-1,
    )

    token_log_probs = log_probs.gather(
        -1,
        target_ids.unsqueeze(-1),
    ).squeeze(-1)

    answer_start = max(prompt_len - 1, 0)

    answer_log_probs = token_log_probs[
        :,
        answer_start:
    ]

    if answer_log_probs.numel() == 0:
        return float("-inf")

    score = answer_log_probs.sum().item()

    if normalize_by_length:
        score = (
            score /
            answer_log_probs.numel()
        )

    return float(score)


def load_boolq_examples(
    split: str,
    limit: Optional[int],
) -> List[Dict]:
    ds = load_dataset(
        "super_glue",
        "boolq",
        split=split,
    )

    examples = []

    for i, row in enumerate(ds):

        examples.append(
            {
                "id": i,
                "passage": row["passage"],
                "question": row["question"],
                "label": int(row["label"]),
            }
        )

        if (
            limit is not None
            and len(examples) >= limit
        ):
            break

    if not examples:
        raise RuntimeError(
            "No valid BoolQ examples found."
        )

    return examples


def evaluate_boolq(
    model,
    tokenizer,
    examples: List[Dict],
    max_length: int,
    normalize_by_length: bool = False,
) -> Dict:
    device = next(model.parameters()).device

    correct = 0
    predictions = []

    model.eval()

    for ex in examples:

        prompt = build_prompt(ex)

        score_yes = score_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            answer="yes",
            device=device,
            max_length=max_length,
            normalize_by_length=normalize_by_length,
        )

        score_no = score_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            answer="no",
            device=device,
            max_length=max_length,
            normalize_by_length=normalize_by_length,
        )

        pred = (
            1 if score_yes > score_no else 0
        )

        gold = ex["label"]

        is_correct = int(pred == gold)

        correct += is_correct

        predictions.append(
            {
                "id": ex["id"],
                "prediction": (
                    "yes"
                    if pred == 1
                    else "no"
                ),
                "gold": (
                    "yes"
                    if gold == 1
                    else "no"
                ),
                "correct": bool(is_correct),
                "score_yes": float(score_yes),
                "score_no": float(score_no),
                "margin": float(
                    score_yes - score_no
                ),
            }
        )

    total = len(examples)

    return {
        "num_examples": total,
        "num_correct": correct,
        "accuracy": (
            correct / total
            if total > 0
            else 0.0
        ),
        "predictions": predictions,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-key",
        required=True,
        choices=sorted(MODEL_CONFIGS.keys()),
    )

    parser.add_argument(
        "--mode",
        default="fp16",
        choices=["fp16", "naive", "super"],
    )

    parser.add_argument(
        "--bits",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--sw-scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--restore-neighborhood",
        type=str,
        default="scalar",
        choices=[
            "scalar",
            "row",
            "column",
            "cross",
        ],
    )

    parser.add_argument(
        "--quant-granularity",
        type=str,
        default="tensor",
        choices=[
            "tensor",
            "block2d",
        ],
    )

    parser.add_argument(
        "--block-rows",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--block-cols",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--dtype",
        default="float16",
        choices=[
            "float16",
            "bfloat16",
            "float32",
        ],
    )

    parser.add_argument(
        "--split",
        default="validation",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--normalize-by-length",
        action="store_true",
    )

    parser.add_argument(
        "--output-json",
        required=True,
    )

    return parser


def main():
    parser = build_arg_parser()

    args = parser.parse_args()

    model_cfg = MODEL_CONFIGS[
        args.model_key
    ]

    model_id = model_cfg["hf_name"]

    torch_dtype = resolve_torch_dtype(
        args.dtype
    )

    print(f"Loading tokenizer: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    if (
        tokenizer.pad_token is None
        and tokenizer.eos_token is not None
    ):
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    print(
        f"Loading model: {model_id} "
        f"({args.dtype})"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    model.eval()

    print(
        f"Preparing model with "
        f"mode={args.mode}, "
        f"bits={args.bits}, "
        f"sw_scale={args.sw_scale}, "
        f"restore_neighborhood="
        f"{args.restore_neighborhood}, "
        f"quant_granularity="
        f"{args.quant_granularity}, "
        f"block_rows={args.block_rows}, "
        f"block_cols={args.block_cols}"
    )

    model, protected_values = prepare_model(
        model=model,
        model_key=args.model_key,
        mode=args.mode,
        bits=args.bits,
        sw_scale=args.sw_scale,
        restore_neighborhood=(
            args.restore_neighborhood
        ),
        quant_granularity=(
            args.quant_granularity
        ),
        block_rows=args.block_rows,
        block_cols=args.block_cols,
    )

    print(
        f"Protected/restored values: "
        f"{len(protected_values)}"
    )

    print(
        f"Loading BoolQ "
        f"split={args.split}, "
        f"limit={args.limit}"
    )

    examples = load_boolq_examples(
        args.split,
        args.limit,
    )

    metrics = evaluate_boolq(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        max_length=args.max_length,
        normalize_by_length=(
            args.normalize_by_length
        ),
    )

    protected_summary = [
        {
            "layer": int(x["layer"]),
            "center_row": int(
                x["center_row"]
            ),
            "center_col": int(
                x["center_col"]
            ),
            "row": int(x["row"]),
            "col": int(x["col"]),
            "is_center": bool(
                x["is_center"]
            ),
        }
        for x in protected_values
    ]

    result = {
        "benchmark": "boolq",
        "model_key": args.model_key,
        "model_id": model_id,
        "mode": args.mode,
        "bits": args.bits,
        "sw_scale": args.sw_scale,
        "restore_neighborhood": (
            args.restore_neighborhood
        ),
        "quant_granularity": (
            args.quant_granularity
        ),
        "block_rows": args.block_rows,
        "block_cols": args.block_cols,
        "normalize_by_length": (
            args.normalize_by_length
        ),
        "num_protected_values": (
            len(protected_values)
        ),
        "protected_values": (
            protected_summary
        ),
        "dtype": args.dtype,
        "split": args.split,
        "limit": args.limit,
        "max_length": args.max_length,
        **metrics,
    }

    output_path = Path(args.output_json)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()