from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.models import MODEL_CONFIGS
from configs.superweights import SUPERWEIGHTS
from src.hooks import get_nested_attr


# =============================================================================
# REPRODUCIBILITY
# =============================================================================


def set_all_seeds(seed: int) -> None:
    """
    Set random seeds for reproducible evaluation.

    The evaluation subset itself is selected deterministically through
    Hugging Face Dataset.shuffle(seed=seed). PyTorch seeds are set as an
    additional safeguard.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# GENERAL UTILITIES
# =============================================================================


def resolve_torch_dtype(name: str):
    name = name.lower()

    if name == "float16":
        return torch.float16

    if name == "bfloat16":
        return torch.bfloat16

    if name == "float32":
        return torch.float32

    raise ValueError(f"Unsupported dtype: {name}")


def make_json_safe(obj: Any) -> Any:
    """
    Recursively replace non-finite floating-point values with None so that
    the output can be serialized as valid JSON.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    if isinstance(obj, dict):
        return {
            key: make_json_safe(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            make_json_safe(value)
            for value in obj
        ]

    return obj


def calculate_accuracy_stderr(
    accuracy: float,
    num_examples: int,
) -> float:
    """
    Standard error of a binomial accuracy estimate.

    stderr = sqrt(p * (1 - p) / n)
    """
    if num_examples <= 0:
        return float("nan")

    return math.sqrt(
        accuracy * (1.0 - accuracy) / num_examples
    )


# =============================================================================
# SUPERWEIGHT RESTORATION
# =============================================================================


def get_superweight_restore_indices(
    row: int,
    col: int,
    shape,
    neighborhood: str = "scalar",
):
    n_rows, n_cols = shape
    indices = set()

    def add(r: int, c: int) -> None:
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


# =============================================================================
# ACTIVATION QUANTIZATION
# =============================================================================


@torch.no_grad()
def uniform_quantize_activation_tensor(
    x: torch.Tensor,
    n_bits: int,
) -> torch.Tensor:
    if n_bits <= 0:
        raise ValueError("n_bits must be positive.")

    if not torch.is_floating_point(x):
        return x

    original_dtype = x.dtype
    x_float = x.float()

    qmin = -(2 ** (n_bits - 1))
    qmax = (2 ** (n_bits - 1)) - 1

    max_abs = x_float.abs().amax(
        dim=-1,
        keepdim=True,
    )

    scale = max_abs / qmax

    scale = torch.where(
        scale == 0,
        torch.ones_like(scale),
        scale,
    )

    quantized = torch.clamp(
        torch.round(x_float / scale),
        qmin,
        qmax,
    )

    return (quantized * scale).to(original_dtype)


def add_activation_quant_hooks(
    model,
    n_bits: int,
):
    handles = []

    def pre_hook(module, inputs):
        if not inputs:
            return inputs

        activation = inputs[0]

        if not torch.is_tensor(activation):
            return inputs

        quantized_activation = uniform_quantize_activation_tensor(
            activation,
            n_bits,
        )

        return (
            quantized_activation,
        ) + tuple(inputs[1:])

    for module in model.modules():
        if isinstance(module, nn.Linear):
            handle = module.register_forward_pre_hook(pre_hook)
            handles.append(handle)

    return handles


# =============================================================================
# WEIGHT QUANTIZATION
# =============================================================================


@torch.no_grad()
def clip_weight_tensor_zscore(
    weight: torch.Tensor,
    clip_z: Optional[float],
) -> torch.Tensor:
    if clip_z is None or clip_z <= 0:
        return weight

    original_dtype = weight.dtype
    weight_float = weight.float()

    mean = weight_float.mean()
    std = weight_float.std(unbiased=False)

    if std.item() == 0:
        return weight

    lower = mean - clip_z * std
    upper = mean + clip_z * std

    return torch.clamp(
        weight_float,
        lower,
        upper,
    ).to(original_dtype)


@torch.no_grad()
def uniform_quantize_weight_tensor(
    weight: torch.Tensor,
    n_bits: int,
) -> torch.Tensor:
    """
    Asymmetric per-tensor round-to-nearest quantization.
    """
    if n_bits <= 0:
        raise ValueError("n_bits must be positive.")

    original_dtype = weight.dtype
    weight_float = weight.float()

    qmin = 0
    qmax = (2 ** n_bits) - 1

    weight_min = weight_float.min()
    weight_max = weight_float.max()

    if (weight_max - weight_min).item() == 0:
        return weight_float.to(original_dtype)

    delta = (weight_max - weight_min) / qmax

    quantized = torch.clamp(
        torch.round(
            (weight_float - weight_min) / delta
        ),
        qmin,
        qmax,
    )

    dequantized = quantized * delta + weight_min

    return dequantized.to(original_dtype)


@torch.no_grad()
def uniform_quantize_weight_tensor_blockwise_2d(
    weight: torch.Tensor,
    n_bits: int,
    block_rows: int = 128,
    block_cols: int = 128,
) -> torch.Tensor:
    """
    Asymmetric block-wise 2D round-to-nearest quantization.
    """
    if n_bits <= 0:
        raise ValueError("n_bits must be positive.")

    if block_rows <= 0 or block_cols <= 0:
        raise ValueError(
            "block_rows and block_cols must be positive."
        )

    if weight.ndim != 2:
        return uniform_quantize_weight_tensor(
            weight,
            n_bits,
        )

    original_dtype = weight.dtype
    weight_float = weight.float()
    output = torch.empty_like(weight_float)

    qmin = 0
    qmax = (2 ** n_bits) - 1

    num_rows, num_cols = weight_float.shape

    for row_start in range(0, num_rows, block_rows):
        row_end = min(
            row_start + block_rows,
            num_rows,
        )

        for col_start in range(0, num_cols, block_cols):
            col_end = min(
                col_start + block_cols,
                num_cols,
            )

            block = weight_float[
                row_start:row_end,
                col_start:col_end,
            ]

            block_min = block.min()
            block_max = block.max()

            if (block_max - block_min).item() == 0:
                output[
                    row_start:row_end,
                    col_start:col_end,
                ] = block
                continue

            delta = (block_max - block_min) / qmax

            quantized = torch.clamp(
                torch.round(
                    (block - block_min) / delta
                ),
                qmin,
                qmax,
            )

            output[
                row_start:row_end,
                col_start:col_end,
            ] = quantized * delta + block_min

    return output.to(original_dtype)


@torch.no_grad()
def quantize_parameter(
    parameter: torch.Tensor,
    n_bits: int,
    quant_granularity: str = "tensor",
    block_rows: int = 128,
    block_cols: int = 128,
    clip_z: Optional[float] = None,
) -> torch.Tensor:
    clipped_parameter = clip_weight_tensor_zscore(
        parameter,
        clip_z,
    )

    if quant_granularity == "tensor":
        return uniform_quantize_weight_tensor(
            clipped_parameter,
            n_bits,
        )

    if quant_granularity == "block2d":
        return uniform_quantize_weight_tensor_blockwise_2d(
            clipped_parameter,
            n_bits=n_bits,
            block_rows=block_rows,
            block_cols=block_cols,
        )

    raise ValueError(
        f"Unsupported quant_granularity: "
        f"{quant_granularity}"
    )


@torch.no_grad()
def apply_weight_quantization(
    model,
    n_bits: int,
    quant_granularity: str = "tensor",
    block_rows: int = 128,
    block_cols: int = 128,
    clip_z: Optional[float] = None,
):
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

        quantized_weight = quantize_parameter(
            module.weight.data,
            n_bits=n_bits,
            quant_granularity=quant_granularity,
            block_rows=block_rows,
            block_cols=block_cols,
            clip_z=clip_z,
        )

        module.weight.data.copy_(quantized_weight)
        quantized_modules += 1

    print(
        "Weight quantization completed: "
        f"quantized Linear modules={quantized_modules}, "
        f"skipped Linear modules={skipped_modules}",
        flush=True,
    )

    return model


# =============================================================================
# SUPERWEIGHT PROTECTION
# =============================================================================


@torch.no_grad()
def collect_protected_superweights(
    model,
    model_key: str,
    sw_scale: float = 1.0,
    restore_neighborhood: str = "scalar",
):
    if model_key not in SUPERWEIGHTS:
        raise ValueError(
            f"No superweights registered for "
            f"model_key={model_key!r}."
        )

    model_config = MODEL_CONFIGS[model_key]
    layers = get_nested_attr(
        model,
        model_config["layer_path"],
    )

    down_proj_path = model_config["down_proj_path"]

    protected_values = []

    for entry in SUPERWEIGHTS[model_key]:
        layer_index = int(entry["layer"])
        row_index = int(entry["row"])
        col_index = int(entry["col"])

        module = get_nested_attr(
            layers[layer_index],
            down_proj_path,
        )

        weight = module.weight.data

        if not (
            0 <= row_index < weight.shape[0]
            and 0 <= col_index < weight.shape[1]
        ):
            raise IndexError(
                "Invalid superweight coordinate: "
                f"layer={layer_index}, "
                f"row={row_index}, "
                f"col={col_index}, "
                f"shape={tuple(weight.shape)}."
            )

        restore_indices = get_superweight_restore_indices(
            row=row_index,
            col=col_index,
            shape=weight.shape,
            neighborhood=restore_neighborhood,
        )

        for restored_row, restored_col in restore_indices:
            is_center = (
                restored_row == row_index
                and restored_col == col_index
            )

            value = (
                weight[restored_row, restored_col]
                .detach()
                .clone()
            )

            if is_center:
                value = value * sw_scale

            protected_values.append(
                {
                    "layer": layer_index,
                    "center_row": row_index,
                    "center_col": col_index,
                    "row": int(restored_row),
                    "col": int(restored_col),
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
    model_config = MODEL_CONFIGS[model_key]

    layers = get_nested_attr(
        model,
        model_config["layer_path"],
    )

    down_proj_path = model_config["down_proj_path"]

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
            activation_handles = add_activation_quant_hooks(
                model,
                activation_bits,
            )

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
            activation_handles = add_activation_quant_hooks(
                model,
                activation_bits,
            )

        return (
            model,
            protected_values,
            activation_handles,
        )

    raise ValueError(f"Unsupported mode: {mode}")


# =============================================================================
# BOOLQ DATASET
# =============================================================================


def build_prompt(example: Dict) -> str:
    passage = example["passage"].strip()
    question = example["question"].strip()

    return (
        f"{passage}\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def load_boolq_examples(
    split: str,
    limit: Optional[int],
    eval_seed: int,
) -> tuple[List[Dict], int]:
    """
    Load and deterministically shuffle BoolQ.

    The original dataset index is stored before shuffling. Consequently,
    FP16, Naive W4, and Super W4 use exactly the same examples whenever
    split, limit, and eval_seed are identical.

    Returns
    -------
    examples:
        Selected BoolQ examples after deterministic shuffling.
    total_dataset_size:
        Number of examples available before applying the limit.
    """
    dataset = load_dataset(
        "super_glue",
        "boolq",
        split=split,
    )

    total_dataset_size = len(dataset)

    if total_dataset_size == 0:
        raise RuntimeError(
            f"BoolQ split {split!r} is empty."
        )

    # Preserve the original dataset index before shuffling.
    dataset = dataset.map(
        lambda example, index: {
            "original_index": index,
        },
        with_indices=True,
    )

    # Deterministic but different sample order for every seed.
    dataset = dataset.shuffle(seed=eval_seed)

    if limit is not None:
        if limit <= 0:
            raise ValueError(
                "--limit must be greater than zero."
            )

        effective_limit = min(
            limit,
            len(dataset),
        )

        dataset = dataset.select(
            range(effective_limit)
        )

    examples: List[Dict] = []

    for row in dataset:
        examples.append(
            {
                "id": int(row["original_index"]),
                "passage": row["passage"],
                "question": row["question"],
                "label": int(row["label"]),
            }
        )

    if not examples:
        raise RuntimeError(
            "No valid BoolQ examples were selected."
        )

    return examples, total_dataset_size


# =============================================================================
# BOOLQ SCORING
# =============================================================================


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

    prompt_length = prompt_ids.shape[1]
    full_length = full_ids.shape[1]

    if full_length <= prompt_length:
        return float("-inf")

    input_ids = full_ids.to(device)

    attention_mask = torch.ones_like(
        input_ids,
        device=device,
    )

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )

    logits = outputs.logits[:, :-1, :].contiguous()
    target_ids = input_ids[:, 1:].contiguous()

    log_probabilities = F.log_softmax(
        logits.float(),
        dim=-1,
    )

    token_log_probabilities = log_probabilities.gather(
        -1,
        target_ids.unsqueeze(-1),
    ).squeeze(-1)

    answer_start = max(
        prompt_length - 1,
        0,
    )

    answer_log_probabilities = token_log_probabilities[
        :,
        answer_start:,
    ]

    if answer_log_probabilities.numel() == 0:
        return float("-inf")

    score = answer_log_probabilities.sum().item()

    if normalize_by_length:
        score = (
            score
            / answer_log_probabilities.numel()
        )

    return float(score)


def evaluate_boolq(
    model,
    tokenizer,
    examples: List[Dict],
    max_length: int,
    normalize_by_length: bool = False,
) -> Dict:
    device = next(model.parameters()).device

    number_correct = 0
    predictions = []

    model.eval()

    total_examples = len(examples)

    for example_number, example in enumerate(
        examples,
        start=1,
    ):
        prompt = build_prompt(example)

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

        prediction = (
            1
            if score_yes > score_no
            else 0
        )

        gold = int(example["label"])
        correct = int(prediction == gold)

        number_correct += correct

        predictions.append(
            {
                "id": int(example["id"]),
                "prediction": (
                    "yes"
                    if prediction == 1
                    else "no"
                ),
                "gold": (
                    "yes"
                    if gold == 1
                    else "no"
                ),
                "correct": bool(correct),
                "score_yes": float(score_yes),
                "score_no": float(score_no),
                "margin": float(
                    score_yes - score_no
                ),
            }
        )

        if (
            example_number % 50 == 0
            or example_number == total_examples
        ):
            running_accuracy = (
                number_correct / example_number
            )

            print(
                f"Progress: "
                f"{example_number}/{total_examples} | "
                f"running accuracy="
                f"{running_accuracy:.4f}",
                flush=True,
            )

    if total_examples == 0:
        raise RuntimeError(
            "No BoolQ examples were scored."
        )

    accuracy = number_correct / total_examples

    accuracy_stderr = calculate_accuracy_stderr(
        accuracy=accuracy,
        num_examples=total_examples,
    )

    return {
        "num_examples": total_examples,
        "num_correct": number_correct,
        "accuracy": accuracy,
        "accuracy_stderr": accuracy_stderr,
        "predictions": predictions,
    }


# =============================================================================
# ARGUMENTS
# =============================================================================


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate FP16, naive quantization, or superweight-aware "
            "quantization on reproducibly sampled BoolQ examples."
        )
    )

    parser.add_argument(
        "--model-key",
        required=True,
        choices=sorted(MODEL_CONFIGS.keys()),
    )

    parser.add_argument(
        "--mode",
        default="fp16",
        choices=[
            "fp16",
            "naive",
            "super",
        ],
    )

    parser.add_argument(
        "--bits",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--activation-bits",
        type=int,
        default=None,
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
        "--clip-z",
        type=float,
        default=None,
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
        default=500,
    )

    parser.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help=(
            "Seed used to shuffle BoolQ before applying --limit. "
            "Use the same seed for all compared methods."
        ),
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


# =============================================================================
# MAIN
# =============================================================================


def main():
    args = build_arg_parser().parse_args()

    if args.bits <= 0:
        raise ValueError(
            "--bits must be greater than zero."
        )

    if (
        args.activation_bits is not None
        and args.activation_bits <= 0
    ):
        raise ValueError(
            "--activation-bits must be greater than zero "
            "when provided."
        )

    if args.max_length <= 0:
        raise ValueError(
            "--max-length must be greater than zero."
        )

    set_all_seeds(args.eval_seed)

    model_config = MODEL_CONFIGS[args.model_key]
    model_id = model_config["hf_name"]

    torch_dtype = resolve_torch_dtype(
        args.dtype
    )

    print("=" * 78)
    print("BOOLQ QUANTIZATION EVALUATION")
    print("=" * 78)
    print(f"Model key:               {args.model_key}")
    print(f"Model ID:                {model_id}")
    print(f"Mode:                    {args.mode}")
    print(f"Weight bits:             {args.bits}")
    print(f"Activation bits:         {args.activation_bits}")
    print(f"Quantization granularity:{args.quant_granularity}")
    print(f"Block rows:              {args.block_rows}")
    print(f"Block columns:           {args.block_cols}")
    print(f"Clip z-score:            {args.clip_z}")
    print(f"Superweight scale:       {args.sw_scale}")
    print(f"Restore neighborhood:    {args.restore_neighborhood}")
    print(f"Evaluation split:        {args.split}")
    print(f"Evaluation limit:        {args.limit}")
    print(f"Evaluation seed:         {args.eval_seed}")
    print(f"Maximum sequence length: {args.max_length}")
    print(f"Normalize by length:     {args.normalize_by_length}")
    print(f"Data type:               {args.dtype}")
    print("=" * 78)

    print(
        f"Loading tokenizer: {model_id}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    if (
        tokenizer.pad_token is None
        and tokenizer.eos_token is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token_id is None:
        raise ValueError(
            "Tokenizer has no pad token and no usable EOS token."
        )

    print(
        f"Loading model: "
        f"{model_id} ({args.dtype})",
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=(
            "auto"
            if torch.cuda.is_available()
            else None
        ),
    )

    model.eval()

    print(
        f"Preparing model with mode={args.mode}, "
        f"bits={args.bits}, "
        f"activation_bits={args.activation_bits}, "
        f"sw_scale={args.sw_scale}, "
        f"restore_neighborhood="
        f"{args.restore_neighborhood}, "
        f"quant_granularity="
        f"{args.quant_granularity}, "
        f"block_rows={args.block_rows}, "
        f"block_cols={args.block_cols}, "
        f"clip_z={args.clip_z}",
        flush=True,
    )

    (
        model,
        protected_values,
        activation_handles,
    ) = prepare_model(
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

    print(
        f"Protected/restored values: "
        f"{len(protected_values)}",
        flush=True,
    )

    print(
        f"Activation quantization hooks: "
        f"{len(activation_handles)}",
        flush=True,
    )

    print(
        f"Loading BoolQ split={args.split}, "
        f"limit={args.limit}, "
        f"eval_seed={args.eval_seed}",
        flush=True,
    )

    examples, total_dataset_size = load_boolq_examples(
        split=args.split,
        limit=args.limit,
        eval_seed=args.eval_seed,
    )

    selected_example_ids = [
        int(example["id"])
        for example in examples
    ]

    print(
        f"Available BoolQ examples: "
        f"{total_dataset_size}",
        flush=True,
    )

    print(
        f"Selected BoolQ examples: "
        f"{len(examples)}",
        flush=True,
    )

    print(
        "First 10 selected original indices: "
        f"{selected_example_ids[:10]}",
        flush=True,
    )

    print(
        f"Evaluating {len(examples)} examples...",
        flush=True,
    )

    metrics = evaluate_boolq(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        max_length=args.max_length,
        normalize_by_length=args.normalize_by_length,
    )

    protected_summary = [
        {
            "layer": int(item["layer"]),
            "center_row": int(item["center_row"]),
            "center_col": int(item["center_col"]),
            "row": int(item["row"]),
            "col": int(item["col"]),
            "is_center": bool(item["is_center"]),
        }
        for item in protected_values
    ]

    result = {
        "benchmark": "boolq",
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
        "num_activation_hooks": len(
            activation_handles
        ),
        "dtype": args.dtype,
        "split": args.split,
        "dataset_size": total_dataset_size,
        "limit": args.limit,
        "eval_seed": args.eval_seed,
        "selected_example_ids": selected_example_ids,
        "max_length": args.max_length,
        **metrics,
    }

    result = make_json_safe(result)

    output_path = Path(args.output_json)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            result,
            output_file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 78)
    print("BOOLQ RESULT")
    print("=" * 78)
    print(f"Mode:             {args.mode}")
    print(f"Evaluation seed:  {args.eval_seed}")
    print(f"Examples:         {metrics['num_examples']}")
    print(f"Correct:          {metrics['num_correct']}")
    print(f"Accuracy:         {metrics['accuracy']:.4f}")
    print(
        f"Accuracy stderr:  "
        f"{metrics['accuracy_stderr']:.4f}"
    )
    print(f"Saved result to:  {output_path}")
    print("=" * 78)

    # Remove hooks cleanly before program exit.
    for handle in activation_handles:
        handle.remove()


if __name__ == "__main__":
    main()