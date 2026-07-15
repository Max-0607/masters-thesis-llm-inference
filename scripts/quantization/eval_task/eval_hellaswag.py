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
# REPRODUCIBILITY AND GENERAL UTILITIES
# =============================================================================


def set_all_seeds(seed: int) -> None:
    """Set Python and PyTorch seeds for reproducible evaluation."""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_torch_dtype(name: str) -> torch.dtype:
    """Convert a dtype name into a PyTorch dtype."""
    name = name.lower()

    if name == "float16":
        return torch.float16

    if name == "bfloat16":
        return torch.bfloat16

    if name == "float32":
        return torch.float32

    raise ValueError(f"Unsupported dtype: {name}")


def make_json_safe(obj: Any) -> Any:
    """Replace non-finite floating-point values with None."""
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
    Calculate the standard error of a binary accuracy estimate.

    SE = sqrt(p * (1 - p) / n)
    """
    if num_examples <= 0:
        raise ValueError(
            "num_examples must be greater than zero."
        )

    return math.sqrt(
        accuracy * (1.0 - accuracy) / num_examples
    )


# =============================================================================
# SUPERWEIGHT RESTORATION UTILITIES
# =============================================================================


def get_superweight_restore_indices(
    row: int,
    col: int,
    shape,
    neighborhood: str = "scalar",
):
    """
    Return the superweight coordinate and optionally adjacent values.
    """
    n_rows, n_cols = shape
    indices = set()

    def add(r: int, c: int) -> None:
        if 0 <= r < n_rows and 0 <= c < n_cols:
            indices.add((r, c))

    add(row, col)

    if neighborhood in {"row", "cross"}:
        add(row, col - 1)
        add(row, col + 1)

    if neighborhood in {"column", "cross"}:
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
    """
    Apply symmetric per-token activation quantization and dequantization.
    """
    if n_bits <= 0:
        raise ValueError(
            "n_bits must be positive."
        )

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
    """Quantize the input activations of every linear layer."""
    handles = []

    def pre_hook(module, inputs):
        if not inputs:
            return inputs

        activation = inputs[0]

        if not torch.is_tensor(activation):
            return inputs

        quantized_activation = (
            uniform_quantize_activation_tensor(
                activation,
                n_bits,
            )
        )

        return (
            quantized_activation,
        ) + tuple(inputs[1:])

    for module in model.modules():
        if isinstance(module, nn.Linear):
            handles.append(
                module.register_forward_pre_hook(
                    pre_hook
                )
            )

    return handles


# =============================================================================
# WEIGHT QUANTIZATION
# =============================================================================


@torch.no_grad()
def clip_weight_tensor_zscore(
    weight: torch.Tensor,
    clip_z: Optional[float],
) -> torch.Tensor:
    """Clip a tensor based on a global z-score threshold."""
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
    Apply asymmetric per-tensor min-max RTN quantization.

    Q(W) = round((W - min(W)) / delta)
    W_q = Q(W) * delta + min(W)
    """
    if n_bits <= 0:
        raise ValueError(
            "n_bits must be positive."
        )

    original_dtype = weight.dtype
    weight_float = weight.float()

    qmin = 0
    qmax = (2 ** n_bits) - 1

    weight_min = weight_float.min()
    weight_max = weight_float.max()
    value_range = weight_max - weight_min

    if value_range.item() == 0:
        return weight_float.to(original_dtype)

    delta = value_range / qmax

    quantized = torch.clamp(
        torch.round(
            (weight_float - weight_min) / delta
        ),
        qmin,
        qmax,
    )

    dequantized = (
        quantized * delta
        + weight_min
    )

    return dequantized.to(original_dtype)


@torch.no_grad()
def uniform_quantize_weight_tensor_blockwise_2d(
    weight: torch.Tensor,
    n_bits: int,
    block_rows: int = 128,
    block_cols: int = 128,
) -> torch.Tensor:
    """
    Apply asymmetric 2D block-wise min-max RTN quantization.
    """
    if n_bits <= 0:
        raise ValueError(
            "n_bits must be positive."
        )

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

    n_rows, n_cols = weight_float.shape

    for row_start in range(0, n_rows, block_rows):
        row_end = min(
            row_start + block_rows,
            n_rows,
        )

        for col_start in range(0, n_cols, block_cols):
            col_end = min(
                col_start + block_cols,
                n_cols,
            )

            block = weight_float[
                row_start:row_end,
                col_start:col_end,
            ]

            block_min = block.min()
            block_max = block.max()
            block_range = block_max - block_min

            if block_range.item() == 0:
                output[
                    row_start:row_end,
                    col_start:col_end,
                ] = block
                continue

            delta = block_range / qmax

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
            ] = (
                quantized * delta
                + block_min
            )

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
    """Clip and quantize a parameter tensor."""
    clipped_weight = clip_weight_tensor_zscore(
        parameter,
        clip_z,
    )

    if quant_granularity == "tensor":
        return uniform_quantize_weight_tensor(
            clipped_weight,
            n_bits,
        )

    if quant_granularity == "block2d":
        return (
            uniform_quantize_weight_tensor_blockwise_2d(
                clipped_weight,
                n_bits=n_bits,
                block_rows=block_rows,
                block_cols=block_cols,
            )
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
    """Quantize all linear layers except the language-model head."""
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

        module.weight.data.copy_(
            quantized_weight
        )

        quantized_modules += 1

    print(
        "Weight quantization completed: "
        f"quantized linear modules={quantized_modules}, "
        f"skipped linear modules={skipped_modules}",
        flush=True,
    )

    return model


# =============================================================================
# SUPERWEIGHT-AWARE WEIGHT QUANTIZATION
# =============================================================================


@torch.no_grad()
def collect_protected_superweights(
    model,
    model_key: str,
    sw_scale: float = 1.0,
    restore_neighborhood: str = "scalar",
):
    """Store the selected superweights before quantization."""
    if model_key not in SUPERWEIGHTS:
        raise ValueError(
            "No superweights registered for "
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

        n_rows, n_cols = weight.shape

        if not (
            0 <= row_index < n_rows
            and 0 <= col_index < n_cols
        ):
            raise IndexError(
                "Invalid superweight coordinate: "
                f"model={model_key}, "
                f"layer={layer_index}, "
                f"row={row_index}, "
                f"col={col_index}, "
                f"shape={tuple(weight.shape)}."
            )

        restore_indices = (
            get_superweight_restore_indices(
                row=row_index,
                col=col_index,
                shape=weight.shape,
                neighborhood=restore_neighborhood,
            )
        )

        for restored_row, restored_col in restore_indices:
            is_center = (
                restored_row == row_index
                and restored_col == col_index
            )

            value = (
                weight[
                    restored_row,
                    restored_col,
                ]
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
    """Restore protected weights after quantization."""
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
    """
    Store superweights, quantize the model, and restore the stored values.
    """
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
    """Prepare FP16, naive W4, or Super W4 model variants."""
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

        if (
            activation_bits is not None
            and activation_bits > 0
        ):
            activation_handles = add_activation_quant_hooks(
                model,
                activation_bits,
            )

        return model, [], activation_handles

    if mode == "super":
        model, protected_values = (
            apply_superweight_quantization(
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
        )

        if (
            activation_bits is not None
            and activation_bits > 0
        ):
            activation_handles = add_activation_quant_hooks(
                model,
                activation_bits,
            )

        return (
            model,
            protected_values,
            activation_handles,
        )

    raise ValueError(
        f"Unsupported mode: {mode}"
    )


# =============================================================================
# HELLASWAG DATASET
# =============================================================================


def load_hellaswag_dataset(split: str):
    """
    Load HellaSwag while supporting the dataset names used by different
    versions of the datasets library.
    """
    candidate_loaders = [
        lambda: load_dataset(
            "hellaswag",
            split=split,
        ),
        lambda: load_dataset(
            "Rowan/hellaswag",
            split=split,
        ),
    ]

    last_error = None

    for loader in candidate_loaders:
        try:
            return loader()
        except Exception as error:
            last_error = error

    raise RuntimeError(
        "Could not load the HellaSwag dataset. "
        f"Last error: {last_error}"
    )


def load_hellaswag_examples(
    split: str,
    limit: Optional[int],
    eval_seed: int,
) -> tuple[List[Dict], int]:
    """
    Load, validate, and reproducibly shuffle HellaSwag examples.

    The original dataset index is stored before shuffling. Therefore,
    FP16, Naive W4, Super W4, GPTQ, and AWQ can use the same examples
    whenever split, limit, and eval_seed are identical.
    """
    dataset = load_hellaswag_dataset(
        split=split,
    )

    total_available = len(dataset)

    if total_available == 0:
        raise RuntimeError(
            f"HellaSwag split {split!r} is empty."
        )

    dataset = dataset.add_column(
        "_original_index",
        list(range(total_available)),
    )

    dataset = dataset.shuffle(
        seed=eval_seed,
    )

    examples: List[Dict] = []

    for row in dataset:
        try:
            label = int(row["label"])
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        endings = row.get(
            "endings",
            None,
        )

        if endings is None or len(endings) != 4:
            continue

        if label not in {0, 1, 2, 3}:
            continue

        original_index = int(
            row["_original_index"]
        )

        examples.append(
            {
                "id": original_index,
                "dataset_id": str(
                    row.get(
                        "ind",
                        original_index,
                    )
                ),
                "ctx": row.get("ctx", ""),
                "ctx_a": row.get("ctx_a", ""),
                "ctx_b": row.get("ctx_b", ""),
                "activity_label": row.get(
                    "activity_label",
                    "",
                ),
                "endings": list(endings),
                "label": label,
            }
        )

        if (
            limit is not None
            and len(examples) >= limit
        ):
            break

    if not examples:
        raise RuntimeError(
            "No valid HellaSwag examples were selected."
        )

    return examples, total_available


def build_prompt(example: Dict) -> str:
    """Build the HellaSwag context presented to the model."""
    context = example.get(
        "ctx",
        "",
    ).strip()

    if context:
        return context

    context_a = example.get(
        "ctx_a",
        "",
    ).strip()

    context_b = example.get(
        "ctx_b",
        "",
    ).strip()

    return (
        f"{context_a} {context_b}"
    ).strip()


# =============================================================================
# HELLASWAG SCORING
# =============================================================================


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
    """
    Score a candidate ending by conditional log-likelihood.

    With normalize_by_length=True, this corresponds to the normalized
    HellaSwag accuracy commonly reported as acc_norm.
    """
    prompt = prompt.strip()
    continuation = continuation.strip()

    full_text = (
        prompt
        + " "
        + continuation
    )

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

    logits = outputs.logits[
        :,
        :-1,
        :,
    ].contiguous()

    target_ids = input_ids[
        :,
        1:,
    ].contiguous()

    log_probabilities = F.log_softmax(
        logits.float(),
        dim=-1,
    )

    token_log_probabilities = (
        log_probabilities.gather(
            -1,
            target_ids.unsqueeze(-1),
        ).squeeze(-1)
    )

    continuation_start = max(
        prompt_length - 1,
        0,
    )

    continuation_log_probabilities = (
        token_log_probabilities[
            :,
            continuation_start:,
        ]
    )

    if (
        continuation_log_probabilities.numel()
        == 0
    ):
        return float("-inf")

    score = (
        continuation_log_probabilities
        .sum()
        .item()
    )

    if normalize_by_length:
        score /= (
            continuation_log_probabilities.numel()
        )

    return float(score)


def evaluate_hellaswag(
    model,
    tokenizer,
    examples: List[Dict],
    max_length: int,
    normalize_by_length: bool = True,
) -> Dict:
    """Evaluate a model on the selected HellaSwag examples."""
    device = next(
        model.parameters()
    ).device

    num_correct = 0
    predictions = []

    model.eval()

    total_examples = len(examples)

    for example_number, example in enumerate(
        examples,
        start=1,
    ):
        prompt = build_prompt(
            example
        )

        scores = []

        for ending in example["endings"]:
            score = score_continuation(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                continuation=ending,
                device=device,
                max_length=max_length,
                normalize_by_length=normalize_by_length,
            )

            scores.append(
                float(score)
            )

        prediction = int(
            max(
                range(len(scores)),
                key=lambda index: scores[index],
            )
        )

        gold = int(
            example["label"]
        )

        correct = int(
            prediction == gold
        )

        num_correct += correct

        competing_scores = (
            scores[:prediction]
            + scores[prediction + 1:]
        )

        margin = (
            scores[prediction]
            - max(competing_scores)
            if competing_scores
            else 0.0
        )

        predictions.append(
            {
                "id": int(example["id"]),
                "dataset_id": example["dataset_id"],
                "prediction": prediction,
                "gold": gold,
                "correct": bool(correct),
                "scores": scores,
                "margin": float(margin),
            }
        )

        if (
            example_number % 50 == 0
            or example_number == total_examples
        ):
            running_accuracy = (
                num_correct
                / example_number
            )

            print(
                f"Progress: "
                f"{example_number}/"
                f"{total_examples} | "
                f"running accuracy="
                f"{running_accuracy:.4f}",
                flush=True,
            )

    if total_examples == 0:
        raise RuntimeError(
            "No HellaSwag examples were scored."
        )

    accuracy = (
        num_correct
        / total_examples
    )

    accuracy_stderr = (
        calculate_accuracy_stderr(
            accuracy=accuracy,
            num_examples=total_examples,
        )
    )

    return {
        "num_examples": total_examples,
        "num_correct": num_correct,
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
            "Evaluate FP16, naive quantization, and "
            "superweight-aware quantization on reproducibly "
            "sampled HellaSwag examples."
        )
    )

    parser.add_argument(
        "--model-key",
        type=str,
        required=True,
        choices=sorted(
            MODEL_CONFIGS.keys()
        ),
    )

    parser.add_argument(
        "--mode",
        type=str,
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
        type=str,
        default="float16",
        choices=[
            "float16",
            "bfloat16",
            "float32",
        ],
    )

    parser.add_argument(
        "--split",
        type=str,
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
            "Seed used to shuffle HellaSwag before applying "
            "--limit. Use the same seed for all methods."
        ),
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--normalize-by-length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Normalize continuation log-likelihood by token count. "
            "Enabled by default to obtain an acc_norm-style metric."
        ),
    )

    parser.add_argument(
        "--output-json",
        type=str,
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
            "--activation-bits must be greater than zero."
        )

    if args.limit <= 0:
        raise ValueError(
            "--limit must be greater than zero."
        )

    if args.max_length <= 1:
        raise ValueError(
            "--max-length must be greater than one."
        )

    if args.block_rows <= 0:
        raise ValueError(
            "--block-rows must be greater than zero."
        )

    if args.block_cols <= 0:
        raise ValueError(
            "--block-cols must be greater than zero."
        )

    set_all_seeds(
        args.eval_seed
    )

    model_config = MODEL_CONFIGS[
        args.model_key
    ]

    model_id = model_config[
        "hf_name"
    ]

    torch_dtype = resolve_torch_dtype(
        args.dtype
    )

    print("=" * 78, flush=True)
    print(
        "HELLASWAG QUANTIZATION EVALUATION",
        flush=True,
    )
    print("=" * 78, flush=True)
    print(
        f"Model key:               "
        f"{args.model_key}",
        flush=True,
    )
    print(
        f"Model ID:                "
        f"{model_id}",
        flush=True,
    )
    print(
        f"Mode:                    "
        f"{args.mode}",
        flush=True,
    )
    print(
        f"Weight bits:             "
        f"{args.bits}",
        flush=True,
    )
    print(
        f"Activation bits:         "
        f"{args.activation_bits}",
        flush=True,
    )
    print(
        f"Quantization granularity:"
        f"{args.quant_granularity}",
        flush=True,
    )
    print(
        f"Block rows:              "
        f"{args.block_rows}",
        flush=True,
    )
    print(
        f"Block columns:           "
        f"{args.block_cols}",
        flush=True,
    )
    print(
        f"Clip z-score:            "
        f"{args.clip_z}",
        flush=True,
    )
    print(
        f"Superweight scale:       "
        f"{args.sw_scale}",
        flush=True,
    )
    print(
        f"Restore neighborhood:    "
        f"{args.restore_neighborhood}",
        flush=True,
    )
    print(
        f"Evaluation split:        "
        f"{args.split}",
        flush=True,
    )
    print(
        f"Evaluation limit:        "
        f"{args.limit}",
        flush=True,
    )
    print(
        f"Evaluation seed:         "
        f"{args.eval_seed}",
        flush=True,
    )
    print(
        f"Maximum sequence length: "
        f"{args.max_length}",
        flush=True,
    )
    print(
        f"Normalize by length:     "
        f"{args.normalize_by_length}",
        flush=True,
    )
    print(
        f"Data type:               "
        f"{args.dtype}",
        flush=True,
    )
    print("=" * 78, flush=True)

    print(
        f"Loading tokenizer: "
        f"{model_id}",
        flush=True,
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
    )

    if (
        tokenizer.pad_token is None
        and tokenizer.eos_token is not None
    ):
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    print(
        f"Loading model: "
        f"{model_id} ({args.dtype})",
        flush=True,
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map=(
                "auto"
                if torch.cuda.is_available()
                else None
            ),
        )
    )

    model.eval()

    print(
        "Preparing model with "
        f"mode={args.mode}, "
        f"bits={args.bits}, "
        f"activation_bits="
        f"{args.activation_bits}, "
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
        f"Loading HellaSwag "
        f"split={args.split}, "
        f"limit={args.limit}, "
        f"eval_seed={args.eval_seed}",
        flush=True,
    )

    examples, total_available = (
        load_hellaswag_examples(
            split=args.split,
            limit=args.limit,
            eval_seed=args.eval_seed,
        )
    )

    selected_example_ids = [
        int(example["id"])
        for example in examples
    ]

    print(
        f"Available HellaSwag examples: "
        f"{total_available}",
        flush=True,
    )

    print(
        f"Selected HellaSwag examples: "
        f"{len(examples)}",
        flush=True,
    )

    print(
        "First 10 selected original indices: "
        f"{selected_example_ids[:10]}",
        flush=True,
    )

    print(
        f"Evaluating "
        f"{len(examples)} examples...",
        flush=True,
    )

    metrics = evaluate_hellaswag(
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
            "layer": int(item["layer"]),
            "center_row": int(
                item["center_row"]
            ),
            "center_col": int(
                item["center_col"]
            ),
            "row": int(item["row"]),
            "col": int(item["col"]),
            "is_center": bool(
                item["is_center"]
            ),
        }
        for item in protected_values
    ]

    result = {
        "benchmark": "hellaswag",
        "metric": (
            "accuracy_normalized"
            if args.normalize_by_length
            else "accuracy_raw"
        ),
        "model_key": args.model_key,
        "model_id": model_id,
        "mode": args.mode,
        "bits": args.bits,
        "activation_bits": (
            args.activation_bits
        ),
        "sw_scale": args.sw_scale,
        "restore_neighborhood": (
            args.restore_neighborhood
        ),
        "quant_granularity": (
            args.quant_granularity
        ),
        "block_rows": args.block_rows,
        "block_cols": args.block_cols,
        "clip_z": args.clip_z,
        "normalize_by_length": (
            args.normalize_by_length
        ),
        "num_protected_values": len(
            protected_values
        ),
        "protected_values": (
            protected_summary
        ),
        "num_activation_hooks": len(
            activation_handles
        ),
        "dtype": args.dtype,
        "split": args.split,
        "available_examples": (
            total_available
        ),
        "requested_limit": args.limit,
        "evaluated_examples": len(
            examples
        ),
        "eval_seed": args.eval_seed,
        "selected_example_ids": (
            selected_example_ids
        ),
        "max_length": args.max_length,
        **metrics,
    }

    result = make_json_safe(
        result
    )

    output_path = Path(
        args.output_json
    )

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
    print("=" * 78, flush=True)
    print(
        "FINAL HELLASWAG RESULT",
        flush=True,
    )
    print("=" * 78, flush=True)
    print(
        f"Mode:              "
        f"{args.mode}",
        flush=True,
    )
    print(
        f"Evaluation seed:   "
        f"{args.eval_seed}",
        flush=True,
    )
    print(
        f"Metric:            "
        f"{result['metric']}",
        flush=True,
    )
    print(
        f"Examples:          "
        f"{metrics['num_examples']}",
        flush=True,
    )
    print(
        f"Correct:           "
        f"{metrics['num_correct']}",
        flush=True,
    )
    print(
        f"Accuracy:          "
        f"{metrics['accuracy']:.4f}",
        flush=True,
    )
    print(
        f"Accuracy stderr:   "
        f"{metrics['accuracy_stderr']:.4f}",
        flush=True,
    )
    print(
        f"Saved result to:   "
        f"{output_path}",
        flush=True,
    )
    print("=" * 78, flush=True)

    for handle in activation_handles:
        handle.remove()


if __name__ == "__main__":
    main()