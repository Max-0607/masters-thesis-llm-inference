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
    """Set Python and PyTorch random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# =============================================================================
# GENERAL UTILITIES
# =============================================================================


def resolve_torch_dtype(name: str) -> torch.dtype:
    """Resolve a string representation to a PyTorch dtype."""
    name = name.lower()

    if name == "float16":
        return torch.float16

    if name == "bfloat16":
        return torch.bfloat16

    if name == "float32":
        return torch.float32

    raise ValueError(
        f"Unsupported dtype: {name}"
    )


def make_json_safe(obj: Any) -> Any:
    """Replace non-finite floats before JSON serialization."""
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
    """Calculate the binomial standard error of accuracy."""
    if num_examples <= 0:
        raise ValueError(
            "num_examples must be greater than zero."
        )

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
    """Return the coordinates restored around a superweight."""
    n_rows, n_cols = shape
    indices = set()

    def add(r: int, c: int) -> None:
        if 0 <= r < n_rows and 0 <= c < n_cols:
            indices.add(
                (r, c)
            )

    add(row, col)

    if neighborhood in {"row", "cross"}:
        add(row, col - 1)
        add(row, col + 1)

    if neighborhood in {"column", "cross"}:
        add(row - 1, col)
        add(row + 1, col)

    return sorted(
        indices
    )


# =============================================================================
# ACTIVATION QUANTIZATION
# =============================================================================


@torch.no_grad()
def uniform_quantize_activation_tensor(
    x: torch.Tensor,
    n_bits: int,
) -> torch.Tensor:
    """Apply symmetric per-token activation fake quantization."""
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

    return (
        quantized * scale
    ).to(original_dtype)


def add_activation_quant_hooks(
    model,
    n_bits: int,
):
    """Add fake activation quantization hooks to Linear layers."""
    handles = []

    def pre_hook(module, inputs):
        if not inputs:
            return inputs

        x = inputs[0]

        if not torch.is_tensor(x):
            return inputs

        x_quantized = uniform_quantize_activation_tensor(
            x,
            n_bits,
        )

        return (
            x_quantized,
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
    """Clip a tensor using a global z-score threshold."""
    if clip_z is None or clip_z <= 0:
        return weight

    original_dtype = weight.dtype
    weight_float = weight.float()

    mean = weight_float.mean()
    std = weight_float.std(
        unbiased=False
    )

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
    """Apply asymmetric per-tensor fake weight quantization."""
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

    if (weight_max - weight_min).item() == 0:
        return weight_float.to(
            original_dtype
        )

    delta = (
        weight_max - weight_min
    ) / qmax

    quantized = torch.clamp(
        torch.round(
            (weight_float - weight_min) / delta
        ),
        qmin,
        qmax,
    )

    return (
        quantized * delta + weight_min
    ).to(original_dtype)


@torch.no_grad()
def uniform_quantize_weight_tensor_blockwise_2d(
    weight: torch.Tensor,
    n_bits: int,
    block_rows: int = 128,
    block_cols: int = 128,
) -> torch.Tensor:
    """Apply asymmetric 2D blockwise fake weight quantization."""
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
    output = torch.empty_like(
        weight_float
    )

    qmin = 0
    qmax = (2 ** n_bits) - 1

    n_rows, n_cols = weight_float.shape

    for row_start in range(
        0,
        n_rows,
        block_rows,
    ):
        row_end = min(
            row_start + block_rows,
            n_rows,
        )

        for col_start in range(
            0,
            n_cols,
            block_cols,
        ):
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

            if (block_max - block_min).item() == 0:
                output[
                    row_start:row_end,
                    col_start:col_end,
                ] = block
                continue

            delta = (
                block_max - block_min
            ) / qmax

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
                quantized * delta + block_min
            )

    return output.to(
        original_dtype
    )


@torch.no_grad()
def quantize_parameter(
    parameter: torch.Tensor,
    n_bits: int,
    quant_granularity: str = "tensor",
    block_rows: int = 128,
    block_cols: int = 128,
    clip_z: Optional[float] = None,
) -> torch.Tensor:
    """Quantize one parameter tensor."""
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
        return uniform_quantize_weight_tensor_blockwise_2d(
            clipped_weight,
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
    """Quantize all supported Linear weights in-place."""
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
        "Weight quantization done: "
        f"quantized Linear modules={quantized_modules}, "
        f"skipped Linear modules={skipped_modules}",
        flush=True,
    )

    return model


# =============================================================================
# SUPERWEIGHT QUANTIZATION
# =============================================================================


@torch.no_grad()
def collect_protected_superweights(
    model,
    model_key: str,
    sw_scale: float = 1.0,
    restore_neighborhood: str = "scalar",
):
    """Save the original values that will be restored after quantization."""
    if model_key not in SUPERWEIGHTS:
        raise ValueError(
            f"No superweights registered for "
            f"model_key={model_key!r}."
        )

    model_config = MODEL_CONFIGS[
        model_key
    ]

    layers = get_nested_attr(
        model,
        model_config["layer_path"],
    )

    down_projection_path = model_config[
        "down_proj_path"
    ]

    protected_values = []

    for entry in SUPERWEIGHTS[model_key]:
        layer_index = int(
            entry["layer"]
        )

        row_index = int(
            entry["row"]
        )

        column_index = int(
            entry["col"]
        )

        module = get_nested_attr(
            layers[layer_index],
            down_projection_path,
        )

        weight = module.weight.data

        restore_indices = get_superweight_restore_indices(
            row=row_index,
            col=column_index,
            shape=weight.shape,
            neighborhood=restore_neighborhood,
        )

        for restored_row, restored_column in restore_indices:
            is_center = (
                restored_row == row_index
                and restored_column == column_index
            )

            value = weight[
                restored_row,
                restored_column,
            ].detach().clone()

            if is_center:
                value = value * sw_scale

            protected_values.append(
                {
                    "layer": layer_index,
                    "center_row": row_index,
                    "center_col": column_index,
                    "row": int(restored_row),
                    "col": int(restored_column),
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
    """Restore the protected values after quantization."""
    model_config = MODEL_CONFIGS[
        model_key
    ]

    layers = get_nested_attr(
        model,
        model_config["layer_path"],
    )

    down_projection_path = model_config[
        "down_proj_path"
    ]

    for item in protected_values:
        module = get_nested_attr(
            layers[item["layer"]],
            down_projection_path,
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
    """Quantize weights and restore protected superweight values."""
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
    """Prepare FP16, naive, or superweight-aware model."""
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

        return model, protected_values, activation_handles

    raise ValueError(
        f"Unsupported mode: {mode}"
    )


# =============================================================================
# WINOGRANDE DATASET
# =============================================================================


def load_winogrande_dataset(
    split: str,
    subset: str,
):
    """Load WinoGrande with a fallback dataset identifier."""
    candidate_loaders = [
        lambda: load_dataset(
            "winogrande",
            subset,
            split=split,
        ),
        lambda: load_dataset(
            "allenai/winogrande",
            subset,
            split=split,
        ),
    ]

    last_error: Optional[Exception] = None

    for loader in candidate_loaders:
        try:
            return loader()
        except Exception as error:
            last_error = error

    raise RuntimeError(
        "Could not load WinoGrande. "
        f"Last error: {last_error}"
    )


def load_winogrande_examples(
    split: str,
    limit: Optional[int],
    subset: str,
    eval_seed: int,
) -> tuple[List[Dict], int]:
    """
    Load a reproducibly sampled WinoGrande subset.

    Original indices are attached before shuffling. Therefore, identical
    split, subset, limit, and eval_seed values select the same examples for
    FP16, Naive W4, Super W4, GPTQ, and AWQ.
    """
    dataset = load_winogrande_dataset(
        split=split,
        subset=subset,
    )

    total_available = len(
        dataset
    )

    if total_available == 0:
        raise RuntimeError(
            f"WinoGrande split {split!r} is empty."
        )

    if limit is not None and limit <= 0:
        raise ValueError(
            "--limit must be greater than zero."
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
        answer = row.get(
            "answer",
            None,
        )

        if answer not in {"1", "2", 1, 2}:
            continue

        label = (
            0
            if str(answer) == "1"
            else 1
        )

        examples.append(
            {
                "id": int(
                    row["_original_index"]
                ),
                "qID": str(
                    row.get(
                        "qID",
                        row["_original_index"],
                    )
                ),
                "sentence": row["sentence"],
                "option1": row["option1"],
                "option2": row["option2"],
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
            "No valid WinoGrande examples were selected."
        )

    return examples, total_available


# =============================================================================
# WINOGRANDE PROMPT AND SCORING
# =============================================================================


def build_prompt_and_choices(
    example: Dict,
) -> Dict[str, str]:
    """Build the prefix and the two possible continuations."""
    sentence = example[
        "sentence"
    ].strip()

    if "_" not in sentence:
        raise ValueError(
            "Expected '_' placeholder in WinoGrande sentence: "
            f"{sentence}"
        )

    prefix, suffix = sentence.split(
        "_",
        1,
    )

    return {
        "prompt": prefix.strip(),
        "choice1": (
            example["option1"].strip()
            + suffix
        ).strip(),
        "choice2": (
            example["option2"].strip()
            + suffix
        ).strip(),
    }


@torch.no_grad()
def score_continuation(
    model,
    tokenizer,
    prompt: str,
    continuation: str,
    device: torch.device,
    max_length: int = 256,
    normalize_by_length: bool = False,
) -> Dict[str, float | int]:
    """Calculate conditional log-likelihood for one completion."""
    prompt = prompt.strip()
    continuation = continuation.strip()

    full_text = (
        prompt + " " + continuation
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

    prompt_length = int(
        prompt_ids.shape[1]
    )

    full_length = int(
        full_ids.shape[1]
    )

    if full_length <= prompt_length:
        return {
            "score": float("-inf"),
            "sum_log_likelihood": float("-inf"),
            "num_continuation_tokens": 0,
        }

    input_ids = full_ids.to(
        device
    )

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

    token_log_probabilities = log_probabilities.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)

    continuation_start = max(
        prompt_length - 1,
        0,
    )

    continuation_log_probabilities = token_log_probabilities[
        :,
        continuation_start:,
    ]

    num_continuation_tokens = int(
        continuation_log_probabilities.numel()
    )

    if num_continuation_tokens == 0:
        return {
            "score": float("-inf"),
            "sum_log_likelihood": float("-inf"),
            "num_continuation_tokens": 0,
        }

    sum_log_likelihood = float(
        continuation_log_probabilities.sum().item()
    )

    score = sum_log_likelihood

    if normalize_by_length:
        score /= num_continuation_tokens

    return {
        "score": float(score),
        "sum_log_likelihood": sum_log_likelihood,
        "num_continuation_tokens": num_continuation_tokens,
    }


def evaluate_winogrande(
    model,
    tokenizer,
    examples: List[Dict],
    max_length: int,
    normalize_by_length: bool = False,
) -> Dict:
    """Evaluate a prepared model on WinoGrande."""
    device = next(
        model.parameters()
    ).device

    num_correct = 0
    predictions: List[Dict] = []
    total_examples = len(
        examples
    )

    model.eval()

    for example_number, example in enumerate(
        examples,
        start=1,
    ):
        built = build_prompt_and_choices(
            example
        )

        option1_result = score_continuation(
            model=model,
            tokenizer=tokenizer,
            prompt=built["prompt"],
            continuation=built["choice1"],
            device=device,
            max_length=max_length,
            normalize_by_length=normalize_by_length,
        )

        option2_result = score_continuation(
            model=model,
            tokenizer=tokenizer,
            prompt=built["prompt"],
            continuation=built["choice2"],
            device=device,
            max_length=max_length,
            normalize_by_length=normalize_by_length,
        )

        score1 = float(
            option1_result["score"]
        )

        score2 = float(
            option2_result["score"]
        )

        prediction = (
            0
            if score1 >= score2
            else 1
        )

        gold = int(
            example["label"]
        )

        correct = int(
            prediction == gold
        )

        num_correct += correct

        predictions.append(
            {
                "id": int(example["id"]),
                "qID": example["qID"],
                "prediction": prediction,
                "gold": gold,
                "correct": bool(correct),
                "score_option1": score1,
                "score_option2": score2,
                "sum_log_likelihood_option1": float(
                    option1_result[
                        "sum_log_likelihood"
                    ]
                ),
                "sum_log_likelihood_option2": float(
                    option2_result[
                        "sum_log_likelihood"
                    ]
                ),
                "option1_token_count": int(
                    option1_result[
                        "num_continuation_tokens"
                    ]
                ),
                "option2_token_count": int(
                    option2_result[
                        "num_continuation_tokens"
                    ]
                ),
                "margin": float(
                    score1 - score2
                ),
            }
        )

        if (
            example_number % 50 == 0
            or example_number == total_examples
        ):
            running_accuracy = (
                num_correct / example_number
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
            "No WinoGrande examples were evaluated."
        )

    accuracy = (
        num_correct / total_examples
    )

    accuracy_stderr = calculate_accuracy_stderr(
        accuracy=accuracy,
        num_examples=total_examples,
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate FP16, Naive, or Superweight-aware OLMo "
            "on a reproducibly sampled WinoGrande subset."
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
        "--subset",
        type=str,
        default="winogrande_xl",
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
            "Seed used only to shuffle WinoGrande before applying "
            "--limit. Use the same value across all methods."
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
        default=False,
        help=(
            "Normalize continuation log-likelihood by token count. "
            "Disabled by default to reproduce the original experiment."
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


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.limit <= 0:
        raise ValueError(
            "--limit must be greater than zero."
        )

    if args.bits <= 0:
        raise ValueError(
            "--bits must be greater than zero."
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

    # The model is deterministic. Only the sampled evaluation subset changes.
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
    print("WINOGRANDE EVALUATION", flush=True)
    print("=" * 78, flush=True)
    print(f"Model key:               {args.model_key}", flush=True)
    print(f"Model ID:                {model_id}", flush=True)
    print(f"Mode:                    {args.mode}", flush=True)
    print(f"Weight bits:             {args.bits}", flush=True)
    print(f"Activation bits:         {args.activation_bits}", flush=True)
    print(f"Superweight scale:       {args.sw_scale}", flush=True)
    print(f"Restore neighborhood:    {args.restore_neighborhood}", flush=True)
    print(f"Quantization granularity:{args.quant_granularity}", flush=True)
    print(f"Block rows:              {args.block_rows}", flush=True)
    print(f"Block cols:              {args.block_cols}", flush=True)
    print(f"Clip z:                  {args.clip_z}", flush=True)
    print(f"Subset:                  {args.subset}", flush=True)
    print(f"Split:                   {args.split}", flush=True)
    print(f"Evaluation limit:        {args.limit}", flush=True)
    print(f"Evaluation seed:         {args.eval_seed}", flush=True)
    print(f"Maximum length:          {args.max_length}", flush=True)
    print(
        f"Normalize by length:     "
        f"{args.normalize_by_length}",
        flush=True,
    )
    print("=" * 78, flush=True)

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

    print(
        f"Loading model: "
        f"{model_id} ({args.dtype})",
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    model.eval()

    print(
        "Preparing model...",
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
        f"Loading WinoGrande "
        f"subset={args.subset}, "
        f"split={args.split}, "
        f"limit={args.limit}, "
        f"eval_seed={args.eval_seed}",
        flush=True,
    )

    examples, total_available = load_winogrande_examples(
        split=args.split,
        limit=args.limit,
        subset=args.subset,
        eval_seed=args.eval_seed,
    )

    selected_example_ids = [
        int(example["id"])
        for example in examples
    ]

    selected_qids = [
        example["qID"]
        for example in examples
    ]

    print(
        f"Available WinoGrande examples: "
        f"{total_available}",
        flush=True,
    )

    print(
        f"Selected WinoGrande examples: "
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

    metrics = evaluate_winogrande(
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
        "benchmark": "winogrande",
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
        "subset": args.subset,
        "split": args.split,
        "available_examples": total_available,
        "requested_limit": args.limit,
        "evaluated_examples": len(examples),
        "eval_seed": args.eval_seed,
        "selected_example_ids": selected_example_ids,
        "selected_qids": selected_qids,
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

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78, flush=True)
    print("FINAL WINOGRANDE RESULT", flush=True)
    print("=" * 78, flush=True)
    print(f"Mode:              {args.mode}", flush=True)
    print(f"Evaluation seed:   {args.eval_seed}", flush=True)
    print(f"Examples:          {metrics['num_examples']}", flush=True)
    print(f"Correct:           {metrics['num_correct']}", flush=True)
    print(f"Accuracy:          {metrics['accuracy']:.4f}", flush=True)
    print(
        f"Accuracy stderr:   "
        f"{metrics['accuracy_stderr']:.4f}",
        flush=True,
    )
    print(f"Saved result to:   {output_path}", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()