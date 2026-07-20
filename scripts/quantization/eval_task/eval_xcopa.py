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
    """Binomial standard error of accuracy."""
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
    """Symmetric per-token activation fake quantization."""
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
    handles = []

    def pre_hook(module, inputs):
        if not inputs:
            return inputs

        x = inputs[0]

        if not torch.is_tensor(x):
            return inputs

        quantized_x = uniform_quantize_activation_tensor(
            x,
            n_bits,
        )

        return (
            quantized_x,
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
    """
    Symmetric per-tensor fake weight quantization.

    This retains the quantization formulation used by the supplied XCOPA
    experiment.
    """
    if n_bits <= 0:
        raise ValueError(
            "n_bits must be positive."
        )

    original_dtype = weight.dtype
    weight_float = weight.float()

    qmin = -(2 ** (n_bits - 1))
    qmax = (2 ** (n_bits - 1)) - 1

    max_abs = weight_float.abs().amax()

    if max_abs.item() == 0:
        return weight_float.to(
            original_dtype
        )

    scale = max_abs / qmax

    quantized = torch.clamp(
        torch.round(weight_float / scale),
        qmin,
        qmax,
    )

    return (
        quantized * scale
    ).to(original_dtype)


@torch.no_grad()
def uniform_quantize_weight_tensor_blockwise_2d(
    weight: torch.Tensor,
    n_bits: int,
    block_rows: int = 128,
    block_cols: int = 128,
) -> torch.Tensor:
    """Asymmetric two-dimensional blockwise fake quantization."""
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

        return model, protected_values, activation_handles

    raise ValueError(
        f"Unsupported mode: {mode}"
    )


# =============================================================================
# XCOPA / COPA DATA
# =============================================================================


def get_language_config(
    language: str,
) -> Dict[str, str]:
    prompts = {
        "en": {
            "cause": "What was the cause?",
            "effect": "What happened as a result?",
        },
        "et": {
            "cause": "Mis oli põhjus?",
            "effect": "Mis juhtus selle tulemusena?",
        },
        "ht": {
            "cause": "Ki sa ki te kòz la?",
            "effect": "Kisa ki te rive kòm rezilta?",
        },
        "id": {
            "cause": "Apa penyebabnya?",
            "effect": "Apa yang terjadi sebagai hasilnya?",
        },
        "it": {
            "cause": "Qual è stata la causa?",
            "effect": "Che cosa è successo come risultato?",
        },
        "qu": {
            "cause": "Imataq karqan?",
            "effect": "Imataq chaymanta pasaran?",
        },
        "sw": {
            "cause": "Sababu ilikuwa nini?",
            "effect": "Nini kilitokea kama matokeo?",
        },
        "ta": {
            "cause": "காரணம் என்ன?",
            "effect": "இதன் விளைவாக என்ன நடந்தது?",
        },
        "th": {
            "cause": "สาเหตุคืออะไร?",
            "effect": "เกิดอะไรขึ้นเป็นผลลัพธ์?",
        },
        "tr": {
            "cause": "Sebep neydi?",
            "effect": "Sonuç olarak ne oldu?",
        },
        "vi": {
            "cause": "Nguyên nhân là gì?",
            "effect": "Kết quả là gì?",
        },
        "zh": {
            "cause": "原因是什么？",
            "effect": "结果发生了什么？",
        },
    }

    if language not in prompts:
        raise ValueError(
            f"Unsupported language: {language}"
        )

    return prompts[language]


def build_prompt(
    example: Dict,
    language: str,
) -> str:
    language_config = get_language_config(
        language
    )

    question = language_config[
        example["question"]
    ]

    premise = example[
        "premise"
    ].strip()

    return (
        f"{premise}\n"
        f"{question}\n"
        f"Answer:"
    )


def normalize_xcopa_example(
    example: Dict,
    original_index: int,
) -> Dict:
    label = example.get(
        "label",
        example.get("answer"),
    )

    if label not in {0, 1}:
        raise ValueError(
            f"Unexpected XCOPA label: {label}"
        )

    return {
        "id": int(original_index),
        "premise": str(example["premise"]),
        "question": str(example["question"]),
        "choice1": str(example["choice1"]),
        "choice2": str(example["choice2"]),
        "label": int(label),
    }


def load_xcopa_dataset(
    language: str,
    split: str,
):
    """
    Load English COPA or one of the multilingual XCOPA configurations.

    For language='en', SuperGLUE COPA is used because XCOPA itself does not
    provide an English configuration.
    """
    if language == "en":
        split_map = {
            "validation": "validation",
            "val": "validation",
            "train": "train",
            "test": "validation",
        }

        dataset_split = split_map.get(
            split,
            split,
        )

        return load_dataset(
            "super_glue",
            "copa",
            split=dataset_split,
        )

    split_map = {
        "validation": "validation",
        "val": "validation",
        "train": "validation",
        "test": "validation",
    }

    dataset_split = split_map.get(
        split,
        split,
    )

    candidate_loaders = [
        lambda: load_dataset(
            "xcopa",
            language,
            split=dataset_split,
        ),
        lambda: load_dataset(
            "cambridgeltl/xcopa",
            language,
            split=dataset_split,
        ),
    ]

    last_error: Optional[Exception] = None

    for loader in candidate_loaders:
        try:
            return loader()
        except Exception as error:
            last_error = error

    raise RuntimeError(
        f"Could not load XCOPA for language={language}. "
        f"Last error: {last_error}"
    )


def load_xcopa_examples(
    language: str,
    split: str,
    limit: Optional[int],
    eval_seed: int,
) -> tuple[List[Dict], int]:
    """
    Load a reproducibly ordered COPA/XCOPA evaluation set.

    Original indices are attached before shuffling. For English validation,
    only 100 examples exist. Therefore, limit=500 selects the full dataset and
    changing eval_seed only changes order, not membership.
    """
    dataset = load_xcopa_dataset(
        language=language,
        split=split,
    )

    total_available = len(
        dataset
    )

    if total_available == 0:
        raise RuntimeError(
            "The selected COPA/XCOPA split is empty."
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

    effective_limit = (
        total_available
        if limit is None
        else min(limit, total_available)
    )

    dataset = dataset.select(
        range(effective_limit)
    )

    examples = [
        normalize_xcopa_example(
            row,
            original_index=int(
                row["_original_index"]
            ),
        )
        for row in dataset
    ]

    if not examples:
        raise RuntimeError(
            "No valid COPA/XCOPA examples were selected."
        )

    return examples, total_available


# =============================================================================
# SCORING
# =============================================================================


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
    prompt = prompt.strip()
    continuation = continuation.strip()

    full_text = (
        f"{prompt} {continuation}".strip()
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


def evaluate_xcopa(
    model,
    tokenizer,
    examples: List[Dict],
    language: str,
    max_length: int,
    normalize_by_length: bool = False,
) -> Dict:
    device = next(
        model.parameters()
    ).device

    num_correct = 0
    predictions: List[Dict] = []

    per_question = {
        "cause": {
            "correct": 0,
            "total": 0,
        },
        "effect": {
            "correct": 0,
            "total": 0,
        },
    }

    model.eval()

    total_examples = len(
        examples
    )

    for example_number, example in enumerate(
        examples,
        start=1,
    ):
        prompt = build_prompt(
            example,
            language,
        )

        choice1_result = score_continuation(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            continuation=example["choice1"],
            device=device,
            max_length=max_length,
            normalize_by_length=normalize_by_length,
        )

        choice2_result = score_continuation(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            continuation=example["choice2"],
            device=device,
            max_length=max_length,
            normalize_by_length=normalize_by_length,
        )

        score1 = float(
            choice1_result["score"]
        )

        score2 = float(
            choice2_result["score"]
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

        question_type = example[
            "question"
        ]

        if question_type in per_question:
            per_question[
                question_type
            ]["correct"] += correct

            per_question[
                question_type
            ]["total"] += 1

        predictions.append(
            {
                "id": int(example["id"]),
                "question_type": question_type,
                "prediction": prediction,
                "gold": gold,
                "correct": bool(correct),
                "score_choice1": score1,
                "score_choice2": score2,
                "sum_log_likelihood_choice1": float(
                    choice1_result[
                        "sum_log_likelihood"
                    ]
                ),
                "sum_log_likelihood_choice2": float(
                    choice2_result[
                        "sum_log_likelihood"
                    ]
                ),
                "choice1_token_count": int(
                    choice1_result[
                        "num_continuation_tokens"
                    ]
                ),
                "choice2_token_count": int(
                    choice2_result[
                        "num_continuation_tokens"
                    ]
                ),
                "margin": float(
                    score1 - score2
                ),
            }
        )

        if (
            example_number % 25 == 0
            or example_number == total_examples
        ):
            print(
                f"Progress: "
                f"{example_number}/{total_examples} | "
                f"running accuracy="
                f"{num_correct / example_number:.4f}",
                flush=True,
            )

    if total_examples == 0:
        raise RuntimeError(
            "No COPA/XCOPA examples were evaluated."
        )

    accuracy = (
        num_correct / total_examples
    )

    accuracy_stderr = calculate_accuracy_stderr(
        accuracy=accuracy,
        num_examples=total_examples,
    )

    per_question_result = {}

    for question_type, stats in per_question.items():
        question_total = stats["total"]

        per_question_result[
            question_type
        ] = {
            "total": question_total,
            "correct": stats["correct"],
            "accuracy": (
                stats["correct"] / question_total
                if question_total > 0
                else None
            ),
        }

    return {
        "num_examples": total_examples,
        "num_correct": num_correct,
        "accuracy": accuracy,
        "accuracy_stderr": accuracy_stderr,
        "per_question": per_question_result,
        "predictions": predictions,
    }


# =============================================================================
# ARGUMENTS
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate FP16, Naive, or Superweight-aware quantization "
            "on English COPA or multilingual XCOPA."
        )
    )

    parser.add_argument(
        "--model-key",
        required=True,
        choices=sorted(
            MODEL_CONFIGS.keys()
        ),
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
        "--language",
        default="en",
        choices=[
            "en",
            "et",
            "ht",
            "id",
            "it",
            "qu",
            "sw",
            "ta",
            "th",
            "tr",
            "vi",
            "zh",
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
    )

    parser.add_argument(
        "--output-json",
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
    print("COPA / XCOPA EVALUATION", flush=True)
    print("=" * 78, flush=True)
    print(f"Model key:              {args.model_key}", flush=True)
    print(f"Model ID:               {model_id}", flush=True)
    print(f"Mode:                   {args.mode}", flush=True)
    print(f"Weight bits:            {args.bits}", flush=True)
    print(f"Language:               {args.language}", flush=True)
    print(f"Split:                  {args.split}", flush=True)
    print(f"Requested limit:        {args.limit}", flush=True)
    print(f"Evaluation seed:        {args.eval_seed}", flush=True)
    print(f"Maximum length:         {args.max_length}", flush=True)
    print(
        f"Normalize by length:    "
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

    examples, total_available = load_xcopa_examples(
        language=args.language,
        split=args.split,
        limit=args.limit,
        eval_seed=args.eval_seed,
    )

    selected_example_ids = [
        int(example["id"])
        for example in examples
    ]

    print(
        f"Available examples:      "
        f"{total_available}",
        flush=True,
    )

    print(
        f"Selected examples:       "
        f"{len(examples)}",
        flush=True,
    )

    if args.language == "en" and args.limit >= total_available:
        print(
            "NOTE: English COPA validation contains only "
            f"{total_available} examples. Changing eval_seed changes "
            "their order but not the selected set.",
            flush=True,
        )

    print(
        "First 10 original indices: "
        f"{selected_example_ids[:10]}",
        flush=True,
    )

    metrics = evaluate_xcopa(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        language=args.language,
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
        "benchmark": "xcopa",
        "dataset": (
            "super_glue_copa"
            if args.language == "en"
            else "xcopa"
        ),
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
        "language": args.language,
        "split": args.split,
        "available_examples": total_available,
        "requested_limit": args.limit,
        "evaluated_examples": len(examples),
        "eval_seed": args.eval_seed,
        "selected_example_ids": selected_example_ids,
        "sample_membership_varies_by_seed": (
            args.limit < total_available
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
    print("FINAL COPA / XCOPA RESULT", flush=True)
    print("=" * 78, flush=True)
    print(f"Mode:              {args.mode}", flush=True)
    print(f"Language:          {args.language}", flush=True)
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