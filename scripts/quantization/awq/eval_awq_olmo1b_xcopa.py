from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)


# =============================================================================
# PROJECT ROOT
# =============================================================================


ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from quantization.awq.awq.quantize.pre_quant import apply_awq
from quantization.awq.awq.quantize.quantizer import (
    pseudo_quantize_model_weight,
)


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
# AWQ MODEL
# =============================================================================


def load_awq_fake_olmo1b(
    model_path: str,
    awq_path: str,
    device: torch.device,
    w_bit: int = 4,
    q_group_size: int = 128,
):
    """
    Load OLMo-1B, apply the fixed AWQ search results, and simulate W4
    quantization.

    The AWQ artifact remains identical across all evaluation seeds.
    """
    q_config = {
        "zero_point": True,
        "q_group_size": q_group_size,
    }

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )

    if (
        tokenizer.pad_token is None
        and tokenizer.eos_token is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,
    )

    print(
        f"Loading AWQ results from: {awq_path}",
        flush=True,
    )

    awq_results = torch.load(
        awq_path,
        map_location="cpu",
    )

    print(
        "Applying AWQ transformations...",
        flush=True,
    )

    apply_awq(
        model,
        awq_results,
    )

    print(
        f"Applying pseudo weight quantization: "
        f"w_bit={w_bit}, "
        f"group_size={q_group_size}",
        flush=True,
    )

    pseudo_quantize_model_weight(
        model,
        w_bit=w_bit,
        q_config=q_config,
    )

    model = model.to(
        device
    )

    model.eval()

    return model, tokenizer


# =============================================================================
# LANGUAGE-SPECIFIC PROMPTS
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
    """Build the same prompt used by all other XCOPA/COPA methods."""
    language_config = get_language_config(
        language
    )

    question_text = language_config[
        example["question"]
    ]

    premise = example[
        "premise"
    ].strip()

    return (
        f"{premise}\n"
        f"{question_text}\n"
        f"Answer:"
    )


# =============================================================================
# COPA / XCOPA DATASET
# =============================================================================


def load_xcopa_dataset(
    language: str,
    split: str,
):
    """
    Load English COPA or multilingual XCOPA.

    English is taken from SuperGLUE COPA because XCOPA has no English
    configuration.
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

    Original indices are attached before shuffling. For English COPA
    validation, limit=500 selects all 100 examples, so the seed changes only
    their order and not their membership.
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

    examples: List[Dict] = []

    for row in dataset:
        label = row.get(
            "label",
            row.get("answer"),
        )

        try:
            label = int(label)
        except (TypeError, ValueError):
            continue

        if label not in {0, 1}:
            continue

        question = str(
            row["question"]
        )

        if question not in {"cause", "effect"}:
            continue

        examples.append(
            {
                "id": int(
                    row["_original_index"]
                ),
                "premise": str(
                    row["premise"]
                ),
                "question": question,
                "choice1": str(
                    row["choice1"]
                ),
                "choice2": str(
                    row["choice2"]
                ),
                "label": label,
            }
        )

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
) -> Dict[str, float | int | bool]:
    """
    Calculate the conditional log-likelihood of one candidate continuation.
    """
    prompt = prompt.strip()
    continuation = continuation.strip()

    full_text = (
        f"{prompt} {continuation}".strip()
    )

    model_max_length = getattr(
        model.config,
        "max_position_embeddings",
        max_length,
    )

    effective_max_length = min(
        int(max_length),
        int(model_max_length),
    )

    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]

    full_ids = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]

    original_prompt_length = int(
        prompt_ids.shape[1]
    )

    original_full_length = int(
        full_ids.shape[1]
    )

    original_continuation_length = (
        original_full_length
        - original_prompt_length
    )

    if original_continuation_length <= 0:
        return {
            "score": float("-inf"),
            "sum_log_likelihood": float("-inf"),
            "num_continuation_tokens": 0,
            "prompt_tokens": original_prompt_length,
            "full_tokens": original_full_length,
            "truncated_tokens": 0,
            "was_truncated": False,
        }

    truncated_tokens = max(
        original_full_length - effective_max_length,
        0,
    )

    if truncated_tokens > 0:
        full_ids = full_ids[
            :,
            truncated_tokens:,
        ]

    retained_prompt_length = max(
        original_prompt_length - truncated_tokens,
        0,
    )

    input_ids = full_ids.to(
        device
    )

    if input_ids.shape[1] <= 1:
        return {
            "score": float("-inf"),
            "sum_log_likelihood": float("-inf"),
            "num_continuation_tokens": 0,
            "prompt_tokens": original_prompt_length,
            "full_tokens": original_full_length,
            "truncated_tokens": truncated_tokens,
            "was_truncated": truncated_tokens > 0,
        }

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
        retained_prompt_length - 1,
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
            "prompt_tokens": original_prompt_length,
            "full_tokens": original_full_length,
            "truncated_tokens": truncated_tokens,
            "was_truncated": truncated_tokens > 0,
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
        "prompt_tokens": original_prompt_length,
        "full_tokens": original_full_length,
        "truncated_tokens": truncated_tokens,
        "was_truncated": truncated_tokens > 0,
    }


def evaluate_xcopa(
    model,
    tokenizer,
    examples: List[Dict],
    language: str,
    device: torch.device,
    max_length: int = 256,
    normalize_by_length: bool = False,
) -> Dict:
    """Evaluate the AWQ-quantized model on COPA/XCOPA."""
    model.eval()

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

    progress_bar = tqdm(
        examples,
        desc=f"Evaluating COPA/XCOPA-{language} AWQ",
    )

    for example in progress_bar:
        prompt = build_prompt(
            example=example,
            language=language,
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
                "truncated_tokens_choice1": int(
                    choice1_result[
                        "truncated_tokens"
                    ]
                ),
                "truncated_tokens_choice2": int(
                    choice2_result[
                        "truncated_tokens"
                    ]
                ),
                "margin": float(
                    score1 - score2
                ),
            }
        )

        progress_bar.set_postfix(
            accuracy=(
                f"{num_correct / len(predictions):.4f}"
            )
        )

    total_examples = len(
        examples
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
        question_total = stats[
            "total"
        ]

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
            "Evaluate a fixed AWQ-quantized OLMo-1B model on "
            "English COPA or multilingual XCOPA."
        )
    )

    parser.add_argument(
        "--model-path",
        default="models/olmo1b",
    )

    parser.add_argument(
        "--awq-path",
        default=(
            "quantization/awq/olmo1b/"
            "olmo1b-w4-g128.pt4"
        ),
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
        help=(
            "Seed used to shuffle the evaluation dataset. For English COPA "
            "with limit >= 100, only the order changes."
        ),
    )

    parser.add_argument(
        "--w-bit",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--q-group-size",
        type=int,
        default=128,
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
        "--device",
        default="cuda:0",
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

    if args.w_bit <= 0:
        raise ValueError(
            "--w-bit must be greater than zero."
        )

    if args.q_group_size <= 0:
        raise ValueError(
            "--q-group-size must be greater than zero."
        )

    if args.max_length <= 1:
        raise ValueError(
            "--max-length must be greater than one."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this AWQ evaluation."
        )

    device = torch.device(
        args.device
    )

    set_all_seeds(
        args.eval_seed
    )

    output_path = Path(
        args.output_json
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78, flush=True)
    print("AWQ COPA / XCOPA EVALUATION", flush=True)
    print("=" * 78, flush=True)
    print(f"Model path:             {args.model_path}", flush=True)
    print(f"AWQ path:               {args.awq_path}", flush=True)
    print(f"Language:               {args.language}", flush=True)
    print(f"Weight bits:            {args.w_bit}", flush=True)
    print(f"Group size:             {args.q_group_size}", flush=True)
    print(f"Evaluation split:       {args.split}", flush=True)
    print(f"Requested limit:        {args.limit}", flush=True)
    print(f"Evaluation seed:        {args.eval_seed}", flush=True)
    print(f"Maximum length:         {args.max_length}", flush=True)
    print(
        f"Normalize by length:   "
        f"{args.normalize_by_length}",
        flush=True,
    )
    print(f"Device:                 {device}", flush=True)
    print("=" * 78, flush=True)

    loading_start = time.time()

    model, tokenizer = load_awq_fake_olmo1b(
        model_path=args.model_path,
        awq_path=args.awq_path,
        device=device,
        w_bit=args.w_bit,
        q_group_size=args.q_group_size,
    )

    loading_seconds = (
        time.time() - loading_start
    )

    print(
        f"AWQ model prepared in "
        f"{loading_seconds:.2f} seconds.",
        flush=True,
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
        f"Available examples:     "
        f"{total_available}",
        flush=True,
    )

    print(
        f"Selected examples:      "
        f"{len(examples)}",
        flush=True,
    )

    if (
        args.language == "en"
        and args.limit >= total_available
    ):
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

    evaluation_start = time.time()

    metrics = evaluate_xcopa(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        language=args.language,
        device=device,
        max_length=args.max_length,
        normalize_by_length=args.normalize_by_length,
    )

    evaluation_seconds = (
        time.time() - evaluation_start
    )

    result = {
        "benchmark": "xcopa",
        "dataset": (
            "super_glue_copa"
            if args.language == "en"
            else "xcopa"
        ),
        "model": "olmo-1b",
        "model_path": args.model_path,
        "method": "awq_fake_quant",
        "metric": (
            "accuracy_normalized"
            if args.normalize_by_length
            else "accuracy"
        ),
        "language": args.language,
        "awq_path": args.awq_path,
        "w_bit": args.w_bit,
        "q_group_size": args.q_group_size,
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
        "normalize_by_length": args.normalize_by_length,
        "model_loading_seconds": loading_seconds,
        "evaluation_seconds": evaluation_seconds,
        **metrics,
    }

    result = make_json_safe(
        result
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
    print("FINAL AWQ COPA / XCOPA RESULT", flush=True)
    print("=" * 78, flush=True)
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
    print(
        f"Evaluation time:   "
        f"{evaluation_seconds:.2f} seconds",
        flush=True,
    )
    print(f"Saved result to:   {output_path}", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()