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
    sys.path.insert(
        0,
        str(ROOT),
    )


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
    Load OLMo-1B, apply fixed AWQ search results, and simulate W4 quantization.

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
        "Applying pseudo W4 quantization...",
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
# HELLASWAG DATASET
# =============================================================================


def load_hellaswag_dataset(split: str):
    """Load HellaSwag with a fallback dataset identifier."""
    attempts = [
        "hellaswag",
        "Rowan/hellaswag",
    ]

    last_error: Optional[Exception] = None

    for dataset_name in attempts:
        try:
            return load_dataset(
                dataset_name,
                split=split,
            )

        except Exception as error:
            last_error = error

    raise RuntimeError(
        "Could not load HellaSwag. "
        f"Last error: {last_error}"
    )


def load_hellaswag_examples(
    split: str,
    limit: Optional[int],
    eval_seed: int,
) -> tuple[List[Dict], int]:
    """
    Load a reproducibly sampled HellaSwag subset.

    Original indices are attached before shuffling. Identical split, limit,
    and eval_seed values therefore select the same examples for FP16,
    Naive W4, Super W4, GPTQ, and AWQ.
    """
    dataset = load_hellaswag_dataset(
        split=split,
    )

    total_available = len(dataset)

    if total_available == 0:
        raise RuntimeError(
            f"HellaSwag split {split!r} is empty."
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
        try:
            label = int(
                row["label"]
            )
        except (KeyError, TypeError, ValueError):
            continue

        endings = row.get(
            "endings",
            None,
        )

        if endings is None:
            continue

        endings = list(
            endings
        )

        if len(endings) != 4:
            continue

        if label not in {0, 1, 2, 3}:
            continue

        examples.append(
            {
                "id": int(
                    row["_original_index"]
                ),
                "dataset_id": row.get(
                    "ind",
                    str(row["_original_index"]),
                ),
                "ctx": str(
                    row.get("ctx", "")
                ),
                "ctx_a": str(
                    row.get("ctx_a", "")
                ),
                "ctx_b": str(
                    row.get("ctx_b", "")
                ),
                "activity_label": str(
                    row.get("activity_label", "")
                ),
                "endings": endings,
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
    """
    Construct the HellaSwag prompt.

    Prefer the complete `ctx` field. Use `ctx_a` and `ctx_b` only as fallback.
    """
    ctx = str(
        example.get("ctx", "")
    ).strip()

    if ctx:
        return ctx

    ctx_a = str(
        example.get("ctx_a", "")
    ).strip()

    ctx_b = str(
        example.get("ctx_b", "")
    ).strip()

    if ctx_b:
        return f"{ctx_a} {ctx_b}".strip()

    return ctx_a


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
    max_length: int = 512,
    normalize_by_length: bool = True,
) -> Dict[str, float | int | bool]:
    """
    Calculate the conditional log-likelihood of one HellaSwag continuation.

    Long sequences are truncated from the left so that the complete ending and
    the most recent context are retained.
    """
    prompt = prompt.strip()
    continuation = continuation.strip()

    full_text = f"{prompt} {continuation}".strip()

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

    original_choice_length = (
        original_full_length
        - original_prompt_length
    )

    if original_choice_length <= 0:
        return {
            "score": float("-inf"),
            "sum_log_likelihood": float("-inf"),
            "num_choice_tokens": 0,
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
            "num_choice_tokens": 0,
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

    # The logit at prompt_length - 1 predicts the first continuation token.
    choice_start = max(
        retained_prompt_length - 1,
        0,
    )

    choice_log_probabilities = token_log_probabilities[
        :,
        choice_start:,
    ]

    num_choice_tokens = int(
        choice_log_probabilities.numel()
    )

    if num_choice_tokens == 0:
        return {
            "score": float("-inf"),
            "sum_log_likelihood": float("-inf"),
            "num_choice_tokens": 0,
            "prompt_tokens": original_prompt_length,
            "full_tokens": original_full_length,
            "truncated_tokens": truncated_tokens,
            "was_truncated": truncated_tokens > 0,
        }

    sum_log_likelihood = float(
        choice_log_probabilities.sum().item()
    )

    score = sum_log_likelihood

    if normalize_by_length:
        score /= num_choice_tokens

    return {
        "score": float(score),
        "sum_log_likelihood": sum_log_likelihood,
        "num_choice_tokens": num_choice_tokens,
        "prompt_tokens": original_prompt_length,
        "full_tokens": original_full_length,
        "truncated_tokens": truncated_tokens,
        "was_truncated": truncated_tokens > 0,
    }


def evaluate_hellaswag(
    model,
    tokenizer,
    examples: List[Dict],
    device: torch.device,
    max_length: int = 512,
    normalize_by_length: bool = True,
) -> Dict:
    """Evaluate the AWQ-quantized model on HellaSwag."""
    model.eval()

    num_correct = 0
    predictions: List[Dict] = []
    total_examples = len(examples)

    progress_bar = tqdm(
        examples,
        desc="Evaluating HellaSwag AWQ",
    )

    for example in progress_bar:
        prompt = build_prompt(
            example
        )

        choice_results = [
            score_continuation(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                continuation=ending,
                device=device,
                max_length=max_length,
                normalize_by_length=normalize_by_length,
            )
            for ending in example["endings"]
        ]

        scores = [
            float(result["score"])
            for result in choice_results
        ]

        prediction = int(
            torch.tensor(
                scores,
                dtype=torch.float32,
            ).argmax().item()
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
                "dataset_id": example.get(
                    "dataset_id"
                ),
                "activity_label": example.get(
                    "activity_label"
                ),
                "prediction": prediction,
                "gold": gold,
                "correct": bool(correct),
                "scores": scores,
                "sum_log_likelihoods": [
                    float(
                        result["sum_log_likelihood"]
                    )
                    for result in choice_results
                ],
                "choice_token_counts": [
                    int(
                        result["num_choice_tokens"]
                    )
                    for result in choice_results
                ],
                "truncated_tokens": [
                    int(
                        result["truncated_tokens"]
                    )
                    for result in choice_results
                ],
                "was_truncated": [
                    bool(
                        result["was_truncated"]
                    )
                    for result in choice_results
                ],
            }
        )

        progress_bar.set_postfix(
            accuracy=f"{num_correct / len(predictions):.4f}"
        )

    if total_examples == 0:
        raise RuntimeError(
            "No HellaSwag examples were evaluated."
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
            "Evaluate a fixed AWQ-quantized OLMo-1B model on a "
            "reproducibly sampled HellaSwag subset."
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
            "Seed used only to shuffle the HellaSwag evaluation data. "
            "Use the same value as for FP16, Naive W4, Super W4, and GPTQ."
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
        default=512,
    )

    parser.add_argument(
        "--normalize-by-length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Normalize each candidate log-likelihood by the number of "
            "continuation tokens. Enabled by default."
        ),
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
    print("AWQ HELLASWAG EVALUATION", flush=True)
    print("=" * 78, flush=True)
    print(f"Model path:              {args.model_path}", flush=True)
    print(f"AWQ path:                {args.awq_path}", flush=True)
    print(f"Weight bits:             {args.w_bit}", flush=True)
    print(f"Group size:              {args.q_group_size}", flush=True)
    print(f"Evaluation split:        {args.split}", flush=True)
    print(f"Evaluation limit:        {args.limit}", flush=True)
    print(f"Evaluation seed:         {args.eval_seed}", flush=True)
    print(f"Maximum length:          {args.max_length}", flush=True)
    print(
        f"Normalize by length:     "
        f"{args.normalize_by_length}",
        flush=True,
    )
    print(f"Device:                  {device}", flush=True)
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

    print(
        f"Loading HellaSwag split={args.split}, "
        f"limit={args.limit}, "
        f"eval_seed={args.eval_seed}",
        flush=True,
    )

    examples, total_available = load_hellaswag_examples(
        split=args.split,
        limit=args.limit,
        eval_seed=args.eval_seed,
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

    evaluation_start = time.time()

    metrics = evaluate_hellaswag(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        device=device,
        max_length=args.max_length,
        normalize_by_length=args.normalize_by_length,
    )

    evaluation_seconds = (
        time.time() - evaluation_start
    )

    result = {
        "benchmark": "hellaswag",
        "model": "olmo-1b",
        "model_path": args.model_path,
        "method": "awq_fake_quant",
        "metric": (
            "accuracy_normalized"
            if args.normalize_by_length
            else "accuracy"
        ),
        "awq_path": args.awq_path,
        "w_bit": args.w_bit,
        "q_group_size": args.q_group_size,
        "split": args.split,
        "available_examples": total_available,
        "requested_limit": args.limit,
        "evaluated_examples": len(examples),
        "eval_seed": args.eval_seed,
        "selected_example_ids": selected_example_ids,
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
    print("FINAL AWQ HELLASWAG RESULT", flush=True)
    print("=" * 78, flush=True)
    print(f"Evaluation seed:   {args.eval_seed}", flush=True)
    print(f"Metric:            {result['metric']}", flush=True)
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