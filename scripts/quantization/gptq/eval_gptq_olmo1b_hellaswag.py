from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from quantization.gptq.gptq_repo.eval_gptq_olmo1b_ppl import (
    DEV,
    get_olmo,
    get_wikitext2_olmo,
    olmo_sequential,
)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================


def set_all_seeds(seed: int) -> None:
    """Set Python and PyTorch seeds."""
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
    """Replace non-finite floats so the result can be stored as JSON."""
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
# HELLASWAG DATASET
# =============================================================================


def preprocess_hellaswag_text(text: str) -> str:
    """
    Apply the same basic HellaSwag preprocessing as in the original script.
    """
    return (
        text
        .replace(" [title]", ". ")
        .replace("[title]", "")
        .strip()
    )


def load_hellaswag_examples(
    split: str,
    limit: Optional[int],
    eval_seed: int,
) -> tuple[List[Dict], int]:
    """
    Load a reproducibly sampled HellaSwag subset.

    Original dataset indices are added before shuffling. Therefore, identical
    split, limit, and eval_seed values select the same examples across FP16,
    Naive W4, Super W4, GPTQ, and AWQ.
    """
    dataset = load_dataset(
        "hellaswag",
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
        ctx_a = preprocess_hellaswag_text(
            row["ctx_a"]
        )

        ctx_b = preprocess_hellaswag_text(
            row["ctx_b"]
        )

        prompt = (
            f"{ctx_a} {ctx_b}".strip()
            if ctx_b
            else ctx_a
        )

        endings = [
            " " + ending.strip()
            for ending in row["endings"]
        ]

        examples.append(
            {
                "id": int(row["_original_index"]),
                "prompt": prompt,
                "endings": endings,
                "label": int(row["label"]),
                "activity_label": row.get(
                    "activity_label",
                    None,
                ),
            }
        )

    if not examples:
        raise RuntimeError(
            "No valid HellaSwag examples were selected."
        )

    return examples, total_available


# =============================================================================
# HELLASWAG SCORING
# =============================================================================


@torch.no_grad()
def score_choice(
    model,
    tokenizer,
    prompt: str,
    choice: str,
    device,
    max_length: int = 512,
    normalize_by_length: bool = True,
) -> Dict[str, float | int]:
    """
    Calculate the conditional log-likelihood of one HellaSwag ending.

    If the complete sequence is too long, tokens are removed from the left.
    This preserves the candidate ending and the most recent prompt context.
    """
    full_text = prompt + choice

    effective_max_length = min(
        max_length,
        int(model.seqlen),
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

    if original_full_length <= original_prompt_length:
        return {
            "score": float("-inf"),
            "sum_log_likelihood": float("-inf"),
            "num_choice_tokens": 0,
            "prompt_tokens": original_prompt_length,
            "full_tokens": original_full_length,
            "truncated_tokens": 0,
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

    input_ids = full_ids.to(device)

    if input_ids.shape[1] <= 1:
        return {
            "score": float("-inf"),
            "sum_log_likelihood": float("-inf"),
            "num_choice_tokens": 0,
            "prompt_tokens": original_prompt_length,
            "full_tokens": original_full_length,
            "truncated_tokens": truncated_tokens,
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
        -1,
        target_ids.unsqueeze(-1),
    ).squeeze(-1)

    # Logit position prompt_length - 1 predicts the first ending token.
    choice_start = max(
        retained_prompt_length - 1,
        0,
    )

    choice_log_probabilities = token_log_probabilities[
        :,
        choice_start:,
    ]

    if choice_log_probabilities.numel() == 0:
        return {
            "score": float("-inf"),
            "sum_log_likelihood": float("-inf"),
            "num_choice_tokens": 0,
            "prompt_tokens": original_prompt_length,
            "full_tokens": original_full_length,
            "truncated_tokens": truncated_tokens,
        }

    sum_log_likelihood = float(
        choice_log_probabilities.sum().item()
    )

    num_choice_tokens = int(
        choice_log_probabilities.numel()
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
    }


def evaluate_hellaswag(
    model,
    tokenizer,
    examples: List[Dict],
    device,
    max_length: int = 512,
    normalize_by_length: bool = True,
) -> Dict:
    """Evaluate GPTQ-quantized OLMo-1B on HellaSwag."""
    model.eval()

    num_correct = 0
    predictions = []
    total_examples = len(examples)

    for example_number, example in enumerate(
        examples,
        start=1,
    ):
        choice_results = [
            score_choice(
                model=model,
                tokenizer=tokenizer,
                prompt=example["prompt"],
                choice=choice,
                device=device,
                max_length=max_length,
                normalize_by_length=normalize_by_length,
            )
            for choice in example["endings"]
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

        gold = int(example["label"])
        correct = int(prediction == gold)

        num_correct += correct

        predictions.append(
            {
                "id": int(example["id"]),
                "prediction": prediction,
                "gold": gold,
                "correct": bool(correct),
                "scores": scores,
                "sum_log_likelihoods": [
                    float(result["sum_log_likelihood"])
                    for result in choice_results
                ],
                "choice_token_counts": [
                    int(result["num_choice_tokens"])
                    for result in choice_results
                ],
                "truncated_tokens": [
                    int(result["truncated_tokens"])
                    for result in choice_results
                ],
                "activity_label": example.get(
                    "activity_label"
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
            "No HellaSwag examples were evaluated."
        )

    accuracy = num_correct / total_examples

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
            "Quantize OLMo-1B using GPTQ and evaluate it on a "
            "reproducibly sampled HellaSwag subset."
        )
    )

    parser.add_argument(
        "--model-id",
        default="allenai/OLMo-1B-0724-hf",
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
            "Seed used to shuffle HellaSwag before applying --limit. "
            "Use the same value as for FP16, Naive W4, Super W4, "
            "and AWQ."
        ),
    )

    parser.add_argument(
        "--calibration-seed",
        type=int,
        default=0,
        help=(
            "Seed used to select the GPTQ WikiText-2 calibration data. "
            "Keep this fixed across all evaluation seeds."
        ),
    )

    parser.add_argument(
        "--nsamples",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--wbits",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--groupsize",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--act-order",
        action=argparse.BooleanOptionalAction,
        default=True,
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
            "Normalize each ending log-likelihood by its token count. "
            "Enabled by default to reproduce the original HellaSwag "
            "evaluation setting."
        ),
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

    if args.nsamples <= 0:
        raise ValueError(
            "--nsamples must be greater than zero."
        )

    if args.wbits <= 0:
        raise ValueError(
            "--wbits must be greater than zero."
        )

    if args.groupsize == 0 or args.groupsize < -1:
        raise ValueError(
            "--groupsize must be -1 or a positive integer."
        )

    if args.percdamp < 0:
        raise ValueError(
            "--percdamp must be non-negative."
        )

    if args.max_length <= 1:
        raise ValueError(
            "--max-length must be greater than one."
        )

    # GPTQ quantization is controlled only by the calibration seed.
    # This seed remains fixed for all evaluation runs.
    set_all_seeds(
        args.calibration_seed
    )

    output_path = Path(
        args.output_json
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    gptq_args = {
        "nsamples": args.nsamples,
        "wbits": args.wbits,
        "groupsize": args.groupsize,
        "percdamp": args.percdamp,
        "act_order": args.act_order,
    }

    print("=" * 78, flush=True)
    print("GPTQ HELLASWAG EVALUATION", flush=True)
    print("=" * 78, flush=True)
    print(f"Model ID:               {args.model_id}", flush=True)
    print(f"Weight bits:            {args.wbits}", flush=True)
    print(f"Group size:             {args.groupsize}", flush=True)
    print(f"Calibration samples:    {args.nsamples}", flush=True)
    print(f"Calibration seed:       {args.calibration_seed}", flush=True)
    print(f"Percent damping:        {args.percdamp}", flush=True)
    print(f"Activation order:       {args.act_order}", flush=True)
    print(f"Evaluation split:       {args.split}", flush=True)
    print(f"Evaluation limit:       {args.limit}", flush=True)
    print(f"Evaluation seed:        {args.eval_seed}", flush=True)
    print(f"Maximum length:         {args.max_length}", flush=True)
    print(
        f"Normalize by length:    "
        f"{args.normalize_by_length}",
        flush=True,
    )
    print("=" * 78, flush=True)

    print(
        f"Loading tokenizer: {args.model_id}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )

    if (
        tokenizer.pad_token is None
        and tokenizer.eos_token is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token

    print(
        f"Loading model: {args.model_id}",
        flush=True,
    )

    model = get_olmo(
        args.model_id
    )

    model.eval()

    print(
        "Loading WikiText-2 calibration data...",
        flush=True,
    )

    calibration_loader, _ = get_wikitext2_olmo(
        nsamples=args.nsamples,
        seed=args.calibration_seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )

    print(
        "Running GPTQ quantization...",
        flush=True,
    )

    quantization_start = time.time()

    quantizers = olmo_sequential(
        model,
        calibration_loader,
        DEV,
        gptq_args,
    )

    quantization_seconds = (
        time.time() - quantization_start
    )

    print(
        f"GPTQ quantization completed in "
        f"{quantization_seconds:.2f} seconds.",
        flush=True,
    )

    print(
        f"Quantized modules returned: "
        f"{len(quantizers)}",
        flush=True,
    )

    model = model.to(DEV)
    model.eval()

    # Only the evaluation sample changes between seeds.
    set_all_seeds(
        args.eval_seed
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

    print(
        "Evaluating GPTQ model on HellaSwag...",
        flush=True,
    )

    metrics = evaluate_hellaswag(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        device=DEV,
        max_length=args.max_length,
        normalize_by_length=args.normalize_by_length,
    )

    result = {
        "benchmark": "hellaswag",
        "model": "olmo-1b",
        "model_id": args.model_id,
        "method": "gptq_runtime",
        "bits": args.wbits,
        "groupsize": args.groupsize,
        "nsamples": args.nsamples,
        "percdamp": args.percdamp,
        "act_order": args.act_order,
        "calibration_dataset": "wikitext2",
        "calibration_seed": args.calibration_seed,
        "quantization_seconds": quantization_seconds,
        "split": args.split,
        "available_examples": total_available,
        "requested_limit": args.limit,
        "evaluated_examples": len(examples),
        "eval_seed": args.eval_seed,
        "selected_example_ids": selected_example_ids,
        "max_length": args.max_length,
        "normalize_by_length": args.normalize_by_length,
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
    print("FINAL GPTQ HELLASWAG RESULT", flush=True)
    print("=" * 78, flush=True)
    print(f"Evaluation seed:   {args.eval_seed}", flush=True)
    print(f"Calibration seed:  {args.calibration_seed}", flush=True)
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