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
    subset: str,
    limit: Optional[int],
    eval_seed: int,
) -> tuple[List[Dict], int]:
    """
    Load a reproducibly sampled WinoGrande subset.

    Original dataset indices are attached before shuffling. Identical subset,
    split, limit, and eval_seed values therefore select the same examples for
    FP16, Naive W4, Super W4, GPTQ, and AWQ.
    """
    dataset = load_winogrande_dataset(
        split=split,
        subset=subset,
    )

    total_available = len(dataset)

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
                "sentence": str(
                    row["sentence"]
                ),
                "option1": str(
                    row["option1"]
                ),
                "option2": str(
                    row["option2"]
                ),
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
# PROMPT CONSTRUCTION
# =============================================================================


def build_prompt_and_choices(
    example: Dict,
) -> Dict[str, str]:
    """
    Split the sentence at the placeholder.

    The prefix is used as the prompt. Each candidate option plus the sentence
    suffix is scored as the continuation.
    """
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


# =============================================================================
# WINOGRANDE SCORING
# =============================================================================


@torch.no_grad()
def score_continuation(
    model,
    tokenizer,
    prompt: str,
    continuation: str,
    device,
    max_length: int = 512,
    normalize_by_length: bool = False,
) -> Dict[str, float | int | bool]:
    """
    Calculate the conditional log-likelihood of one continuation.

    Long sequences are truncated from the left so that the candidate
    continuation and the most recent prompt context are preserved.
    """
    prompt = prompt.strip()
    continuation = continuation.strip()

    full_text = (
        f"{prompt} {continuation}".strip()
    )

    effective_max_length = min(
        int(max_length),
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


def evaluate_winogrande(
    model,
    tokenizer,
    examples: List[Dict],
    device,
    max_length: int = 512,
    normalize_by_length: bool = False,
) -> Dict:
    """Evaluate GPTQ-quantized OLMo-1B on WinoGrande."""
    model.eval()

    num_correct = 0
    predictions: List[Dict] = []
    total_examples = len(examples)

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
                "truncated_tokens_option1": int(
                    option1_result[
                        "truncated_tokens"
                    ]
                ),
                "truncated_tokens_option2": int(
                    option2_result[
                        "truncated_tokens"
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
            "Quantize OLMo-1B using GPTQ and evaluate it on a "
            "reproducibly sampled WinoGrande subset."
        )
    )

    parser.add_argument(
        "--model-id",
        default="allenai/OLMo-1B-0724-hf",
    )

    parser.add_argument(
        "--subset",
        default="winogrande_xl",
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
            "Seed used only to shuffle WinoGrande before applying --limit. "
            "Use the same value as for FP16, Naive W4, Super W4, and AWQ."
        ),
    )

    parser.add_argument(
        "--calibration-seed",
        type=int,
        default=0,
        help=(
            "Seed used to select the GPTQ WikiText-2 calibration samples. "
            "Keep this fixed across all evaluation seeds."
        ),
    )

    parser.add_argument(
        "--nsamples",
        type=int,
        default=4,
        help=(
            "Number of GPTQ calibration samples. The original WinoGrande "
            "experiment used 4."
        ),
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
        default=False,
        help=(
            "Normalize continuation scores by token count. Disabled by "
            "default to reproduce the original WinoGrande experiment."
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

    # GPTQ quantization depends only on the fixed calibration seed.
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
    print("GPTQ WINOGRANDE EVALUATION", flush=True)
    print("=" * 78, flush=True)
    print(f"Model ID:               {args.model_id}", flush=True)
    print(f"Weight bits:            {args.wbits}", flush=True)
    print(f"Group size:             {args.groupsize}", flush=True)
    print(f"Calibration samples:    {args.nsamples}", flush=True)
    print(f"Calibration seed:       {args.calibration_seed}", flush=True)
    print(f"Percent damping:        {args.percdamp}", flush=True)
    print(f"Activation order:       {args.act_order}", flush=True)
    print(f"Subset:                 {args.subset}", flush=True)
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
        "Loading fixed WikiText-2 calibration data...",
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

    model = model.to(
        DEV
    )

    model.eval()

    # Only the evaluation sample changes here.
    set_all_seeds(
        args.eval_seed
    )

    print(
        f"Loading WinoGrande subset={args.subset}, "
        f"split={args.split}, "
        f"limit={args.limit}, "
        f"eval_seed={args.eval_seed}",
        flush=True,
    )

    examples, total_available = load_winogrande_examples(
        split=args.split,
        subset=args.subset,
        limit=args.limit,
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
        "Evaluating GPTQ model on WinoGrande...",
        flush=True,
    )

    metrics = evaluate_winogrande(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        device=DEV,
        max_length=args.max_length,
        normalize_by_length=args.normalize_by_length,
    )

    result = {
        "benchmark": "winogrande",
        "model": "olmo-1b",
        "model_id": args.model_id,
        "method": "gptq_runtime",
        "metric": (
            "accuracy_normalized"
            if args.normalize_by_length
            else "accuracy"
        ),
        "bits": args.wbits,
        "groupsize": args.groupsize,
        "nsamples": args.nsamples,
        "percdamp": args.percdamp,
        "act_order": args.act_order,
        "calibration_dataset": "wikitext2",
        "calibration_seed": args.calibration_seed,
        "quantization_seconds": quantization_seconds,
        "subset": args.subset,
        "split": args.split,
        "available_examples": total_available,
        "requested_limit": args.limit,
        "evaluated_examples": len(examples),
        "eval_seed": args.eval_seed,
        "selected_example_ids": selected_example_ids,
        "selected_qids": selected_qids,
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
    print("FINAL GPTQ WINOGRANDE RESULT", flush=True)
    print("=" * 78, flush=True)
    print(f"Evaluation seed:   {args.eval_seed}", flush=True)
    print(f"Calibration seed:  {args.calibration_seed}", flush=True)
    print(f"Calibration n:     {args.nsamples}", flush=True)
    print(f"Metric:            {result['metric']}", flush=True)
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