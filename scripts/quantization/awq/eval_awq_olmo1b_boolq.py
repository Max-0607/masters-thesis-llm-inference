from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from quantization.awq.awq.quantize.pre_quant import apply_awq
from quantization.awq.awq.quantize.quantizer import (
    pseudo_quantize_model_weight,
)


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


def make_json_safe(obj: Any) -> Any:
    """Replace non-finite floats so the result can be serialized."""
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
    """Calculate the binomial standard error of an accuracy estimate."""
    if num_examples <= 0:
        raise ValueError(
            "num_examples must be greater than zero."
        )

    return math.sqrt(
        accuracy * (1.0 - accuracy) / num_examples
    )


def resolve_device(device_name: Optional[str]) -> torch.device:
    """
    Resolve the requested device.

    When no device is provided, CUDA is used when available.
    """
    if device_name is None:
        return torch.device(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device_name!r} was requested, "
            "but CUDA is not available."
        )

    return device


# =============================================================================
# AWQ MODEL LOADING
# =============================================================================


def load_awq_fake_olmo1b(
    model_path: str,
    awq_path: str,
    device: torch.device,
    w_bit: int = 4,
    q_group_size: int = 128,
):
    """
    Load OLMo-1B, apply precomputed AWQ search results, and simulate
    group-wise weight quantization.
    """
    if w_bit <= 0:
        raise ValueError(
            "w_bit must be greater than zero."
        )

    if q_group_size <= 0:
        raise ValueError(
            "q_group_size must be greater than zero."
        )

    awq_file = Path(awq_path)

    if not awq_file.exists():
        raise FileNotFoundError(
            f"AWQ result file does not exist: {awq_file}"
        )

    q_config = {
        "zero_point": True,
        "q_group_size": q_group_size,
    }

    print(
        f"Loading model configuration: {model_path}",
        flush=True,
    )

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    config.use_cache = False

    print(
        f"Loading tokenizer: {model_path}",
        flush=True,
    )

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

    if tokenizer.pad_token_id is None:
        raise ValueError(
            "Tokenizer has neither a pad token nor a usable EOS token."
        )

    torch_dtype = (
        torch.float16
        if device.type == "cuda"
        else torch.float32
    )

    print(
        f"Loading model: {model_path} "
        f"with dtype={torch_dtype}",
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=None,
    )

    print(
        f"Loading AWQ results from: {awq_file}",
        flush=True,
    )

    awq_results = torch.load(
        awq_file,
        map_location="cpu",
    )

    print(
        "Applying AWQ scaling and clipping results...",
        flush=True,
    )

    apply_awq(
        model,
        awq_results,
    )

    print(
        "Applying pseudo weight quantization: "
        f"w_bit={w_bit}, "
        f"group_size={q_group_size}",
        flush=True,
    )

    pseudo_quantize_model_weight(
        model,
        w_bit=w_bit,
        q_config=q_config,
    )

    model = model.to(device)
    model.eval()

    return model, tokenizer


# =============================================================================
# BOOLQ DATASET
# =============================================================================


def load_boolq_examples(
    split: str,
    limit: Optional[int],
    eval_seed: int,
) -> tuple[List[Dict], int]:
    """
    Load and deterministically shuffle BoolQ.

    The original dataset index is stored before shuffling. Therefore,
    the same split, limit, and eval_seed result in the same examples
    across FP16, Naive W4, Super W4, GPTQ, and AWQ.
    """
    dataset = load_dataset(
        "super_glue",
        "boolq",
        split=split,
    )

    total_available = len(dataset)

    if total_available == 0:
        raise RuntimeError(
            f"BoolQ split {split!r} is empty."
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
        examples.append(
            {
                "id": int(row["_original_index"]),
                "passage": row["passage"],
                "question": row["question"],
                "label": int(row["label"]),
            }
        )

    if not examples:
        raise RuntimeError(
            "No valid BoolQ examples were selected."
        )

    return examples, total_available


def build_prompt_boolq(example: Dict) -> str:
    """
    Build exactly the same prompt used by the other Table 5.5 methods.
    """
    passage = example["passage"].strip()
    question = example["question"].strip()

    return (
        f"{passage}\n"
        f"Question: {question}\n"
        f"Answer:"
    )


# =============================================================================
# BOOLQ SCORING
# =============================================================================


@torch.no_grad()
def score_continuation(
    model,
    tokenizer,
    prompt: str,
    continuation: str,
    device: torch.device,
    max_length: int = 512,
    normalize_by_length: bool = False,
) -> float:
    """
    Calculate the conditional log-likelihood of one answer candidate.
    """
    prompt = prompt.strip()
    continuation = continuation.strip()

    full_text = prompt + " " + continuation

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

    if continuation_log_probabilities.numel() == 0:
        return float("-inf")

    score = continuation_log_probabilities.sum().item()

    if normalize_by_length:
        score /= continuation_log_probabilities.numel()

    return float(score)


def evaluate_boolq(
    model,
    tokenizer,
    examples: List[Dict],
    device: torch.device,
    max_length: int = 512,
    normalize_by_length: bool = False,
) -> Dict:
    """Evaluate the AWQ model on the selected BoolQ examples."""
    predictions = []
    num_correct = 0

    model.eval()

    progress_bar = tqdm(
        examples,
        desc="Evaluating BoolQ AWQ",
    )

    for example_number, example in enumerate(
        progress_bar,
        start=1,
    ):
        prompt = build_prompt_boolq(
            example
        )

        yes_score = score_continuation(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            continuation="yes",
            device=device,
            max_length=max_length,
            normalize_by_length=normalize_by_length,
        )

        no_score = score_continuation(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            continuation="no",
            device=device,
            max_length=max_length,
            normalize_by_length=normalize_by_length,
        )

        prediction = (
            1
            if yes_score > no_score
            else 0
        )

        gold = int(example["label"])
        correct = int(prediction == gold)

        num_correct += correct

        predictions.append(
            {
                "id": int(example["id"]),
                "question": example["question"],
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
                "yes_score": float(yes_score),
                "no_score": float(no_score),
                "margin": float(
                    yes_score - no_score
                ),
            }
        )

        progress_bar.set_postfix(
            accuracy=f"{num_correct / example_number:.4f}"
        )

    num_examples = len(examples)

    if num_examples == 0:
        raise RuntimeError(
            "No BoolQ examples were evaluated."
        )

    accuracy = num_correct / num_examples

    accuracy_stderr = calculate_accuracy_stderr(
        accuracy=accuracy,
        num_examples=num_examples,
    )

    return {
        "num_examples": num_examples,
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
            "Load an AWQ-processed OLMo-1B model and evaluate it "
            "on a reproducibly sampled BoolQ subset."
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
        "--output-json",
        required=True,
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
            "Use the same seed as for the other Table 5.5 methods."
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
        action="store_true",
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Torch device, for example cuda:0 or cpu. "
            "Defaults to cuda:0 when available."
        ),
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

    set_all_seeds(
        args.eval_seed
    )

    device = resolve_device(
        args.device
    )

    output_path = Path(
        args.output_json
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78, flush=True)
    print("AWQ BOOLQ EVALUATION", flush=True)
    print("=" * 78, flush=True)
    print(f"Model path:             {args.model_path}", flush=True)
    print(f"AWQ path:               {args.awq_path}", flush=True)
    print(f"Weight bits:            {args.w_bit}", flush=True)
    print(f"Group size:             {args.q_group_size}", flush=True)
    print(f"Evaluation split:       {args.split}", flush=True)
    print(f"Evaluation limit:       {args.limit}", flush=True)
    print(f"Evaluation seed:        {args.eval_seed}", flush=True)
    print(f"Maximum length:         {args.max_length}", flush=True)
    print(
        f"Normalize by length:    "
        f"{args.normalize_by_length}",
        flush=True,
    )
    print(f"Device:                 {device}", flush=True)
    print("=" * 78, flush=True)

    model, tokenizer = load_awq_fake_olmo1b(
        model_path=args.model_path,
        awq_path=args.awq_path,
        device=device,
        w_bit=args.w_bit,
        q_group_size=args.q_group_size,
    )

    print(
        f"Loading BoolQ split={args.split}, "
        f"limit={args.limit}, "
        f"eval_seed={args.eval_seed}",
        flush=True,
    )

    examples, total_available = load_boolq_examples(
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
        f"{total_available}",
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

    metrics = evaluate_boolq(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        device=device,
        max_length=args.max_length,
        normalize_by_length=args.normalize_by_length,
    )

    result = {
        "benchmark": "boolq",
        "task": "boolq",
        "method": "awq",
        "model": "olmo-1b",
        "model_path": args.model_path,
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
        "device": str(device),
        **metrics,
    }

    result = make_json_safe(
        result
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 78, flush=True)
    print("FINAL AWQ BOOLQ RESULT", flush=True)
    print("=" * 78, flush=True)
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