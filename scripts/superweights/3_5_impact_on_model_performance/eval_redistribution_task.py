from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lm_eval import evaluator, tasks
from lm_eval.base import BaseLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.hooks import get_nested_attr


MODEL_SPECS = {
    "mistral-7b": {
        "hf_name": "mistralai/Mistral-7B-v0.1",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [
            (1, 2070, 7310),
        ],
    },
    "llama-7b": {
        "hf_name": "huggyllama/llama-7b",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [
            (2, 3968, 7003),
        ],
    },
    "llama-13b": {
        "hf_name": "huggyllama/llama-13b",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [
            (2, 2231, 2278),
            (2, 2231, 6939),
        ],
    },
    "llama-30b": {
        "hf_name": "huggyllama/llama-30b",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [
            (3, 5633, 12817),
            (3, 5633, 17439),
            (10, 5633, 14386),
        ],
    },
    "llama3-8b": {
        "hf_name": "meta-llama/Meta-Llama-3-8B",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [
            (1, 788, 2427),
            (1, 1384, 2427),
            (1, 4062, 2427),
        ],
    },
    "olmo1b": {
        "hf_name": "allenai/OLMo-1B-0724-hf",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [
            (1, 1764, 1710),
            (1, 1764, 8041),
        ],
    },
    "olmo7b": {
        "hf_name": "allenai/OLMo-7B-0724-hf",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [
            (1, 269, 7467),
            (2, 269, 8275),
            (7, 269, 453),
            (24, 269, 2300),
        ],
    },
    "phi3-mini": {
        "hf_name": "microsoft/Phi-3-mini-4k-instruct",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [
            (2, 525, 808),
            (2, 1693, 808),
            (2, 1113, 808),
            (4, 525, 2723),
            (4, 1113, 2723),
            (4, 1693, 2723),
        ],
    },
}


def to_sw_dicts(tuples: list[tuple[int, int, int]]) -> list[dict[str, int]]:
    """Convert superweight tuples to dictionaries."""
    return [
        {
            "layer": layer,
            "row": row,
            "col": col,
        }
        for layer, row, col in tuples
    ]


def set_all_seeds(seed: int) -> None:
    """
    Set random seeds for reproducible model evaluation.

    This controls Python, NumPy, PyTorch CPU, and PyTorch CUDA randomness.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def shuffle_task_documents(task: Any, seed: int) -> tuple[str, int]:
    """
    Shuffle the evaluation documents deterministically.

    lm-eval normally evaluates the first N documents when --limit is used.
    By shuffling the task documents before evaluation, every evaluation seed
    produces a different but reproducible subset.

    Returns
    -------
    split_name:
        Name of the shuffled split, e.g. "validation" or "test".
    document_count:
        Total number of available documents before --limit is applied.
    """
    rng = random.Random(seed)

    if task.has_validation_docs():
        documents = list(task.validation_docs())
        rng.shuffle(documents)

        # Default arguments prevent late-binding closure problems.
        task.validation_docs = lambda docs=documents: docs

        return "validation", len(documents)

    if task.has_test_docs():
        documents = list(task.test_docs())
        rng.shuffle(documents)

        task.test_docs = lambda docs=documents: docs

        return "test", len(documents)

    if task.has_training_docs():
        documents = list(task.training_docs())
        rng.shuffle(documents)

        task.training_docs = lambda docs=documents: docs

        return "training", len(documents)

    raise ValueError(
        "The selected task has no validation, test, or training documents."
    )


class LoadedHFLM(BaseLM):
    """
    Minimal lm-evaluation-harness wrapper around an already loaded
    Hugging Face causal language model.
    """

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return 2048

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return next(self.model.parameters()).device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(
            string,
            add_special_tokens=False,
        )

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        first_device = next(self.model.parameters()).device

        with torch.no_grad():
            outputs = self.model(inps.to(first_device))

        return outputs.logits

    def _model_generate(self, context, max_length, eos_token_id):
        first_device = next(self.model.parameters()).device

        with torch.no_grad():
            return self.model.generate(
                context.to(first_device),
                max_length=max_length,
                eos_token_id=eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
            )


def get_down_proj_weight(
    model,
    layer: int,
    layer_path: str,
    down_proj_path: str,
):
    """Return the down-projection weight tensor for a specific layer."""
    layers = get_nested_attr(model, layer_path)
    module = get_nested_attr(
        layers[layer],
        down_proj_path,
    )

    return module.weight


def validate_position(
    weight: torch.Tensor,
    layer: int,
    row: int,
    col: int,
) -> None:
    """Check whether a requested weight coordinate exists."""
    rows, cols = weight.shape

    if row < 0 or row >= rows:
        raise IndexError(
            f"Invalid row {row} in layer {layer}. "
            f"Weight shape is {tuple(weight.shape)}."
        )

    if col < 0 or col >= cols:
        raise IndexError(
            f"Invalid column {col} in layer {layer}. "
            f"Weight shape is {tuple(weight.shape)}."
        )


def ablate_positions(
    model,
    positions: list[dict[str, Any]],
    layer_path: str,
    down_proj_path: str,
) -> None:
    """Set selected individual weights to zero."""
    with torch.no_grad():
        for position in positions:
            layer = int(position["layer"])
            row = int(position["row"])
            col = int(position["col"])

            weight = get_down_proj_weight(
                model=model,
                layer=layer,
                layer_path=layer_path,
                down_proj_path=down_proj_path,
            )

            validate_position(
                weight=weight,
                layer=layer,
                row=row,
                col=col,
            )

            old_value = float(
                weight[row, col]
                .detach()
                .float()
                .cpu()
                .item()
            )

            weight[row, col] = 0.0

            position["old_value"] = old_value

            print(
                "Ablated "
                f"layer={layer} "
                f"row={row} "
                f"col={col} "
                f"old_value={old_value}"
            )


def sample_random_positions(
    model,
    superweights: list[dict[str, int]],
    layer_path: str,
    down_proj_path: str,
    seed: int,
) -> list[dict[str, int]]:
    """
    Select random non-superweight positions.

    For comparability, one random position is selected in the same layer
    for every listed superweight.
    """
    rng = random.Random(seed)

    superweight_set = {
        (
            sw["layer"],
            sw["row"],
            sw["col"],
        )
        for sw in superweights
    }

    selected_set: set[tuple[int, int, int]] = set()
    random_positions: list[dict[str, int]] = []

    for superweight in superweights:
        layer = superweight["layer"]

        weight = get_down_proj_weight(
            model=model,
            layer=layer,
            layer_path=layer_path,
            down_proj_path=down_proj_path,
        )

        rows, cols = weight.shape

        while True:
            row = rng.randrange(rows)
            col = rng.randrange(cols)

            candidate = (
                layer,
                row,
                col,
            )

            if (
                candidate not in superweight_set
                and candidate not in selected_set
            ):
                break

        selected_set.add(candidate)

        random_positions.append(
            {
                "layer": layer,
                "row": row,
                "col": col,
            }
        )

    return random_positions


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate baseline, random-ablation, or superweight-ablation "
            "performance with reproducible evaluation subsets."
        )
    )

    parser.add_argument(
        "--model-key",
        choices=list(MODEL_SPECS.keys()),
        required=True,
    )

    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Optional local model path or Hugging Face model ID. "
            "If omitted, MODEL_SPECS['hf_name'] is used."
        ),
    )

    parser.add_argument(
        "--task",
        default="hellaswag",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help=(
            "Number of shuffled task examples to evaluate. "
            "Use the same limit for all conditions."
        ),
    )

    parser.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help=(
            "Seed used to shuffle evaluation examples before --limit "
            "is applied."
        ),
    )

    parser.add_argument(
        "--ablate-superweights",
        action="store_true",
        help="Set all predefined superweights of the selected model to zero.",
    )

    parser.add_argument(
        "--ablate-random",
        action="store_true",
        help=(
            "Ablate the same number of randomly selected weights as "
            "predefined superweights."
        ),
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help=(
            "Seed used to select random weight positions for random ablation."
        ),
    )

    parser.add_argument(
        "--output-json",
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be greater than zero.")

    if args.ablate_superweights and args.ablate_random:
        raise ValueError(
            "Use only one ablation mode: "
            "--ablate-superweights or --ablate-random."
        )

    # Set all general random seeds before loading the model.
    set_all_seeds(args.eval_seed)

    spec = MODEL_SPECS[args.model_key]

    model_path = (
        args.model_path
        if args.model_path is not None
        else spec["hf_name"]
    )

    layer_path = spec["layer_path"]
    down_proj_path = spec["down_proj_path"]
    superweights = to_sw_dicts(spec["superweights"])

    if args.ablate_superweights:
        ablation_mode = "superweights"
    elif args.ablate_random:
        ablation_mode = "random"
    else:
        ablation_mode = "none"

    print("=" * 70)
    print("SUPERWEIGHT ABLATION EVALUATION")
    print("=" * 70)
    print(f"Model key:              {args.model_key}")
    print(f"Loading model:          {model_path}")
    print(f"Task:                   {args.task}")
    print(f"Limit:                  {args.limit}")
    print(f"Evaluation seed:        {args.eval_seed}")
    print(f"Ablation mode:          {ablation_mode}")
    print(f"Random ablation seed:   {args.random_seed}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has neither a pad token nor an EOS token."
            )

        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=(
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        ),
        device_map=(
            "auto"
            if torch.cuda.is_available()
            else None
        ),
        trust_remote_code=True,
    )

    model.eval()

    ablated_positions: list[dict[str, Any]] = []

    if args.ablate_superweights:
        ablated_positions = [
            dict(superweight)
            for superweight in superweights
        ]

        ablate_positions(
            model=model,
            positions=ablated_positions,
            layer_path=layer_path,
            down_proj_path=down_proj_path,
        )

    elif args.ablate_random:
        ablated_positions = sample_random_positions(
            model=model,
            superweights=superweights,
            layer_path=layer_path,
            down_proj_path=down_proj_path,
            seed=args.random_seed,
        )

        ablate_positions(
            model=model,
            positions=ablated_positions,
            layer_path=layer_path,
            down_proj_path=down_proj_path,
        )

    lm = LoadedHFLM(
        model=model,
        tokenizer=tokenizer,
    )

    task_dict = tasks.get_task_dict([args.task])

    if args.task not in task_dict:
        raise KeyError(
            f"Task {args.task!r} was not found in the task dictionary. "
            f"Available keys: {list(task_dict.keys())}"
        )

    task = task_dict[args.task]

    evaluation_split, total_document_count = shuffle_task_documents(
        task=task,
        seed=args.eval_seed,
    )

    effective_limit = min(
        args.limit,
        total_document_count,
    )

    print(f"Evaluation split:       {evaluation_split}")
    print(f"Available documents:    {total_document_count}")
    print(f"Evaluated documents:    {effective_limit}")
    print("=" * 70)

    results = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=effective_limit,
        bootstrap_iters=0,
    )

    task_results = results["results"][args.task]

    output = {
        "model_key": args.model_key,
        "model_path": model_path,
        "task": args.task,
        "evaluation_split": evaluation_split,
        "available_document_count": total_document_count,
        "evaluated_document_count": effective_limit,
        "limit": args.limit,
        "eval_seed": args.eval_seed,
        "ablation_mode": ablation_mode,
        "random_seed": (
            args.random_seed
            if args.ablate_random
            else None
        ),
        "layer_path": layer_path,
        "down_proj_path": down_proj_path,
        "superweights": superweights,
        "ablated_positions": ablated_positions,
        "results": results,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== RESULTS ===")
    print(
        json.dumps(
            task_results,
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()