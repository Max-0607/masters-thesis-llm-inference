from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.hooks import get_nested_attr


MODEL_SPECS = {
    "mistral-7b": {
        "hf_name": "mistralai/Mistral-7B-v0.1",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [(1, 2070, 7310)],
    },
    "llama-7b": {
        "hf_name": "huggyllama/llama-7b",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [(2, 3968, 7003)],
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


def to_sw_dicts(tuples):
    return [
        {
            "layer": layer,
            "row": row,
            "col": col,
        }
        for layer, row, col in tuples
    ]


def get_down_proj_weight(
    model,
    layer: int,
    layer_path: str,
    down_proj_path: str,
):
    layers = get_nested_attr(model, layer_path)
    module = get_nested_attr(layers[layer], down_proj_path)
    return module.weight


def ablate_positions(
    model,
    positions,
    layer_path: str,
    down_proj_path: str,
):
    with torch.no_grad():
        for pos in positions:
            weight = get_down_proj_weight(
                model=model,
                layer=pos["layer"],
                layer_path=layer_path,
                down_proj_path=down_proj_path,
            )

            old_value = float(
                weight[pos["row"], pos["col"]]
                .detach()
                .float()
                .cpu()
                .item()
            )

            weight[pos["row"], pos["col"]] = 0.0
            pos["old_value"] = old_value

            print(
                "Ablated "
                f"layer={pos['layer']} "
                f"row={pos['row']} "
                f"col={pos['col']} "
                f"old={old_value}"
            )


def sample_random_positions(
    model,
    superweights,
    layer_path: str,
    down_proj_path: str,
    seed: int,
):
    rng = random.Random(seed)

    sw_set = {
        (sw["layer"], sw["row"], sw["col"])
        for sw in superweights
    }

    chosen_set = set()
    random_positions = []

    for sw in superweights:
        layer = sw["layer"]

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
            key = (layer, row, col)

            if key not in sw_set and key not in chosen_set:
                chosen_set.add(key)
                break

        random_positions.append(
            {
                "layer": layer,
                "row": row,
                "col": col,
            }
        )

    return random_positions


def recursively_find_key(obj: Any, key: str):
    """
    Find the first occurrence of a key in a nested JSON structure.
    """
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]

        for value in obj.values():
            result = recursively_find_key(value, key)
            if result is not None:
                return result

    elif isinstance(obj, list):
        for value in obj:
            result = recursively_find_key(value, key)
            if result is not None:
                return result

    return None


def load_reference_ids(reference_json: str | None):
    """
    Read selected_example_ids from an existing Table 3.2 JSON file.
    """
    if reference_json is None:
        return None

    path = Path(reference_json)

    if not path.exists():
        raise FileNotFoundError(
            f"Reference JSON does not exist: {path}"
        )

    data = json.loads(path.read_text(encoding="utf-8"))

    ids = recursively_find_key(
        data,
        "selected_example_ids",
    )

    if ids is None:
        raise KeyError(
            "The reference JSON does not contain "
            "'selected_example_ids'."
        )

    try:
        return [int(example_id) for example_id in ids]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "selected_example_ids must contain dataset indices "
            "that can be converted to integers."
        ) from exc


def load_hellaswag_validation():
    """
    Load the HellaSwag validation split.

    Different versions of datasets may use different dataset names,
    therefore a fallback is included.
    """
    try:
        return load_dataset(
            "Rowan/hellaswag",
            split="validation",
        )
    except Exception:
        return load_dataset(
            "hellaswag",
            split="validation",
        )


def select_example_ids(
    dataset_size: int,
    limit: int | None,
    eval_seed: int,
    reference_ids: list[int] | None,
):
    """
    Select evaluation examples.

    Priority:
    1. IDs from --reference-json
    2. Deterministic random sample controlled by --eval-seed
    """
    if reference_ids is not None:
        invalid_ids = [
            example_id
            for example_id in reference_ids
            if example_id < 0 or example_id >= dataset_size
        ]

        if invalid_ids:
            raise ValueError(
                "Reference JSON contains invalid dataset IDs: "
                f"{invalid_ids[:10]}"
            )

        if limit is not None and len(reference_ids) != limit:
            raise ValueError(
                "Number of IDs in reference JSON does not match "
                f"--limit: ids={len(reference_ids)}, limit={limit}"
            )

        return reference_ids

    available_ids = list(range(dataset_size))

    if limit is None or limit >= dataset_size:
        return available_ids

    rng = random.Random(eval_seed)
    return rng.sample(available_ids, limit)


def build_hellaswag_context(example: dict) -> str:
    """
    Reproduce the standard HellaSwag context construction.

    The original task combines ctx_a and ctx_b and capitalizes ctx_b.
    """
    if "ctx_a" in example and "ctx_b" in example:
        ctx_a = str(example["ctx_a"]).strip()
        ctx_b = str(example["ctx_b"]).strip()

        if ctx_b:
            ctx_b = ctx_b[0].upper() + ctx_b[1:]

        return f"{ctx_a} {ctx_b}".strip()

    if "ctx" in example:
        return str(example["ctx"]).strip()

    raise KeyError(
        "HellaSwag example contains neither "
        "'ctx_a'/'ctx_b' nor 'ctx'."
    )


def get_model_input_device(model):
    """
    Determine the device on which input IDs should be placed.

    This is safer than using next(model.parameters()).device when
    device_map='auto' distributes the model across devices.
    """
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def score_continuation(
    model,
    tokenizer,
    context: str,
    continuation: str,
    max_length: int,
):
    """
    Compute the total and length-normalized conditional log-likelihood
    of a continuation given a context.
    """
    context_ids = tokenizer.encode(
        context,
        add_special_tokens=False,
    )

    continuation_text = continuation

    if not continuation_text.startswith(" "):
        continuation_text = " " + continuation_text

    continuation_ids = tokenizer.encode(
        continuation_text,
        add_special_tokens=False,
    )

    if not continuation_ids:
        return float("-inf"), float("-inf"), 0

    # A causal model needs at least one context token to predict the first
    # continuation token.
    if not context_ids:
        fallback_token = tokenizer.bos_token_id

        if fallback_token is None:
            fallback_token = tokenizer.eos_token_id

        if fallback_token is None:
            raise ValueError(
                "Tokenizer has neither BOS nor EOS token."
            )

        context_ids = [fallback_token]

    # Keep the continuation and truncate older context tokens if required.
    total_length = len(context_ids) + len(continuation_ids)

    if total_length > max_length:
        max_context_length = max_length - len(continuation_ids)

        if max_context_length < 1:
            continuation_ids = continuation_ids[-(max_length - 1):]
            max_context_length = 1

        context_ids = context_ids[-max_context_length:]

    input_ids = context_ids + continuation_ids

    input_tensor = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=get_model_input_device(model),
    )

    with torch.no_grad():
        logits = model(
            input_ids=input_tensor,
            use_cache=False,
        ).logits

    # logits[:, t] predicts input_ids[:, t + 1]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_tensor[:, 1:]

    continuation_start = len(context_ids) - 1
    continuation_logits = shift_logits[
        :,
        continuation_start:,
        :,
    ]
    continuation_labels = shift_labels[
        :,
        continuation_start:,
    ]

    log_probs = F.log_softmax(
        continuation_logits.float(),
        dim=-1,
    )

    token_log_probs = log_probs.gather(
        dim=-1,
        index=continuation_labels.unsqueeze(-1),
    ).squeeze(-1)

    total_log_likelihood = float(
        token_log_probs.sum().item()
    )

    num_tokens = int(token_log_probs.numel())

    normalized_log_likelihood = (
        total_log_likelihood / num_tokens
        if num_tokens > 0
        else float("-inf")
    )

    return (
        total_log_likelihood,
        normalized_log_likelihood,
        num_tokens,
    )


def evaluate_hellaswag(
    model,
    tokenizer,
    selected_examples,
    max_length: int,
):
    num_correct = 0
    per_example_results = []

    for example_id, example in tqdm(
        selected_examples,
        desc="Evaluating HellaSwag",
    ):
        context = build_hellaswag_context(example)
        endings = example["endings"]
        gold_label = int(example["label"])

        raw_scores = []
        normalized_scores = []
        token_counts = []

        for ending in endings:
            raw_score, normalized_score, num_tokens = score_continuation(
                model=model,
                tokenizer=tokenizer,
                context=context,
                continuation=str(ending),
                max_length=max_length,
            )

            raw_scores.append(raw_score)
            normalized_scores.append(normalized_score)
            token_counts.append(num_tokens)

        prediction = max(
            range(len(normalized_scores)),
            key=lambda index: normalized_scores[index],
        )

        correct = int(prediction == gold_label)
        num_correct += correct

        per_example_results.append(
            {
                "example_id": int(example_id),
                "gold_label": gold_label,
                "prediction": prediction,
                "correct": bool(correct),
                "raw_scores": raw_scores,
                "normalized_scores": normalized_scores,
                "continuation_token_counts": token_counts,
            }
        )

    num_examples = len(selected_examples)

    accuracy = (
        num_correct / num_examples
        if num_examples > 0
        else float("nan")
    )

    accuracy_stderr = (
        math.sqrt(
            accuracy * (1.0 - accuracy) / num_examples
        )
        if num_examples > 0
        else float("nan")
    )

    return {
        "metric": "acc_norm",
        "accuracy": accuracy,
        "acc_norm": accuracy,
        "accuracy_stderr": accuracy_stderr,
        "num_correct": num_correct,
        "num_examples": num_examples,
        "per_example_results": per_example_results,
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-key",
        choices=list(MODEL_SPECS.keys()),
        required=True,
    )

    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Model path or Hugging Face ID. If omitted, the "
            "MODEL_SPECS hf_name is used."
        ),
    )

    parser.add_argument(
        "--task",
        default="hellaswag",
        choices=["hellaswag"],
        help="Table 3.8 uses HellaSwag acc_norm.",
    )

    parser.add_argument(
        "--split",
        default="validation",
        choices=["validation"],
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
            "Seed used for selecting evaluation examples when no "
            "--reference-json is supplied."
        ),
    )

    parser.add_argument(
        "--reference-json",
        default=None,
        help=(
            "Existing Table 3.2 JSON whose selected_example_ids "
            "must be reused exactly."
        ),
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--ablate-superweights",
        action="store_true",
    )

    parser.add_argument(
        "--ablate-random",
        action="store_true",
        help=(
            "Ablate the same number of random weights as "
            "superweights."
        ),
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help=(
            "Seed for selecting random weight positions. "
            "This does not control the evaluation sample."
        ),
    )

    parser.add_argument(
        "--output-json",
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.ablate_superweights and args.ablate_random:
        raise ValueError(
            "Use only one ablation mode: "
            "--ablate-superweights or --ablate-random."
        )

    random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.eval_seed)

    spec = MODEL_SPECS[args.model_key]

    model_path = (
        args.model_path
        if args.model_path is not None
        else spec["hf_name"]
    )

    layer_path = spec["layer_path"]
    down_proj_path = spec["down_proj_path"]
    superweights = to_sw_dicts(spec["superweights"])

    print("=" * 80)
    print("TABLE 3.8 REDISTRIBUTION EVALUATION")
    print("=" * 80)
    print(f"Model key             : {args.model_key}")
    print(f"Model path            : {model_path}")
    print(f"Task                  : {args.task}")
    print(f"Split                 : {args.split}")
    print(f"Limit                 : {args.limit}")
    print(f"Evaluation seed       : {args.eval_seed}")
    print(f"Reference JSON        : {args.reference_json}")
    print(f"Max length            : {args.max_length}")
    print(f"Ablate superweights   : {args.ablate_superweights}")
    print(f"Ablate random         : {args.ablate_random}")
    print(f"Random-ablation seed  : {args.random_seed}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
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
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.eval()

    ablated_positions = []

    if args.ablate_superweights:
        ablated_positions = [
            dict(sw)
            for sw in superweights
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

    dataset = load_hellaswag_validation()

    reference_ids = load_reference_ids(
        args.reference_json
    )

    selected_example_ids = select_example_ids(
        dataset_size=len(dataset),
        limit=args.limit,
        eval_seed=args.eval_seed,
        reference_ids=reference_ids,
    )

    selected_examples = [
        (
            example_id,
            dataset[example_id],
        )
        for example_id in selected_example_ids
    ]

    print(f"Selected examples     : {len(selected_examples)}")
    print(
        "First selected IDs    : "
        f"{selected_example_ids[:10]}"
    )

    evaluation = evaluate_hellaswag(
        model=model,
        tokenizer=tokenizer,
        selected_examples=selected_examples,
        max_length=args.max_length,
    )

    ablation_mode = (
        "superweights"
        if args.ablate_superweights
        else "random"
        if args.ablate_random
        else "none"
    )

    out = {
        "benchmark": "hellaswag",
        "metric": "acc_norm",
        "model_key": args.model_key,
        "model_path": str(model_path),
        "task": args.task,
        "split": args.split,
        "limit": args.limit,
        "max_length": args.max_length,
        "normalize_by_length": True,
        "eval_seed": args.eval_seed,
        "reference_json": args.reference_json,
        "sampling_mode": (
            "reference_ids"
            if args.reference_json is not None
            else "seeded_random_sample"
        ),
        "selected_example_ids": selected_example_ids,
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
        "accuracy": evaluation["accuracy"],
        "acc_norm": evaluation["acc_norm"],
        "accuracy_stderr": evaluation["accuracy_stderr"],
        "num_correct": evaluation["num_correct"],
        "num_examples": evaluation["num_examples"],
        "per_example_results": evaluation["per_example_results"],
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            out,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"Examples         : {evaluation['num_examples']}")
    print(f"Correct          : {evaluation['num_correct']}")
    print(f"Accuracy acc_norm: {evaluation['acc_norm']:.4f}")
    print(
        "Accuracy stderr : "
        f"{evaluation['accuracy_stderr']:.4f}"
    )
    print(f"Saved to         : {output_path}")


if __name__ == "__main__":
    main()