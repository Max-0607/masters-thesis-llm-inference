import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.models import MODEL_CONFIGS
from configs.superweights import SUPERWEIGHTS
from src.hooks import get_nested_attr
from src.quantization import ActivationQuantHook


def resolve_torch_dtype(name: str):
    name = name.lower()

    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32

    raise ValueError(f"Unsupported dtype: {name}")


def load_eval_pool(
    dataset_name: str,
    split: str,
    limit: int,
) -> List[Tuple[int, str]]:
    """
    Load exactly the same evaluation pool as the original Table 5.1
    experiment: the first `limit` non-empty texts.

    Each item is returned as:
        (original_dataset_index, text)

    The original dataset index is stored so that bootstrap samples can be
    reproduced exactly across FP16, Naive, and Superweight runs.
    """
    if limit <= 0:
        raise ValueError("--limit must be greater than zero.")

    pool: List[Tuple[int, str]] = []

    if dataset_name == "wikitext2":
        ds = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split=split,
        )

        for dataset_index, row in enumerate(ds):
            text = row.get("text", "")

            if text and text.strip():
                pool.append((dataset_index, text))

            if len(pool) >= limit:
                break

    elif dataset_name == "c4":
        ds = load_dataset(
            "allenai/c4",
            "en",
            split=split,
            streaming=True,
        )

        for dataset_index, row in enumerate(ds):
            text = row.get("text", "")

            if text and text.strip():
                pool.append((dataset_index, text))

            if len(pool) >= limit:
                break

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    if len(pool) < limit:
        raise RuntimeError(
            f"Only {len(pool)} non-empty texts were found, "
            f"but --limit={limit} was requested."
        )

    return pool


def load_reference_result(reference_json: str) -> dict:
    reference_path = Path(reference_json)

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference JSON does not exist: {reference_path}"
        )

    with open(reference_path, "r", encoding="utf-8") as f:
        reference = json.load(f)

    if "selected_example_ids" not in reference:
        raise ValueError(
            f"Reference JSON '{reference_path}' does not contain "
            "'selected_example_ids'."
        )

    selected_ids = reference["selected_example_ids"]

    if not isinstance(selected_ids, list):
        raise ValueError(
            "'selected_example_ids' in the reference JSON must be a list."
        )

    return reference


def validate_reference_result(
    reference: dict,
    dataset_name: str,
    split: str,
    limit: int,
    eval_seed: Optional[int],
):
    """
    Ensure that the reference JSON belongs to the same evaluation setting.

    Duplicate selected IDs are intentionally allowed because bootstrap
    sampling draws with replacement.
    """
    checks = {
        "dataset": dataset_name,
        "split": split,
        "limit": limit,
    }

    for key, expected_value in checks.items():
        reference_value = reference.get(key)

        if reference_value != expected_value:
            raise ValueError(
                f"Reference JSON mismatch for '{key}': "
                f"expected {expected_value!r}, "
                f"found {reference_value!r}."
            )

    if eval_seed is not None:
        reference_seed = reference.get("eval_seed")

        if reference_seed != eval_seed:
            raise ValueError(
                "Reference JSON seed mismatch: "
                f"--eval-seed={eval_seed}, "
                f"but reference contains eval_seed={reference_seed}."
            )


def select_eval_texts(
    pool: List[Tuple[int, str]],
    eval_seed: Optional[int],
    reference_json: Optional[str],
) -> Tuple[List[str], List[int]]:
    """
    Select evaluation texts.

    Without --eval-seed and --reference-json:
        Use the original evaluation pool unchanged.

    With --eval-seed:
        Draw a bootstrap sample of the same size as the original pool,
        with replacement.

    With --reference-json:
        Reuse selected_example_ids exactly, including order and duplicates.
    """
    id_to_text = {
        dataset_index: text
        for dataset_index, text in pool
    }

    pool_ids = [
        dataset_index
        for dataset_index, _ in pool
    ]

    if reference_json is not None:
        reference = load_reference_result(reference_json)

        selected_ids = reference["selected_example_ids"]

        missing_ids = sorted({
            example_id
            for example_id in selected_ids
            if example_id not in id_to_text
        })

        if missing_ids:
            raise ValueError(
                "The reference JSON contains IDs that are not part of the "
                f"original evaluation pool: {missing_ids}"
            )

    elif eval_seed is not None:
        rng = random.Random(eval_seed)

        selected_ids = rng.choices(
            pool_ids,
            k=len(pool_ids),
        )

    else:
        # Backward-compatible behavior for reproducing the original table.
        selected_ids = pool_ids.copy()

    selected_texts = [
        id_to_text[example_id]
        for example_id in selected_ids
    ]

    return selected_texts, selected_ids


def build_quant_hook(
    model,
    model_key: str,
    mode: str,
    bits: int,
):
    """
    Build the existing activation-quantization hook.

    The quantization logic itself is intentionally unchanged.
    """
    if mode == "fp16":
        return None

    model_cfg = MODEL_CONFIGS[model_key]
    layers = get_nested_attr(
        model,
        model_cfg["layer_path"],
    )

    if mode == "naive":
        all_layers = list(range(len(layers)))

        return ActivationQuantHook(
            layers=layers,
            module_path=model_cfg["down_proj_path"],
            layer_indices=all_layers,
            n_bits=bits,
            mode=mode,
        )

    if mode == "super":
        if model_key not in SUPERWEIGHTS:
            raise ValueError(
                f"No superweights registered for model_key='{model_key}'"
            )

        sw_layers = sorted({
            entry["layer"]
            for entry in SUPERWEIGHTS[model_key]
        })

        return ActivationQuantHook(
            layers=layers,
            module_path=model_cfg["down_proj_path"],
            layer_indices=sw_layers,
            n_bits=bits,
            mode=mode,
        )

    raise ValueError(f"Unsupported mode: {mode}")


def evaluate_perplexity(
    model,
    tokenizer,
    texts: Iterable[str],
    max_length: int = 512,
) -> dict:
    device = next(model.parameters()).device

    total_nll = 0.0
    total_tokens = 0
    num_examples = 0

    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        if input_ids.size(1) < 2:
            continue

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

        logits = outputs.logits[:, :-1, :].float()
        labels = input_ids[:, 1:]

        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")

        nll = loss_fct(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )

        num_target_tokens = labels.numel()

        total_nll += float(nll.item())
        total_tokens += int(num_target_tokens)
        num_examples += 1

    if total_tokens == 0:
        raise RuntimeError("No valid evaluation tokens found.")

    avg_nll = total_nll / total_tokens
    perplexity = math.exp(avg_nll)

    return {
        "num_examples": num_examples,
        "num_tokens": total_tokens,
        "avg_nll": avg_nll,
        "perplexity": perplexity,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate FP16 or activation-quantized language models "
            "on the original Table 5.1 evaluation pool."
        )
    )

    parser.add_argument(
        "--model-key",
        type=str,
        required=True,
        choices=sorted(MODEL_CONFIGS.keys()),
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="fp16",
        choices=["fp16", "naive", "super"],
    )

    parser.add_argument(
        "--bits",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4"],
    )

    parser.add_argument(
        "--split",
        type=str,
        default="validation",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=128,
        help=(
            "Size of the original evaluation pool. The first `limit` "
            "non-empty texts are loaded before bootstrap sampling."
        ),
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help=(
            "Seed for bootstrap sampling from the original evaluation pool. "
            "If omitted, the original texts are evaluated without sampling."
        ),
    )

    parser.add_argument(
        "--reference-json",
        type=str,
        default=None,
        help=(
            "JSON file whose selected_example_ids should be reused exactly. "
            "Duplicates and order are preserved."
        ),
    )

    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    if args.reference_json is not None:
        reference = load_reference_result(args.reference_json)

        validate_reference_result(
            reference=reference,
            dataset_name=args.dataset,
            split=args.split,
            limit=args.limit,
            eval_seed=args.eval_seed,
        )

    model_cfg = MODEL_CONFIGS[args.model_key]
    model_id = model_cfg["hf_name"]
    torch_dtype = resolve_torch_dtype(args.dtype)

    print(f"Loading tokenizer: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_id} ({args.dtype})")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    model.eval()

    print(
        f"Loading original evaluation pool: "
        f"{args.dataset} [{args.split}] limit={args.limit}"
    )

    eval_pool = load_eval_pool(
        dataset_name=args.dataset,
        split=args.split,
        limit=args.limit,
    )

    texts, selected_example_ids = select_eval_texts(
        pool=eval_pool,
        eval_seed=args.eval_seed,
        reference_json=args.reference_json,
    )

    if args.reference_json is not None:
        sampling_description = (
            f"reference JSON: {args.reference_json}"
        )
    elif args.eval_seed is not None:
        sampling_description = (
            f"bootstrap with replacement, seed={args.eval_seed}"
        )
    else:
        sampling_description = "original evaluation order"

    print(f"Evaluation selection: {sampling_description}")
    print(f"Selected texts: {len(selected_example_ids)}")
    print(
        "Unique selected texts: "
        f"{len(set(selected_example_ids))}"
    )

    quant_hook = build_quant_hook(
        model=model,
        model_key=args.model_key,
        mode=args.mode,
        bits=args.bits,
    )

    try:
        metrics = evaluate_perplexity(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            max_length=args.max_length,
        )

    finally:
        if quant_hook is not None:
            quant_hook.remove()

    result = {
        "model_key": args.model_key,
        "model_id": model_id,
        "mode": args.mode,
        "bits": args.bits,
        "dtype": args.dtype,
        "dataset": args.dataset,
        "split": args.split,
        "limit": args.limit,
        "max_length": args.max_length,
        "eval_seed": args.eval_seed,
        "sampling_method": (
            "bootstrap_with_replacement"
            if args.eval_seed is not None
            else "original_order"
        ),
        "reference_json": args.reference_json,
        "pool_size": len(eval_pool),
        "bootstrap_sample_size": len(selected_example_ids),
        "num_unique_selected_examples": len(
            set(selected_example_ids)
        ),
        "selected_example_ids": selected_example_ids,
        **metrics,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()