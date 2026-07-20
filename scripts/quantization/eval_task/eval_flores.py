import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.models import MODEL_CONFIGS
from configs.superweights import SUPERWEIGHTS
from src.hooks import get_nested_attr
from src.quantization import ActivationQuantHook


LANG_MAP = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
}


def resolve_torch_dtype(name: str):
    name = name.lower()

    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32

    raise ValueError(f"Unsupported dtype: {name}")


def build_quant_hook(
    model,
    model_key: str,
    mode: str,
    bits: int,
):
    """
    Preserve the existing ActivationQuantHook logic.
    """
    if mode == "fp16":
        return None

    model_cfg = MODEL_CONFIGS[model_key]
    layers = get_nested_attr(
        model,
        model_cfg["layer_path"],
    )

    if mode == "naive":
        return ActivationQuantHook(
            layers=layers,
            module_path=model_cfg["down_proj_path"],
            layer_indices=list(range(len(layers))),
            n_bits=bits,
            mode="naive",
        )

    if mode == "super":
        if model_key not in SUPERWEIGHTS:
            raise ValueError(
                f"No superweights registered for "
                f"model_key='{model_key}'"
            )

        sw_layers = sorted({
            int(entry["layer"])
            for entry in SUPERWEIGHTS[model_key]
        })

        return ActivationQuantHook(
            layers=layers,
            module_path=model_cfg["down_proj_path"],
            layer_indices=sw_layers,
            n_bits=bits,
            mode="super",
        )

    raise ValueError(f"Unsupported mode: {mode}")


def load_flores_pool(
    src: str,
    tgt: str,
    limit: int,
) -> List[Dict]:
    """
    Load exactly the original evaluation pool: the first `limit`
    valid FLORES devtest examples.

    Original dataset indices are stored for reproducible bootstrap
    sampling across FP16, Naive, and Super.
    """
    if limit <= 0:
        raise ValueError("--limit must be greater than zero.")

    ds = load_dataset(
        "yash9439/flores200",
        split="devtest",
    )

    print("FLORES columns loaded")

    src_col = LANG_MAP[src]
    tgt_col = LANG_MAP[tgt]

    pool = []

    for dataset_index, row in enumerate(ds):
        src_text = row.get(src_col)
        tgt_text = row.get(tgt_col)

        if src_text is None or tgt_text is None:
            continue

        src_text = str(src_text).strip()
        tgt_text = str(tgt_text).strip()

        if not src_text or not tgt_text:
            continue

        pool.append(
            {
                "id": int(dataset_index),
                "src": src_text,
                "tgt": tgt_text,
            }
        )

        if len(pool) >= limit:
            break

    if len(pool) < limit:
        raise RuntimeError(
            f"Only {len(pool)} valid FLORES examples were found "
            f"for {src}->{tgt}, but --limit={limit} was requested."
        )

    return pool


def load_reference_json(path: str) -> dict:
    reference_path = Path(path)

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference JSON does not exist: {reference_path}"
        )

    with reference_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        reference = json.load(f)

    if "selected_example_ids" not in reference:
        raise ValueError(
            "Reference JSON does not contain "
            "'selected_example_ids'."
        )

    if not isinstance(
        reference["selected_example_ids"],
        list,
    ):
        raise ValueError(
            "'selected_example_ids' must be a list."
        )

    return reference


def validate_reference(
    reference: dict,
    model_key: str,
    src: str,
    tgt: str,
    limit: int,
    eval_seed: Optional[int],
):
    expected = {
        "model_key": model_key,
        "src_lang": src,
        "tgt_lang": tgt,
        "limit": limit,
    }

    for key, expected_value in expected.items():
        actual_value = reference.get(key)

        if actual_value != expected_value:
            raise ValueError(
                f"Reference mismatch for '{key}': "
                f"expected {expected_value!r}, "
                f"found {actual_value!r}."
            )

    if eval_seed is not None:
        reference_seed = reference.get("eval_seed")

        if reference_seed != eval_seed:
            raise ValueError(
                "Reference seed mismatch: "
                f"--eval-seed={eval_seed}, "
                f"reference eval_seed={reference_seed}."
            )


def select_examples(
    pool: List[Dict],
    eval_seed: Optional[int],
    reference_json: Optional[str],
) -> Tuple[List[Dict], List[int]]:
    """
    Without a seed:
        Use the original pool unchanged.

    With --eval-seed:
        Draw a bootstrap sample of the same size with replacement.

    With --reference-json:
        Reuse IDs exactly, including duplicates and order.
    """
    id_to_example = {
        int(example["id"]): example
        for example in pool
    }

    pool_ids = [
        int(example["id"])
        for example in pool
    ]

    if reference_json is not None:
        reference = load_reference_json(
            reference_json
        )

        selected_ids = [
            int(example_id)
            for example_id
            in reference["selected_example_ids"]
        ]

    elif eval_seed is not None:
        rng = random.Random(eval_seed)

        selected_ids = rng.choices(
            pool_ids,
            k=len(pool_ids),
        )

    else:
        selected_ids = pool_ids.copy()

    missing_ids = sorted({
        example_id
        for example_id in selected_ids
        if example_id not in id_to_example
    })

    if missing_ids:
        raise ValueError(
            "Selected IDs are not part of the original "
            f"FLORES pool: {missing_ids}"
        )

    selected_examples = [
        id_to_example[example_id].copy()
        for example_id in selected_ids
    ]

    return selected_examples, selected_ids


def build_prompt(
    src_text: str,
    src: str,
    tgt: str,
) -> str:
    """
    Preserve the prompt used in the original Table 5.3 experiment.
    """
    return (
        f"Translate the following sentence from {src} to {tgt}.\n"
        "Return only the translation.\n"
        "Do not explain.\n"
        "Do not add comments.\n\n"
        f"Sentence: {src_text}\n"
        "Translation:"
    )


def evaluate(
    model,
    tokenizer,
    examples: List[Dict],
    src: str,
    tgt: str,
) -> dict:
    """
    Preserve the original per-example mean-loss PPL calculation.

    This intentionally computes:
        exp(mean(per-example loss))

    Changing to a token-weighted corpus PPL would alter the evaluation
    definition and make the new results incomparable with Table 5.3.
    """
    device = next(model.parameters()).device

    predictions = []
    losses = []

    total_examples = len(examples)

    for example_number, example in enumerate(
        examples,
        start=1,
    ):
        prompt = build_prompt(
            example["src"],
            src,
            tgt,
        )

        gold = example["tgt"]
        full_text = prompt + " " + gold

        inputs = tokenizer(
            full_text,
            return_tensors="pt",
        ).to(device)

        prompt_inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(device)

        labels = inputs["input_ids"].clone()
        prompt_len = prompt_inputs["input_ids"].shape[1]

        labels[:, :prompt_len] = -100

        with torch.no_grad():
            outputs = model(
                **inputs,
                labels=labels,
                use_cache=False,
            )

        loss = float(
            outputs.loss.detach().float().item()
        )

        if not math.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss for example "
                f"id={example['id']}: {loss}"
            )

        losses.append(loss)

        predictions.append(
            {
                "id": int(example["id"]),
                "src": example["src"],
                "gold": gold,
                "loss": loss,
            }
        )

        if (
            example_number % 10 == 0
            or example_number == total_examples
        ):
            running_loss = (
                sum(losses) / len(losses)
            )

            running_ppl = math.exp(
                running_loss
            )

            print(
                f"Progress: {example_number}/{total_examples} | "
                f"running_loss={running_loss:.4f} | "
                f"running_ppl={running_ppl:.4f}",
                flush=True,
            )

    if not losses:
        raise RuntimeError(
            "No FLORES examples were evaluated."
        )

    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(mean_loss)

    return {
        "chrf": None,
        "loss": mean_loss,
        "ppl": ppl,
        "num_examples": len(examples),
        "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate FP16 or activation-quantized models "
            "on reproducibly bootstrapped FLORES examples."
        )
    )

    parser.add_argument(
        "--model-key",
        required=True,
        choices=sorted(MODEL_CONFIGS.keys()),
    )

    parser.add_argument(
        "--mode",
        default="fp16",
        choices=["fp16", "naive", "super"],
    )

    parser.add_argument(
        "--bits",
        type=int,
        default=4,
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
        "--src-lang",
        required=True,
        choices=list(LANG_MAP.keys()),
    )

    parser.add_argument(
        "--tgt-lang",
        required=True,
        choices=list(LANG_MAP.keys()),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help=(
            "Size of the original FLORES evaluation pool "
            "before bootstrap sampling."
        ),
    )

    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help=(
            "Seed for bootstrap sampling with replacement "
            "from the original evaluation pool."
        ),
    )

    parser.add_argument(
        "--reference-json",
        type=str,
        default=None,
        help=(
            "Reuse selected_example_ids from an FP16 JSON. "
            "Duplicates and order are preserved."
        ),
    )

    parser.add_argument(
        "--output-json",
        required=True,
    )

    args = parser.parse_args()

    if args.bits <= 0:
        raise ValueError(
            "--bits must be greater than zero."
        )

    if args.src_lang == args.tgt_lang:
        raise ValueError(
            "Source and target language must differ."
        )

    if args.reference_json is not None:
        reference = load_reference_json(
            args.reference_json
        )

        validate_reference(
            reference=reference,
            model_key=args.model_key,
            src=args.src_lang,
            tgt=args.tgt_lang,
            limit=args.limit,
            eval_seed=args.eval_seed,
        )

    model_cfg = MODEL_CONFIGS[args.model_key]
    model_id = model_cfg["hf_name"]
    dtype = resolve_torch_dtype(args.dtype)

    print("=" * 78)
    print("FLORES ACTIVATION QUANTIZATION EVALUATION")
    print("=" * 78)
    print(f"Model key:       {args.model_key}")
    print(f"Model ID:        {model_id}")
    print(f"Mode:            {args.mode}")
    print(f"Activation bits: {args.bits}")
    print(f"Source language: {args.src_lang}")
    print(f"Target language: {args.tgt_lang}")
    print(f"Original pool:   {args.limit}")
    print(f"Evaluation seed: {args.eval_seed}")
    print(f"Reference JSON:  {args.reference_json}")
    print("=" * 78)

    print(f"Loading tokenizer: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
    )

    if (
        tokenizer.pad_token is None
        and tokenizer.eos_token is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_id} ({args.dtype})")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    )

    model.eval()

    print(
        f"Loading original FLORES pool: "
        f"{args.src_lang}->{args.tgt_lang}, "
        f"limit={args.limit}"
    )

    pool = load_flores_pool(
        src=args.src_lang,
        tgt=args.tgt_lang,
        limit=args.limit,
    )

    examples, selected_example_ids = (
        select_examples(
            pool=pool,
            eval_seed=args.eval_seed,
            reference_json=args.reference_json,
        )
    )

    print(
        f"Selected examples: "
        f"{len(selected_example_ids)}"
    )

    print(
        f"Unique selected examples: "
        f"{len(set(selected_example_ids))}"
    )

    hook = build_quant_hook(
        model=model,
        model_key=args.model_key,
        mode=args.mode,
        bits=args.bits,
    )

    try:
        metrics = evaluate(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            src=args.src_lang,
            tgt=args.tgt_lang,
        )

    finally:
        if hook is not None:
            hook.remove()

    result = {
        "benchmark": "flores",
        "model_key": args.model_key,
        "model_id": model_id,
        "mode": args.mode,
        "bits": args.bits,
        "dtype": args.dtype,
        "src_lang": args.src_lang,
        "tgt_lang": args.tgt_lang,
        "split": "devtest",
        "limit": args.limit,
        "eval_seed": args.eval_seed,
        "sampling_method": (
            "bootstrap_with_replacement"
            if args.eval_seed is not None
            else "original_order"
        ),
        "reference_json": args.reference_json,
        "pool_size": len(pool),
        "bootstrap_sample_size": len(
            selected_example_ids
        ),
        "num_unique_selected_examples": len(
            set(selected_example_ids)
        ),
        "selected_example_ids": (
            selected_example_ids
        ),
        "num_examples": metrics["num_examples"],
        "chrf": metrics["chrf"],
        "loss": metrics["loss"],
        "ppl": metrics["ppl"],
        "predictions": metrics["predictions"],
    }

    output_path = Path(args.output_json)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    summary = {
        key: result[key]
        for key in [
            "model_key",
            "model_id",
            "mode",
            "bits",
            "src_lang",
            "tgt_lang",
            "limit",
            "eval_seed",
            "sampling_method",
            "num_examples",
            "num_unique_selected_examples",
            "loss",
            "ppl",
        ]
    }

    print()
    print("=" * 78)
    print("FINAL FLORES RESULT")
    print("=" * 78)
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Saved result to: {output_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()