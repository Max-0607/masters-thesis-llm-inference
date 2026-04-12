import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
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


def build_quant_hook(model, model_key: str, mode: str, bits: int):
    if mode == "fp16":
        return None

    model_cfg = MODEL_CONFIGS[model_key]
    layers = get_nested_attr(model, model_cfg["layer_path"])

    if mode == "naive":
        return ActivationQuantHook(
            layers=layers,
            module_path=model_cfg["down_proj_path"],
            layer_indices=list(range(len(layers))),
            n_bits=bits,
            mode=mode,
        )

    if mode == "super":
        if model_key not in SUPERWEIGHTS:
            raise ValueError(f"No superweights for {model_key}")

        sw_layers = sorted({entry["layer"] for entry in SUPERWEIGHTS[model_key]})

        return ActivationQuantHook(
            layers=layers,
            module_path=model_cfg["down_proj_path"],
            layer_indices=sw_layers,
            n_bits=bits,
            mode=mode,
        )

    raise ValueError(f"Unsupported mode: {mode}")


# -----------------------------
# BoolQ
# -----------------------------

def build_prompt(example: Dict) -> str:
    passage = example["passage"].strip()
    question = example["question"].strip()

    return f"{passage}\nQuestion: {question}\nAnswer:"


def score_answer(model, tokenizer, prompt, answer, device):
    full = prompt + " " + answer

    input_ids = tokenizer(full, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]

    labels = input_ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)

    loss = outputs.loss
    num_tokens = (labels != -100).sum().item()

    return -loss.item() * num_tokens


def load_boolq_examples(split: str, limit: Optional[int]) -> List[Dict]:
    ds = load_dataset("super_glue", "boolq", split=split)

    examples = []
    for i, row in enumerate(ds):
        examples.append({
            "id": i,
            "passage": row["passage"],
            "question": row["question"],
            "label": int(row["label"]),
        })

        if limit is not None and len(examples) >= limit:
            break

    return examples


def evaluate_boolq(model, tokenizer, examples):
    device = next(model.parameters()).device

    correct = 0

    for ex in examples:
        prompt = build_prompt(ex)

        score_yes = score_answer(model, tokenizer, prompt, "yes", device)
        score_no = score_answer(model, tokenizer, prompt, "no", device)

        pred = 1 if score_yes > score_no else 0
        correct += int(pred == ex["label"])

    total = len(examples)

    return {
        "num_examples": total,
        "num_correct": correct,
        "accuracy": correct / total if total > 0 else 0,
    }


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True, choices=sorted(MODEL_CONFIGS.keys()))
    parser.add_argument("--mode", default="fp16", choices=["fp16", "naive", "super"])
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--dtype", default="float16")

    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=200)

    parser.add_argument("--output-json", required=True)

    args = parser.parse_args()

    model_cfg = MODEL_CONFIGS[args.model_key]
    model_id = model_cfg["hf_name"]
    torch_dtype = resolve_torch_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.eval()

    examples = load_boolq_examples(args.split, args.limit)

    quant_hook = build_quant_hook(model, args.model_key, args.mode, args.bits)

    try:
        metrics = evaluate_boolq(model, tokenizer, examples)
    finally:
        if quant_hook:
            quant_hook.remove()

    result = {
        "benchmark": "boolq",
        "model_key": args.model_key,
        "mode": args.mode,
        "bits": args.bits,
        **metrics,
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()