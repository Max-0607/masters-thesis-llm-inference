import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

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


def build_quant_hook(model, model_key: str, mode: str, bits: int):
    if mode == "fp16":
        return None

    model_cfg = MODEL_CONFIGS[model_key]
    layers = get_nested_attr(model, model_cfg["layer_path"])

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
            raise ValueError(f"No superweights registered for model_key='{model_key}'")

        sw_layers = sorted({entry["layer"] for entry in SUPERWEIGHTS[model_key]})

        return ActivationQuantHook(
            layers=layers,
            module_path=model_cfg["down_proj_path"],
            layer_indices=sw_layers,
            n_bits=bits,
            mode=mode,
        )

    raise ValueError(f"Unsupported mode: {mode}")


def load_mgsm_examples(language: str, split: str, limit: Optional[int]) -> List[Dict]:
    ds = load_dataset("juletxara/mgsm", language, split=split)

    examples = []
    for row in ds:
        question = row.get("question")

        # 🔥 WICHTIG: MGSM FIX
        if "answer_number" in row and row["answer_number"] is not None:
            answer = str(row["answer_number"])
        else:
            answer = row.get("answer")

        if question is None or answer is None:
            continue

        examples.append(
            {
                "question": question.strip(),
                "answer": answer.strip(),
            }
        )

        if limit is not None and len(examples) >= limit:
            break

    if not examples:
        raise RuntimeError("No valid MGSM examples found.")

    return examples


def build_prompt(question: str) -> str:
    return (
        "Solve the following math word problem step by step.\n"
        "At the end, give the final answer as a single number.\n\n"
        f"Question: {question}\nAnswer:"
    )


def extract_final_number(text: str) -> Optional[str]:
    matches = re.findall(r"-?\d+(?:[.,]\d+)?", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def normalize_answer(text: str) -> Optional[str]:
    value = extract_final_number(text)
    if value is None:
        return None

    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return str(number)
    except Exception:
        return value


def generate_answer(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def evaluate_mgsm(model, tokenizer, examples):
    device = next(model.parameters()).device

    correct = 0
    total = 0

    for ex in examples:
        prompt = build_prompt(ex["question"])
        pred_text = generate_answer(model, tokenizer, prompt, device)

        pred = normalize_answer(pred_text)
        gold = normalize_answer(ex["answer"])

        if pred is not None and gold is not None and pred == gold:
            correct += 1

        total += 1

    return {
        "num_examples": total,
        "num_correct": correct,
        "accuracy": correct / total if total > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", type=str, required=True)
    parser.add_argument("--mode", type=str, default="fp16", choices=["fp16", "naive", "super"])
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="float16")

    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output-json", type=str, required=True)

    args = parser.parse_args()

    model_cfg = MODEL_CONFIGS[args.model_key]
    model_id = model_cfg["hf_name"]
    dtype = resolve_torch_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    examples = load_mgsm_examples("en", "test", args.limit)

    quant_hook = build_quant_hook(model, args.model_key, args.mode, args.bits)

    try:
        metrics = evaluate_mgsm(model, tokenizer, examples)
    finally:
        if quant_hook:
            quant_hook.remove()

    result = {
        "model_key": args.model_key,
        "mode": args.mode,
        **metrics,
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)

    print(result)


if __name__ == "__main__":
    main()