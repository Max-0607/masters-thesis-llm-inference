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


SUPPORTED_LANGUAGES = {
    "en": "EN_US",
    "de": "DE_DE",
    "es": "ES_LA",
    "fr": "FR_FR",
}

LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}
INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


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


def first_present(row: Dict, keys: List[str]):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def normalize_answer_letter(value) -> Optional[str]:
    if value is None:
        return None

    value = str(value).strip().upper()

    if value in LETTER_TO_INDEX:
        return value

    if value in {"0", "1", "2", "3"}:
        return INDEX_TO_LETTER[int(value)]

    return None


def load_mmmlu_examples(language: str, split: str, limit: Optional[int]) -> List[Dict]:
    locale = SUPPORTED_LANGUAGES[language]
    ds = load_dataset("openai/MMMLU", locale, split=split)

    print("MMMLU columns:", ds.column_names)

    examples = []
    for row in ds:
        question = first_present(row, ["Question", "question", "input"])

        choice_a = first_present(row, ["OptionA", "option_a", "A", "a"])
        choice_b = first_present(row, ["OptionB", "option_b", "B", "b"])
        choice_c = first_present(row, ["OptionC", "option_c", "C", "c"])
        choice_d = first_present(row, ["OptionD", "option_d", "D", "d"])

        answer_raw = first_present(row, ["Answer", "answer", "label", "target"])
        subject = first_present(row, ["Subject", "subject", "category"])

        answer = normalize_answer_letter(answer_raw)

        if question is None:
            continue
        if any(choice is None for choice in [choice_a, choice_b, choice_c, choice_d]):
            continue
        if answer is None:
            continue

        examples.append(
            {
                "question": str(question).strip(),
                "choices": [
                    str(choice_a).strip(),
                    str(choice_b).strip(),
                    str(choice_c).strip(),
                    str(choice_d).strip(),
                ],
                "answer_letter": answer,
                "answer_index": LETTER_TO_INDEX[answer],
                "subject": str(subject).strip() if subject is not None else None,
            }
        )

        if limit is not None and len(examples) >= limit:
            break

    if not examples:
        raise RuntimeError(
            f"No valid MMMLU examples found for language='{language}' and split='{split}'. "
            f"Columns were: {ds.column_names}"
        )

    return examples


def build_prompt(question: str, choices: List[str], language: str) -> str:
    if language == "en":
        instruction = (
            "Answer the following multiple-choice question.\n"
            "Respond with only one letter: A, B, C, or D.\n"
            "Do not explain.\n\n"
        )
        answer_token = "Answer:"

    elif language == "de":
        instruction = (
            "Beantworte die folgende Multiple-Choice-Frage.\n"
            "Antworte nur mit einem Buchstaben: A, B, C oder D.\n"
            "Gib keine Erklärung.\n\n"
        )
        answer_token = "Antwort:"

    elif language == "es":
        instruction = (
            "Responde a la siguiente pregunta de opción múltiple.\n"
            "Responde solo con una letra: A, B, C o D.\n"
            "No des explicación.\n\n"
        )
        answer_token = "Respuesta:"

    elif language == "fr":
        instruction = (
            "Réponds à la question à choix multiple suivante.\n"
            "Réponds uniquement avec une lettre : A, B, C ou D.\n"
            "Ne donne pas d'explication.\n\n"
        )
        answer_token = "Réponse :"

    else:
        raise ValueError(f"Unsupported language: {language}")

    return (
        instruction
        + f"{question}\n\n"
        + f"A. {choices[0]}\n"
        + f"B. {choices[1]}\n"
        + f"C. {choices[2]}\n"
        + f"D. {choices[3]}\n\n"
        + answer_token
    )

def build_prompt(question: str, choices: List[str], language: str) -> str:
    if language == "de":
        return (
            "Beantworte die folgende Multiple-Choice-Frage.\n"
            "Antworte nur mit einem Buchstaben: A, B, C oder D.\n"
            "Gib keine Erklärung.\n\n"
            f"{question}\n\n"
            f"A. {choices[0]}\n"
            f"B. {choices[1]}\n"
            f"C. {choices[2]}\n"
            f"D. {choices[3]}\n\n"
            "Antwort:"
        )
    if language == "es":
        return (
            "Responde a la siguiente pregunta de opción múltiple.\n"
            "Responde solo con una letra: A, B, C o D.\n"
            "No des explicación.\n\n"
            f"{question}\n\n"
            f"A. {choices[0]}\n"
            f"B. {choices[1]}\n"
            f"C. {choices[2]}\n"
            f"D. {choices[3]}\n\n"
            "Respuesta:"
        )
    if language == "fr":
        return (
            "Réponds à la question à choix multiple suivante.\n"
            "Réponds uniquement avec une lettre : A, B, C ou D.\n"
            "Ne donne pas d'explication.\n\n"
            f"{question}\n\n"
            f"A. {choices[0]}\n"
            f"B. {choices[1]}\n"
            f"C. {choices[2]}\n"
            f"D. {choices[3]}\n\n"
            "Réponse :"
        )
    raise ValueError(f"Unsupported language: {language}")


def extract_answer_letter(text: str) -> Optional[str]:
    text = text.strip()

    match = re.search(r"\b([ABCD])\b", text.upper())
    if match:
        return match.group(1)

    for ch in text:
        up = ch.upper()
        if up in LETTER_TO_INDEX:
            return up

    return None


def generate_answer(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    text = text.splitlines()[0].strip()
    return text


def evaluate_mmmlu(model, tokenizer, examples, language: str):
    device = next(model.parameters()).device

    correct = 0
    total = 0
    predictions = []

    for idx, ex in enumerate(examples):
        prompt = build_prompt(ex["question"], ex["choices"], language)
        pred_text = generate_answer(model, tokenizer, prompt, device)
        pred_letter = extract_answer_letter(pred_text)
        gold_letter = ex["answer_letter"]
        is_correct = pred_letter == gold_letter

        if is_correct:
            correct += 1
        total += 1

        predictions.append(
            {
                "index": idx,
                "subject": ex["subject"],
                "question": ex["question"],
                "choices": ex["choices"],
                "gold_letter": gold_letter,
                "pred_text": pred_text,
                "pred_letter": pred_letter,
                "correct": is_correct,
            }
        )

    return {
        "num_examples": total,
        "num_correct": correct,
        "accuracy": correct / total if total > 0 else 0.0,
        "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", type=str, required=True)
    parser.add_argument("--mode", type=str, default="fp16", choices=["fp16", "naive", "super"])
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--language", type=str, required=True, choices=list(SUPPORTED_LANGUAGES.keys()))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output-json", type=str, required=True)

    args = parser.parse_args()

    if args.model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model-key: {args.model_key}")

    model_cfg = MODEL_CONFIGS[args.model_key]
    model_id = model_cfg["hf_name"]
    dtype = resolve_torch_dtype(args.dtype)

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"Loading MMMLU language='{args.language}', split='{args.split}', limit={args.limit}")
    examples = load_mmmlu_examples(args.language, args.split, args.limit)

    quant_hook = build_quant_hook(model, args.model_key, args.mode, args.bits)

    try:
        metrics = evaluate_mmmlu(model, tokenizer, examples, args.language)
    finally:
        if quant_hook:
            quant_hook.remove()

    result = {
        "model_key": args.model_key,
        "hf_name": model_id,
        "mode": args.mode,
        "bits": args.bits,
        "dtype": args.dtype,
        "language": args.language,
        "locale": SUPPORTED_LANGUAGES[args.language],
        "split": args.split,
        "num_examples": metrics["num_examples"],
        "num_correct": metrics["num_correct"],
        "accuracy": metrics["accuracy"],
        "predictions": metrics["predictions"],
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "model_key": result["model_key"],
                "mode": result["mode"],
                "bits": result["bits"],
                "language": result["language"],
                "locale": result["locale"],
                "num_examples": result["num_examples"],
                "num_correct": result["num_correct"],
                "accuracy": result["accuracy"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()