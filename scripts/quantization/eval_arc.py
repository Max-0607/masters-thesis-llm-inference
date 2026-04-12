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


def build_prompt(example: Dict) -> str:
    question = example["question"].strip()
    return f"Question: {question}\nAnswer:"


def score_continuation(
    model,
    tokenizer,
    prompt: str,
    continuation: str,
    device: torch.device,
    max_length: int = 256,
) -> float:
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    full_text = prompt + " " + continuation.strip()
    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    prompt_len = prompt_ids.shape[1]
    full_len = full_ids.shape[1]

    if full_len <= prompt_len:
        return float("-inf")

    input_ids = full_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]

    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    cont_start = max(prompt_len - 1, 0)
    cont_log_probs = token_log_probs[:, cont_start:]

    if cont_log_probs.numel() == 0:
        return float("-inf")

    return float(cont_log_probs.sum().item())


def normalize_arc_example(row: Dict) -> Optional[Dict]:
    choices = row.get("choices", {})
    texts = choices.get("text", [])
    labels = choices.get("label", [])
    answer_key = row.get("answerKey", None)

    if not texts or not labels or answer_key is None:
        return None

    if len(texts) != len(labels):
        return None

    answer_idx = None
    for i, lbl in enumerate(labels):
        if str(lbl).strip() == str(answer_key).strip():
            answer_idx = i
            break

    if answer_idx is None:
        return None

    return {
        "id": row.get("id", None),
        "question": row["question"],
        "choices": [str(t).strip() for t in texts],
        "label": int(answer_idx),
        "answer_key": str(answer_key),
        "choice_labels": [str(x).strip() for x in labels],
    }


def load_arc_examples(split: str, limit: Optional[int], subset: str) -> List[Dict]:
    ds = load_dataset("allenai/ai2_arc", subset, split=split)

    examples = []
    for row in ds:
        ex = normalize_arc_example(row)
        if ex is None:
            continue

        examples.append(ex)

        if limit is not None and len(examples) >= limit:
            break

    if not examples:
        raise RuntimeError(f"No valid ARC examples found for subset={subset}.")

    return examples


def evaluate_arc(
    model,
    tokenizer,
    examples: List[Dict],
    max_length: int,
) -> Dict:
    device = next(model.parameters()).device

    num_correct = 0
    scored_examples = 0

    for ex in examples:
        prompt = build_prompt(ex)

        scores = []
        for choice in ex["choices"]:
            score = score_continuation(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                continuation=choice,
                device=device,
                max_length=max_length,
            )
            scores.append(score)

        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        gold = ex["label"]
        correct = int(pred == gold)

        num_correct += correct
        scored_examples += 1

    if scored_examples == 0:
        raise RuntimeError("No ARC examples were scored.")

    return {
        "num_examples": scored_examples,
        "num_correct": num_correct,
        "accuracy": num_correct / scored_examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-key",
        type=str,
        required=True,
        choices=sorted(MODEL_CONFIGS.keys()),
    )
    parser.add_argument("--mode", type=str, default="fp16", choices=["fp16", "naive", "super"])
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--subset", type=str, default="ARC-Challenge", choices=["ARC-Challenge", "ARC-Easy"])
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=256)

    parser.add_argument("--output-json", type=str, required=True)

    args = parser.parse_args()

    model_cfg = MODEL_CONFIGS[args.model_key]
    model_id = model_cfg["hf_name"]
    torch_dtype = resolve_torch_dtype(args.dtype)

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

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

    print(f"Loading ARC subset={args.subset} split={args.split} limit={args.limit}")
    examples = load_arc_examples(
        split=args.split,
        limit=args.limit,
        subset=args.subset,
    )

    quant_hook = build_quant_hook(
        model=model,
        model_key=args.model_key,
        mode=args.mode,
        bits=args.bits,
    )

    try:
        metrics = evaluate_arc(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
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
        "benchmark": "arc",
        "subset": args.subset,
        "split": args.split,
        "limit": args.limit,
        "max_length": args.max_length,
        **metrics,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()