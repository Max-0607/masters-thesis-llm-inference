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


def build_prompt(example: Dict) -> str:
    goal = example["goal"].strip()
    return f"Question: {goal}\nAnswer:"


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


def load_piqa_examples(split: str, limit: Optional[int]) -> List[Dict]:
    candidate_loaders = [
        lambda: load_dataset("ybisk/piqa", split=split),
        lambda: load_dataset("lighteval/piqa", split=split),
        lambda: load_dataset("regisss/piqa", split=split),
    ]

    last_error = None
    ds = None
    for loader in candidate_loaders:
        try:
            ds = loader()
            break
        except Exception as e:
            last_error = e

    if ds is None:
        raise RuntimeError(f"Could not load PIQA dataset. Last error: {last_error}")

    examples = []
    for i, row in enumerate(ds):
        label = row.get("label", None)
        if label not in [0, 1]:
            continue

        examples.append(
            {
                "id": i,
                "goal": row["goal"],
                "sol1": row["sol1"],
                "sol2": row["sol2"],
                "label": int(label),
            }
        )

        if limit is not None and len(examples) >= limit:
            break

    if not examples:
        raise RuntimeError("No valid PIQA examples found.")

    return examples

    examples = []
    for i, row in enumerate(ds):
        label = row.get("label", None)
        if label not in [0, 1]:
            continue

        examples.append(
            {
                "id": i,
                "goal": row["goal"],
                "sol1": row["sol1"],
                "sol2": row["sol2"],
                "label": int(label),
            }
        )

        if limit is not None and len(examples) >= limit:
            break

    if not examples:
        raise RuntimeError("No valid PIQA examples found.")

    return examples


def evaluate_piqa(
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

        score1 = score_continuation(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            continuation=ex["sol1"],
            device=device,
            max_length=max_length,
        )
        score2 = score_continuation(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            continuation=ex["sol2"],
            device=device,
            max_length=max_length,
        )

        pred = 0 if score1 >= score2 else 1
        gold = ex["label"]
        correct = int(pred == gold)

        num_correct += correct
        scored_examples += 1

    if scored_examples == 0:
        raise RuntimeError("No PIQA examples were scored.")

    return {
        "num_examples": scored_examples,
        "accuracy": num_correct / scored_examples,
        "num_correct": num_correct,
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

    print(f"Loading PIQA split={args.split} limit={args.limit}")
    examples = load_piqa_examples(
        split=args.split,
        limit=args.limit,
    )

    quant_hook = build_quant_hook(
        model=model,
        model_key=args.model_key,
        mode=args.mode,
        bits=args.bits,
    )

    try:
        metrics = evaluate_piqa(
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
        "benchmark": "piqa",
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