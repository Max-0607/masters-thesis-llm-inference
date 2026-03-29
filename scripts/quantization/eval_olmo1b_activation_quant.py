import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List

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


def load_eval_texts(dataset_name: str, split: str, limit: int) -> List[str]:
    if dataset_name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        texts = [x["text"] for x in ds if x["text"] and x["text"].strip()]
        return texts[:limit]

    if dataset_name == "c4":
        ds = load_dataset("allenai/c4", "en", split=split, streaming=False)
        texts = []
        for row in ds:
            text = row.get("text", "")
            if text and text.strip():
                texts.append(text)
            if len(texts) >= limit:
                break
        return texts

    raise ValueError(f"Unknown dataset_name: {dataset_name}")


def build_quant_hook(model, model_key: str, mode: str, bits: int):
    if mode == "fp16":
        return None

    model_cfg = MODEL_CONFIGS[model_key]
    layers = get_nested_attr(model, model_cfg["layer_path"])
    sw_layers = sorted({entry["layer"] for entry in SUPERWEIGHTS[model_key]})

    return ActivationQuantHook(
        layers=layers,
        module_path=model_cfg["down_proj_path"],
        layer_indices=sw_layers,
        n_bits=bits,
        mode=mode,
    )


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
        nll = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        num_target_tokens = labels.numel()
        total_nll += float(nll.item())
        total_tokens += int(num_target_tokens)
        num_examples += 1

    if total_tokens == 0:
        raise RuntimeError("No valid evaluation tokens found.")

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)

    return {
        "num_examples": num_examples,
        "num_tokens": total_tokens,
        "avg_nll": avg_nll,
        "perplexity": ppl,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", type=str, default="olmo-1b", choices=["olmo-1b"])
    parser.add_argument("--mode", type=str, default="fp16", choices=["fp16", "naive", "super"])
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2", "c4"])
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)

    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
    )

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

    print(f"Loading dataset: {args.dataset} [{args.split}] limit={args.limit}")
    texts = load_eval_texts(args.dataset, args.split, args.limit)

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
        **metrics,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()