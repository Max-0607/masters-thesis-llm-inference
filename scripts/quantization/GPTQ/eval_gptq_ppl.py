import json
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from quantization.gptq.gptq_repo.eval_gptq_olmo1b_ppl import (
    get_olmo,
    get_wikitext2_olmo,
    olmo_sequential,
    DEV,
)


def load_texts(dataset, limit):
    if dataset == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [x["text"] for x in ds if x["text"].strip()]
        return texts[:limit]

    if dataset == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []

        for x in ds:
            text = x.get("text", "")
            if text.strip():
                texts.append(text)

            if len(texts) >= limit:
                break

        return texts

    raise ValueError(f"Unknown dataset: {dataset}")


@torch.no_grad()
def compute_ppl(model, tokenizer, texts, max_length=2048, device=DEV):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, text in enumerate(texts):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        input_ids = enc.input_ids.to(device)

        if input_ids.shape[1] < 2:
            continue

        outputs = model(
            input_ids=input_ids,
            labels=input_ids,
            use_cache=False,
            return_dict=True,
        )

        num_tokens = input_ids.shape[1] - 1
        loss = outputs.loss.float().item()

        total_loss += loss * num_tokens
        total_tokens += num_tokens

        if (i + 1) % 25 == 0:
            current_ppl = math.exp(total_loss / total_tokens)
            print(f"{i+1}/{len(texts)} | current_ppl={current_ppl:.4f}")

    if total_tokens == 0:
        raise RuntimeError("No valid tokens evaluated.")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    return {
        "loss": avg_loss,
        "perplexity": ppl,
        "num_tokens": total_tokens,
        "num_texts": len(texts),
    }


def run(dataset, output_json, limit=512, max_length=2048):
    model_id = "allenai/OLMo-1B-0724-hf"

    args = {
        "nsamples": 32,
        "wbits": 4,
        "groupsize": 128,
        "percdamp": 0.01,
        "act_order": True,
    }

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_id}")
    model = get_olmo(model_id)
    model.eval()

    print("Loading calibration data ...")
    dataloader, _ = get_wikitext2_olmo(
        nsamples=args["nsamples"],
        seed=0,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )

    print("Running GPTQ quantization ...")
    tick = time.time()

    olmo_sequential(
        model,
        dataloader,
        DEV,
        args,
    )

    print(f"Quantization done in {time.time() - tick:.2f}s")

    model = model.to(DEV)
    model.eval()

    print(f"Loading evaluation dataset={dataset}, limit={limit}")
    texts = load_texts(dataset, limit)

    print(f"Evaluating GPTQ PPL on {dataset} ...")
    metrics = compute_ppl(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=max_length,
        device=DEV,
    )

    out = {
        "model": "olmo-1b",
        "model_id": model_id,
        "method": "gptq_runtime",
        "dataset": dataset,
        "bits": args["wbits"],
        "groupsize": args["groupsize"],
        "nsamples": args["nsamples"],
        "limit": limit,
        "max_length": max_length,
        **metrics,
    }

    output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(out, indent=2))
    print(f"Saved results to {output_json}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["wikitext2", "c4"])
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--output-json", required=True)

    args = parser.parse_args()

    run(
        dataset=args.dataset,
        output_json=args.output_json,
        limit=args.limit,
        max_length=args.max_length,
    )