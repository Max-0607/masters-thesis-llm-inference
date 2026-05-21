import argparse
import json
import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from quantization.awq.awq.quantize.pre_quant import apply_awq
from quantization.awq.awq.quantize.quantizer import pseudo_quantize_model_weight


def load_texts(dataset_name: str, limit: int | None = None):
    if dataset_name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [x["text"] for x in ds if x["text"].strip()]
        return texts[:limit] if limit is not None else texts

    if dataset_name == "c4":
        n = limit if limit is not None else 1000

        ds = load_dataset(
            "allenai/c4",
            "en",
            split="validation",
            streaming=True,
        )

        texts = []
        for x in ds:
            text = x.get("text", "")
            if text.strip():
                texts.append(text)

            if len(texts) >= n:
                break

        return texts

    raise ValueError(f"Unsupported dataset: {dataset_name}")


@torch.no_grad()
def compute_ppl(model, tokenizer, texts, max_length=512, device="cuda"):
    total_loss = 0.0
    total_tokens = 0

    model.eval()

    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        input_ids = enc.input_ids.to(device)

        if input_ids.shape[1] < 2:
            continue

        out = model(
            input_ids=input_ids,
            labels=input_ids,
            use_cache=False,
            return_dict=True,
        )

        num_tokens = input_ids.shape[1] - 1
        loss = out.loss.float().item()

        total_loss += loss * num_tokens
        total_tokens += num_tokens

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


def load_awq_fake_model(model_path: str, awq_path: str, w_bit=4, q_group_size=128):
    q_config = {"zero_point": True, "q_group_size": q_group_size}
    torch_dtype = torch.float16

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=None,
    )

    print(f"Loading AWQ results from: {awq_path}", flush=True)
    awq_results = torch.load(awq_path, map_location="cpu")

    apply_awq(model, awq_results)

    print(
        f"Applying pseudo weight quantization: w_bit={w_bit}, group_size={q_group_size}",
        flush=True,
    )

    pseudo_quantize_model_weight(
        model,
        w_bit=w_bit,
        q_config=q_config,
    )

    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--awq-path", required=True)
    parser.add_argument("--dataset", required=True, choices=["wikitext2", "c4"])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--w-bit", type=int, default=4)
    parser.add_argument("--q-group-size", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"Loading dataset={args.dataset}, limit={args.limit}, max_length={args.max_length}",
        flush=True,
    )
    texts = load_texts(args.dataset, limit=args.limit)

    print(f"Loaded {len(texts)} texts.", flush=True)

    model, tokenizer = load_awq_fake_model(
        model_path=args.model_path,
        awq_path=args.awq_path,
        w_bit=args.w_bit,
        q_group_size=args.q_group_size,
    )

    model.to(device)

    metrics = compute_ppl(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=args.max_length,
        device=device,
    )

    result = {
        "model_path": args.model_path,
        "awq_path": args.awq_path,
        "dataset": args.dataset,
        **metrics,
        "limit": args.limit,
        "max_length": args.max_length,
        "w_bit": args.w_bit,
        "q_group_size": args.q_group_size,
        "backend": "fake_awq",
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()