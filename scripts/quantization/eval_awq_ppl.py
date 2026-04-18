import argparse
import json
import sys
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
        if limit is not None:
            texts = texts[:limit]
    elif dataset_name == "c4":
        n = limit if limit is not None else 1000
        ds = load_dataset("allenai/c4", "en", split=f"validation[:{n}]")
        texts = [x["text"] for x in ds if x["text"].strip()]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return texts


@torch.no_grad()
def compute_ppl(model, tokenizer, texts, max_length=512, device="cuda"):
    enc = tokenizer("\n\n".join(texts), return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    nlls = []
    seq_len = input_ids.size(1)

    for i in range(0, seq_len - 1, max_length):
        chunk = input_ids[:, i:min(i + max_length, seq_len)]
        if chunk.size(1) < 2:
            continue

        out = model(chunk, labels=chunk)
        neg_log_likelihood = out.loss * (chunk.size(1) - 1)
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (seq_len - 1))
    return ppl.item()


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

    awq_results = torch.load(awq_path, map_location="cpu")
    apply_awq(model, awq_results)
    pseudo_quantize_model_weight(model, w_bit=w_bit, q_config=q_config)

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
    texts = load_texts(args.dataset, limit=args.limit)
    model, tokenizer = load_awq_fake_model(
        model_path=args.model_path,
        awq_path=args.awq_path,
        w_bit=args.w_bit,
        q_group_size=args.q_group_size,
    )
    model.to(device)

    ppl = compute_ppl(
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
        "perplexity": ppl,
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

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()