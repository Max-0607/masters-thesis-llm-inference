import argparse
import json
import math
import os
import copy
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_STOPWORDS = [
    "the", "a", "an", ".", ",", "to", "of", "in", "and", "is", "are",
    "was", "were", "for", "on", "with", "as", "at", "by", "that",
    "this", "it", "be", "or", "from", "but", "if", "then", "so",
    "you", "he", "she", "they", "we", "I", "my", "his", "her", "their"
]

DEFAULT_PROMPTS = {
    "reasoning": [
        "If Alice has 3 apples and buys 2 more, how many apples does she have?",
        "Tom is older than Jim. Jim is older than Sam. Who is the oldest?",
        "A train travels 60 miles in 1 hour. How far does it travel in 3 hours?",
        "If all roses are flowers and some flowers fade quickly, what can we conclude about roses?",
        "What comes next in the pattern: 2, 4, 8, 16, ?"
    ],
    "factual": [
        "The capital of France is",
        "The largest planet in the solar system is",
        "Water freezes at",
        "The author of Hamlet is",
        "The chemical symbol for gold is"
    ],
    "language": [
        "Summer is hot. Winter is",
        "The opposite of happy is",
        "A dog is an animal and a rose is a",
        "Complete the sentence: The child opened the door and",
        "Write one word that best fits: Bread is made from"
    ]
}


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Could not find transformer layers at model.model.layers")


def get_down_proj(layer):
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
        return layer.mlp.down_proj
    raise ValueError("Could not find mlp.down_proj")


def ablate_single_weight_(model, layer_idx: int, out_ch: int, in_ch: int) -> float:
    """
    Sets weight[out_ch, in_ch] = 0 in-place.
    Returns original value for logging.
    """
    layers = get_transformer_layers(model)
    down_proj = get_down_proj(layers[layer_idx])
    with torch.no_grad():
        old_val = down_proj.weight[out_ch, in_ch].item()
        down_proj.weight[out_ch, in_ch] = 0.0
    return old_val


def tokenize_stopwords(tokenizer, stopwords: List[str]) -> Dict[str, List[int]]:
    """
    Map stopword string -> token ids.
    We try both raw word and word with leading space to better match BPE behavior.
    """
    mapping = {}
    for w in stopwords:
        forms = [w, " " + w]
        ids = set()
        for form in forms:
            toks = tokenizer.encode(form, add_special_tokens=False)
            if len(toks) == 1:
                ids.add(toks[0])
        mapping[w] = sorted(ids)
    return mapping


def build_stopword_id_set(tokenizer, stopwords: List[str]) -> Tuple[set, Dict[str, List[int]]]:
    mapping = tokenize_stopwords(tokenizer, stopwords)
    id_set = set()
    for ids in mapping.values():
        id_set.update(ids)
    return id_set, mapping


def shannon_entropy(probs: torch.Tensor) -> float:
    eps = 1e-12
    p = probs.clamp_min(eps)
    return float(-(p * p.log()).sum().item())


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    eps = 1e-12
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return float((p * (p.log() - q.log())).sum().item())


def get_next_token_distribution(model, tokenizer, prompt: str, max_len: int = 64):
    device = next(model.parameters()).device
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_len
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, use_cache=False, return_dict=True)

    logits = out.logits[0, -1, :]
    probs = torch.softmax(logits.float(), dim=-1)
    return probs


def decode_topk(tokenizer, probs: torch.Tensor, k: int = 10):
    vals, idxs = torch.topk(probs, k)
    out = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        token_str = tokenizer.decode([i]).replace("\n", "\\n")
        out.append({
            "token_id": i,
            "token": token_str,
            "prob": v
        })
    return out


def compute_metrics(
    probs_orig: torch.Tensor,
    probs_ablt: torch.Tensor,
    stopword_ids: set
) -> Dict:
    vocab_size = probs_orig.shape[0]

    mask = torch.zeros(vocab_size, dtype=torch.bool, device=probs_orig.device)
    if stopword_ids:
        idx = torch.tensor(sorted(stopword_ids), device=probs_orig.device)
        mask[idx] = True

    stopword_mass_orig = float(probs_orig[mask].sum().item())
    stopword_mass_ablt = float(probs_ablt[mask].sum().item())

    top10_idx_orig = torch.topk(probs_orig, 10).indices
    top10_idx_ablt = torch.topk(probs_ablt, 10).indices

    top10_stopword_mass_orig = float(
        probs_orig[top10_idx_orig][mask[top10_idx_orig]].sum().item()
    )
    top10_stopword_mass_ablt = float(
        probs_ablt[top10_idx_ablt][mask[top10_idx_ablt]].sum().item()
    )

    return {
        "stopword_mass_orig": stopword_mass_orig,
        "stopword_mass_ablt": stopword_mass_ablt,
        "stopword_mass_delta": stopword_mass_ablt - stopword_mass_orig,
        "top10_stopword_mass_orig": top10_stopword_mass_orig,
        "top10_stopword_mass_ablt": top10_stopword_mass_ablt,
        "top10_stopword_mass_delta": top10_stopword_mass_ablt - top10_stopword_mass_orig,
        "entropy_orig": shannon_entropy(probs_orig),
        "entropy_ablt": shannon_entropy(probs_ablt),
        "entropy_delta": shannon_entropy(probs_ablt) - shannon_entropy(probs_orig),
        "kl_orig_to_ablt": kl_divergence(probs_orig, probs_ablt),
        "kl_ablt_to_orig": kl_divergence(probs_ablt, probs_orig),
    }


def aggregate_results(rows: List[Dict]) -> Dict:
    if not rows:
        return {}

    keys = [
        "stopword_mass_orig",
        "stopword_mass_ablt",
        "stopword_mass_delta",
        "top10_stopword_mass_orig",
        "top10_stopword_mass_ablt",
        "top10_stopword_mass_delta",
        "entropy_orig",
        "entropy_ablt",
        "entropy_delta",
        "kl_orig_to_ablt",
        "kl_ablt_to_orig",
    ]

    summary = {"n": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows]
        summary[k] = {
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--category", type=str, default="reasoning",
                        choices=list(DEFAULT_PROMPTS.keys()) + ["all"])
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--max-len", type=int, default=64)

    # Mistral-7B v0.1 super weight from the paper:
    # layer 1, mlp.down_proj [2070, 7310]
    parser.add_argument("--layer-idx", type=int, default=1)
    parser.add_argument("--out-ch", type=int, default=2070)
    parser.add_argument("--in-ch", type=int, default=7310)

    args = parser.parse_args()

    categories = (
        DEFAULT_PROMPTS.keys() if args.category == "all" else [args.category]
    )
    prompts = []
    for cat in categories:
        for p in DEFAULT_PROMPTS[cat]:
            prompts.append({"category": cat, "prompt": p})

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading original model: {args.model_id}")
    model_orig = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model_orig.eval()

    print("Loading ablated model copy...")
    model_ablt = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model_ablt.eval()

    old_val = ablate_single_weight_(model_ablt, args.layer_idx, args.out_ch, args.in_ch)
    print(
        f"Ablated weight at layer={args.layer_idx}, "
        f"weight[{args.out_ch}, {args.in_ch}] "
        f"(old value={old_val:.6f})"
    )

    stopword_ids, stopword_map = build_stopword_id_set(tokenizer, DEFAULT_STOPWORDS)
    print(f"Matched {len(stopword_ids)} single-token stopword ids")

    rows = []

    for item in prompts:
        category = item["category"]
        prompt = item["prompt"]

        probs_orig = get_next_token_distribution(
            model_orig, tokenizer, prompt, max_len=args.max_len
        )
        probs_ablt = get_next_token_distribution(
            model_ablt, tokenizer, prompt, max_len=args.max_len
        )

        metrics = compute_metrics(probs_orig, probs_ablt, stopword_ids)

        row = {
            "category": category,
            "prompt": prompt,
            **metrics,
            "top10_orig": decode_topk(tokenizer, probs_orig, k=10),
            "top10_ablt": decode_topk(tokenizer, probs_ablt, k=10),
        }
        rows.append(row)

        print(
            f"[{category}] stopword_mass "
            f"{metrics['stopword_mass_orig']:.4f} -> {metrics['stopword_mass_ablt']:.4f} | "
            f"KL={metrics['kl_orig_to_ablt']:.4f}"
        )

    summary = {
        "model_id": args.model_id,
        "superweight": {
            "layer_idx": args.layer_idx,
            "out_ch": args.out_ch,
            "in_ch": args.in_ch,
            "old_value": old_val
        },
        "stopword_map": stopword_map,
        "summary_all": aggregate_results(rows),
        "summary_by_category": {},
        "rows": rows
    }

    for cat in DEFAULT_PROMPTS.keys():
        cat_rows = [r for r in rows if r["category"] == cat]
        if cat_rows:
            summary["summary_by_category"][cat] = aggregate_results(cat_rows)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main()