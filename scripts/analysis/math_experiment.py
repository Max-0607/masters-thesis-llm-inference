import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_STOPWORDS = [
    "the", "a", "an", ".", ",", "to", "of", "in", "and", "is", "are",
    "was", "were", "for", "on", "with", "as", "at", "by", "that",
    "this", "it", "be", "or", "from", "but", "if", "then", "so",
    "you", "he", "she", "they", "we", "I", "my", "his", "her", "their"
]

MATH_PROMPTS = {
    "arithmetic": [
        {"prompt": "2 + 3 =", "answer": "5"},
        {"prompt": "7 - 4 =", "answer": "3"},
        {"prompt": "6 + 1 =", "answer": "7"},
        {"prompt": "9 - 2 =", "answer": "7"},
        {"prompt": "4 + 4 =", "answer": "8"},
        {"prompt": "8 - 5 =", "answer": "3"},
        {"prompt": "3 + 6 =", "answer": "9"},
        {"prompt": "5 + 2 =", "answer": "7"},
    ],
    "multiplication": [
        {"prompt": "2 * 3 =", "answer": "6"},
        {"prompt": "3 * 3 =", "answer": "9"},
        {"prompt": "4 * 2 =", "answer": "8"},
        {"prompt": "5 * 1 =", "answer": "5"},
        {"prompt": "6 * 1 =", "answer": "6"},
        {"prompt": "7 * 1 =", "answer": "7"},
    ],
    "word_math": [
        {"prompt": "Anna has 3 apples and gets 2 more. She now has", "answer": "5"},
        {"prompt": "Tom had 7 books and gave away 4. He now has", "answer": "3"},
        {"prompt": "A car drives 2 hours at 3 miles per hour. It travels", "answer": "6"},
        {"prompt": "Lisa has 4 pencils and buys 4 more. She now has", "answer": "8"},
        {"prompt": "There are 9 birds and 2 fly away. Remaining birds:", "answer": "7"},
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


def tokenize_single_token_forms(tokenizer, text: str) -> List[int]:
    ids = set()
    for form in [text, " " + text]:
        toks = tokenizer.encode(form, add_special_tokens=False)
        if len(toks) == 1:
            ids.add(toks[0])
    return sorted(ids)


def build_id_set(tokenizer, items: List[str]) -> Tuple[set, Dict[str, List[int]]]:
    mapping = {}
    id_set = set()
    for item in items:
        ids = tokenize_single_token_forms(tokenizer, item)
        mapping[item] = ids
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


def rank_of_best_answer_token(probs: torch.Tensor, answer_ids: List[int]) -> int:
    if not answer_ids:
        return -1
    sorted_ids = torch.argsort(probs, descending=True)
    rank_map = {tok.item(): idx + 1 for idx, tok in enumerate(sorted_ids)}
    ranks = [rank_map[i] for i in answer_ids if i in rank_map]
    return min(ranks) if ranks else -1


def prob_of_answer_ids(probs: torch.Tensor, answer_ids: List[int]) -> float:
    if not answer_ids:
        return 0.0
    idx = torch.tensor(answer_ids, device=probs.device, dtype=torch.long)
    return float(probs[idx].sum().item())


def mass_from_id_set(probs: torch.Tensor, id_set: set) -> float:
    if not id_set:
        return 0.0
    idx = torch.tensor(sorted(id_set), device=probs.device, dtype=torch.long)
    return float(probs[idx].sum().item())


def aggregate_results(rows: List[Dict]) -> Dict:
    if not rows:
        return {}

    numeric_keys = [
        "correct_prob_orig",
        "correct_prob_ablt",
        "correct_prob_delta",
        "digit_mass_orig",
        "digit_mass_ablt",
        "digit_mass_delta",
        "stopword_mass_orig",
        "stopword_mass_ablt",
        "stopword_mass_delta",
        "entropy_orig",
        "entropy_ablt",
        "entropy_delta",
        "kl_orig_to_ablt",
        "kl_ablt_to_orig",
    ]

    summary = {"n": len(rows)}

    for key in numeric_keys:
        vals = [r[key] for r in rows]
        summary[key] = {
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
        }

    valid_rank_orig = [r["correct_rank_orig"] for r in rows if r["correct_rank_orig"] > 0]
    valid_rank_ablt = [r["correct_rank_ablt"] for r in rows if r["correct_rank_ablt"] > 0]

    summary["correct_rank_orig"] = {
        "mean": (sum(valid_rank_orig) / len(valid_rank_orig)) if valid_rank_orig else -1,
        "min": min(valid_rank_orig) if valid_rank_orig else -1,
        "max": max(valid_rank_orig) if valid_rank_orig else -1,
    }
    summary["correct_rank_ablt"] = {
        "mean": (sum(valid_rank_ablt) / len(valid_rank_ablt)) if valid_rank_ablt else -1,
        "min": min(valid_rank_ablt) if valid_rank_ablt else -1,
        "max": max(valid_rank_ablt) if valid_rank_ablt else -1,
    }

    summary["top1_correct_orig_rate"] = sum(
        1 for r in rows if r["correct_rank_orig"] == 1
    ) / len(rows)
    summary["top1_correct_ablt_rate"] = sum(
        1 for r in rows if r["correct_rank_ablt"] == 1
    ) / len(rows)

    summary["top5_correct_orig_rate"] = sum(
        1 for r in rows if 0 < r["correct_rank_orig"] <= 5
    ) / len(rows)
    summary["top5_correct_ablt_rate"] = sum(
        1 for r in rows if 0 < r["correct_rank_ablt"] <= 5
    ) / len(rows)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--category", type=str, default="all",
                        choices=list(MATH_PROMPTS.keys()) + ["all"])
    parser.add_argument("--layer-idx", type=int, required=True)
    parser.add_argument("--out-ch", type=int, required=True)
    parser.add_argument("--in-ch", type=int, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--max-len", type=int, default=64)

    args = parser.parse_args()

    categories = list(MATH_PROMPTS.keys()) if args.category == "all" else [args.category]
    prompts = []
    for cat in categories:
        for item in MATH_PROMPTS[cat]:
            prompts.append({
                "category": cat,
                "prompt": item["prompt"],
                "answer": item["answer"],
            })

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()

    layers = get_transformer_layers(model)
    down_proj = get_down_proj(layers[args.layer_idx])

    stopword_id_set, stopword_map = build_id_set(tokenizer, DEFAULT_STOPWORDS)
    digit_id_set, digit_map = build_id_set(tokenizer, [str(i) for i in range(10)])

    print(f"Matched {len(stopword_id_set)} stopword token ids")
    print(f"Matched {len(digit_id_set)} digit token ids")

    rows = []

    with torch.no_grad():
        old_val = down_proj.weight[args.out_ch, args.in_ch].item()

    print(
        f"Using superweight at layer={args.layer_idx}, "
        f"weight[{args.out_ch}, {args.in_ch}] "
        f"(old value={old_val:.6f})"
    )

    for item in prompts:
        category = item["category"]
        prompt = item["prompt"]
        answer = item["answer"]

        answer_ids = tokenize_single_token_forms(tokenizer, answer)

        probs_orig = get_next_token_distribution(model, tokenizer, prompt, max_len=args.max_len)

        with torch.no_grad():
            down_proj.weight[args.out_ch, args.in_ch] = 0.0

        probs_ablt = get_next_token_distribution(model, tokenizer, prompt, max_len=args.max_len)

        with torch.no_grad():
            down_proj.weight[args.out_ch, args.in_ch] = old_val

        correct_prob_orig = prob_of_answer_ids(probs_orig, answer_ids)
        correct_prob_ablt = prob_of_answer_ids(probs_ablt, answer_ids)

        row = {
            "category": category,
            "prompt": prompt,
            "answer": answer,
            "answer_token_ids": answer_ids,
            "correct_prob_orig": correct_prob_orig,
            "correct_prob_ablt": correct_prob_ablt,
            "correct_prob_delta": correct_prob_ablt - correct_prob_orig,
            "correct_rank_orig": rank_of_best_answer_token(probs_orig, answer_ids),
            "correct_rank_ablt": rank_of_best_answer_token(probs_ablt, answer_ids),
            "digit_mass_orig": mass_from_id_set(probs_orig, digit_id_set),
            "digit_mass_ablt": mass_from_id_set(probs_ablt, digit_id_set),
            "digit_mass_delta": mass_from_id_set(probs_ablt, digit_id_set) - mass_from_id_set(probs_orig, digit_id_set),
            "stopword_mass_orig": mass_from_id_set(probs_orig, stopword_id_set),
            "stopword_mass_ablt": mass_from_id_set(probs_ablt, stopword_id_set),
            "stopword_mass_delta": mass_from_id_set(probs_ablt, stopword_id_set) - mass_from_id_set(probs_orig, stopword_id_set),
            "entropy_orig": shannon_entropy(probs_orig),
            "entropy_ablt": shannon_entropy(probs_ablt),
            "entropy_delta": shannon_entropy(probs_ablt) - shannon_entropy(probs_orig),
            "kl_orig_to_ablt": kl_divergence(probs_orig, probs_ablt),
            "kl_ablt_to_orig": kl_divergence(probs_ablt, probs_orig),
            "top10_orig": decode_topk(tokenizer, probs_orig, k=10),
            "top10_ablt": decode_topk(tokenizer, probs_ablt, k=10),
        }

        rows.append(row)

        print(
            f"[{category}] ans={answer!r} "
            f"p(correct): {correct_prob_orig:.4f} -> {correct_prob_ablt:.4f} | "
            f"rank: {row['correct_rank_orig']} -> {row['correct_rank_ablt']} | "
            f"KL={row['kl_orig_to_ablt']:.4f}"
        )

    summary = {
        "model_id": args.model_id,
        "superweight": {
            "layer_idx": args.layer_idx,
            "out_ch": args.out_ch,
            "in_ch": args.in_ch,
            "old_value": old_val
        },
        "digit_map": digit_map,
        "stopword_map": stopword_map,
        "summary_all": aggregate_results(rows),
        "summary_by_category": {},
        "rows": rows
    }

    for cat in categories:
        cat_rows = [r for r in rows if r["category"] == cat]
        summary["summary_by_category"][cat] = aggregate_results(cat_rows)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main()