import argparse
import json
import os

from configs.models import MODEL_CONFIGS
from src.model_loader import load_model_and_tokenizer
from src.ablation import zero_out_weight
from src.utils import ensure_dir


def unique_candidates(candidates):
    seen = set()
    unique = []

    for c in candidates:
        key = (c["layer"], c["row"], c["col"])
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


def apply_ablations(model, cfg, candidates, top_k):
    ablated = []

    for c in candidates[:top_k]:
        original = zero_out_weight(
            model=model,
            layer_path=cfg["layer_path"],
            down_proj_path=cfg["down_proj_path"],
            layer_idx=c["layer"],
            row=c["row"],
            col=c["col"],
        )

        ablated.append({
            "layer": c["layer"],
            "row": c["row"],
            "col": c["col"],
            "original_value": original,
            "score": c.get("score"),
            "category": c.get("category"),
        })

    return ablated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", required=True)
    parser.add_argument("--candidate_json", required=True)
    parser.add_argument("--output_dir", default="outputs/ablated_models")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model_key]
    ensure_dir(args.output_dir)

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(cfg["hf_name"])

    print("Loading candidates...")
    with open(args.candidate_json, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    candidates = unique_candidates(candidates)
    print(f"Unique candidates available: {len(candidates)}")

    print(f"Ablating top {args.top_k} weights...")
    ablated_info = apply_ablations(model, cfg, candidates, args.top_k)

    save_path = os.path.join(args.output_dir, f"{args.model_key}_top{args.top_k}_ablated")
    ensure_dir(save_path)

    print(f"Saving model to: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    with open(os.path.join(save_path, "ablated_weights.json"), "w", encoding="utf-8") as f:
        json.dump(ablated_info, f, indent=2)

    print("Done.")
    print(json.dumps(ablated_info, indent=2))


if __name__ == "__main__":
    main()