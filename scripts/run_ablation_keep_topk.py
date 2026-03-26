import argparse
import json
import os
import torch

from configs.models import MODEL_CONFIGS
from src.model_loader import load_model_and_tokenizer
from src.utils import ensure_dir


def get_layers(model, layer_path: str):
    obj = model
    for attr in layer_path.split("."):
        obj = getattr(obj, attr)
    return obj


def get_module(layer, module_path: str):
    obj = layer
    for attr in module_path.split("."):
        obj = getattr(obj, attr)
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, required=True)
    parser.add_argument("--candidate_json", type=str, required=True)
    parser.add_argument("--top_k", type=int, required=True)
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model_key]
    hf_name = cfg["hf_name"]
    layer_path = cfg["layer_path"]
    down_proj_path = cfg["down_proj_path"]

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(hf_name)

    print("Loading candidates...")
    with open(args.candidate_json, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    candidates = candidates[:args.top_k]

    layers = get_layers(model, layer_path)

    print(f"Keeping only top {args.top_k} weights (zeroing all other down_proj weights)...")

    # Save original weights, then zero everything
    original_weights = {}
    for layer_idx, layer in enumerate(layers):
        module = get_module(layer, down_proj_path)
        W = module.weight.data
        original_weights[layer_idx] = W.clone()
        W.zero_()

    # Restore only top-k weights
    kept = []
    for c in candidates:
        layer_idx = c["layer"]
        row = c["row"]
        col = c["col"]

        module = get_module(layers[layer_idx], down_proj_path)
        W = module.weight.data
        W[row, col] = original_weights[layer_idx][row, col]

        kept.append({
            "layer": layer_idx,
            "row": row,
            "col": col,
            "restored_value": float(original_weights[layer_idx][row, col].item()),
            "score": c.get("score"),
            "category": c.get("category"),
        })

    save_path = f"outputs/ablated_models/{args.model_key}_keep_top{args.top_k}"
    ensure_dir(save_path)

    print(f"Saving model to: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    with open(os.path.join(save_path, "kept_weights.json"), "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2)

    print("Done.")
    print(json.dumps(kept, indent=2))


if __name__ == "__main__":
    main()