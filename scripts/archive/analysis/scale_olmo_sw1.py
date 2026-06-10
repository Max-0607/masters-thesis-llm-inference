import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SW1 = {
    "layer": 1,
    "row": 1764,
    "col": 1710,
}


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Could not find transformer layers at model.model.layers")


def scale_single_weight(model, scale: float):
    layers = get_transformer_layers(model)

    layer_idx = SW1["layer"]
    row = SW1["row"]
    col = SW1["col"]

    with torch.no_grad():
        weight = layers[layer_idx].mlp.down_proj.weight
        before = weight[row, col].item()
        weight[row, col] *= scale
        after = weight[row, col].item()

    return before, after


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print(f"Loading model from {args.model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"Scaling SW1 by factor {args.scale} ...")
    before, after = scale_single_weight(model, args.scale)

    print(f"SW1 value: {before:.6f} -> {after:.6f}")

    print(f"Saving model to {output_dir} ...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    meta = {
        "model_id": args.model_id,
        "scale": args.scale,
        "scaled_weight": SW1,
        "before": before,
        "after": after,
    }

    with open(output_dir / "scale_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()