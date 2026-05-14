import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SUPERWEIGHTS_OLMO1B = [
    (1, 1764, 1710),
    (1, 1764, 8041),
]


def apply_superweight_scaling(model, alpha: float):
    for layer, row, col in SUPERWEIGHTS_OLMO1B:
        weight = model.model.layers[layer].mlp.down_proj.weight
        old_value = weight[row, col].item()

        with torch.no_grad():
            weight[row, col] *= alpha

        new_value = weight[row, col].item()
        print(
            f"Layer {layer} down_proj[{row}, {col}]: "
            f"{old_value:.6f} -> {new_value:.6f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/olmo1b")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    print(f"Applying superweight scaling alpha={args.alpha}")
    apply_superweight_scaling(model, args.alpha)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving scaled model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("Done.")


if __name__ == "__main__":
    main()