import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from configs.models import MODEL_CONFIGS
from src.model_loader import load_model_and_tokenizer
from src.hooks import ActivationRecorder, get_nested_attr
from src.activation_analysis import summarize_all_layers
from src.utils import ensure_dir


def run_single_prompt(model, tokenizer, prompt, layer_path, down_proj_path):
    layers = get_nested_attr(model, layer_path)

    recorder = ActivationRecorder()
    recorder.register_on_layers(layers, down_proj_path)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=64,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        model(**inputs)

    recorder.remove()

    input_summaries = summarize_all_layers(recorder.inputs)
    output_summaries = summarize_all_layers(recorder.outputs)

    return input_summaries, output_summaries


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-key",
        type=str,
        default="olmo-1b",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model key from configs/models.py",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="If it is winter, then it is cold. It is winter. What follows?",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/activation_analysis",
    )

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    cfg = MODEL_CONFIGS[args.model_key]

    model, tokenizer = load_model_and_tokenizer(cfg["hf_name"])

    input_summaries, output_summaries = run_single_prompt(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        layer_path=cfg["layer_path"],
        down_proj_path=cfg["down_proj_path"],
    )

    layers_in = [x["layer"] for x in input_summaries]
    values_in = [x["max_abs_value"] for x in input_summaries]

    layers_out = [x["layer"] for x in output_summaries]
    values_out = [x["max_abs_value"] for x in output_summaries]

    plt.figure(figsize=(10, 5))
    plt.plot(layers_in, values_in, marker="o", label="input max")
    plt.plot(layers_out, values_out, marker="o", label="output max")
    plt.xlabel("Layer")
    plt.ylabel("Max abs activation")
    plt.title(f"{args.model_key}: Input vs Output Max Activation per Layer")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(
        args.output_dir,
        f"{args.model_key}_input_output_max.png",
    )
    plt.savefig(plot_path, dpi=200)
    plt.close()

    json_path = os.path.join(
        args.output_dir,
        f"{args.model_key}_input_output_max.json",
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_key": args.model_key,
                "model_id": cfg["hf_name"],
                "prompt": args.prompt,
                "input": input_summaries,
                "output": output_summaries,
            },
            f,
            indent=2,
        )

    print(f"Saved plot to: {plot_path}")
    print(f"Saved json to: {json_path}")


if __name__ == "__main__":
    main()