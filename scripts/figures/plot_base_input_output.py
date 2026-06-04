from __future__ import annotations

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


def run_single_prompt(
    model,
    tokenizer,
    prompt: str,
    layer_path: str,
    down_proj_path: str,
    max_length: int,
):
    layers = get_nested_attr(model, layer_path)

    recorder = ActivationRecorder()
    recorder.register_on_layers(layers, down_proj_path)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()

    with torch.no_grad():
        model(**inputs)

    recorder.remove()

    input_summaries = summarize_all_layers(recorder.inputs)
    output_summaries = summarize_all_layers(recorder.outputs)

    return input_summaries, output_summaries


def plot_input_output_max(
    input_summaries,
    output_summaries,
    title: str,
    plot_path: str,
):
    layers_in = [x["layer"] for x in input_summaries]
    values_in = [x["max_abs_value"] for x in input_summaries]

    layers_out = [x["layer"] for x in output_summaries]
    values_out = [x["max_abs_value"] for x in output_summaries]

    plt.figure(figsize=(10, 5))
    plt.plot(
        layers_in,
        values_in,
        marker="o",
        label="input max",
    )
    plt.plot(
        layers_out,
        values_out,
        marker="o",
        label="output max",
    )
    plt.xlabel("Layer")
    plt.ylabel("Max abs activation")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


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
        "--model-path",
        type=str,
        default=None,
        help=(
            "Optional local or HF model path. "
            "If omitted, cfg['hf_name'] from MODEL_CONFIGS is used."
        ),
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name used in plot/json filenames.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "If it is winter, then it is cold. "
            "It is winter. What follows?"
        ),
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/activation_analysis",
    )

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    cfg = MODEL_CONFIGS[args.model_key]

    model_id = (
        args.model_path
        if args.model_path is not None
        else cfg["hf_name"]
    )

    run_name = (
        args.run_name
        if args.run_name is not None
        else args.model_key
    )

    safe_run_name = (
        run_name
        .replace("/", "_")
        .replace(" ", "_")
    )

    print(f"Model key: {args.model_key}")
    print(f"Model id/path: {model_id}")
    print(f"Run name: {run_name}")

    model, tokenizer = load_model_and_tokenizer(model_id)

    input_summaries, output_summaries = run_single_prompt(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        layer_path=cfg["layer_path"],
        down_proj_path=cfg["down_proj_path"],
        max_length=args.max_length,
    )

    plot_path = os.path.join(
        args.output_dir,
        f"{safe_run_name}_input_output_max.png",
    )

    title = f"{run_name}: Input vs Output Max Activation per Layer"

    plot_input_output_max(
        input_summaries=input_summaries,
        output_summaries=output_summaries,
        title=title,
        plot_path=plot_path,
    )

    json_path = os.path.join(
        args.output_dir,
        f"{safe_run_name}_input_output_max.json",
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_key": args.model_key,
                "model_id": model_id,
                "run_name": run_name,
                "prompt": args.prompt,
                "max_length": args.max_length,
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