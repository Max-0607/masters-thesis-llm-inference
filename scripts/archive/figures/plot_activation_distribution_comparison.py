from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from configs.models import MODEL_CONFIGS
from src.model_loader import load_model_and_tokenizer
from src.hooks import get_nested_attr
from src.utils import ensure_dir


def collect_down_proj_outputs(
    model,
    tokenizer,
    prompt,
    layer_path,
    down_proj_path,
    max_length,
):
    layers = get_nested_attr(model, layer_path)
    outputs = {}
    handles = []

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            outputs[layer_idx] = output.detach().float().cpu()
        return hook

    for layer_idx, layer in enumerate(layers):
        module = get_nested_attr(layer, down_proj_path)
        handles.append(module.register_forward_hook(make_hook(layer_idx)))

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

    for handle in handles:
        handle.remove()

    return outputs


def compute_metrics(outputs):
    metrics = []

    for layer_idx, y in sorted(outputs.items()):
        flat = y.reshape(-1, y.shape[-1]).abs()

        total = flat.sum(dim=-1, keepdim=True) + 1e-12
        p = flat / total

        top1_share = p.max(dim=-1).values
        top10_share = torch.topk(p, k=10, dim=-1).values.sum(dim=-1)

        metrics.append(
            {
                "layer": int(layer_idx),
                "top1_share_mean": float(top1_share.mean().item()),
                "top10_share_mean": float(top10_share.mean().item()),
            }
        )

    return metrics


def run_model(model_path, model_key, prompt, max_length):
    cfg = MODEL_CONFIGS[model_key]

    model, tokenizer = load_model_and_tokenizer(model_path)

    outputs = collect_down_proj_outputs(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        layer_path=cfg["layer_path"],
        down_proj_path=cfg["down_proj_path"],
        max_length=max_length,
    )

    metrics = compute_metrics(outputs)

    del model
    del tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def plot_metrics(original, redistributed, output_path):
    layers = [x["layer"] for x in original]

    orig_top1 = [x["top1_share_mean"] for x in original]
    red_top1 = [x["top1_share_mean"] for x in redistributed]

    orig_top10 = [x["top10_share_mean"] for x in original]
    red_top10 = [x["top10_share_mean"] for x in redistributed]

    delta_top1 = [
        red - orig
        for red, orig in zip(red_top1, orig_top1)
    ]

    delta_top10 = [
        red - orig
        for red, orig in zip(red_top10, orig_top10)
    ]

    plt.figure(figsize=(10, 5))

    plt.plot(
        layers,
        delta_top1,
        marker="o",
        label="Top-1 share Δ",
    )

    plt.plot(
        layers,
        delta_top10,
        marker="o",
        label="Top-10 share Δ",
    )

    plt.axhline(
        0,
        color="black",
        linewidth=1,
    )

    plt.xlabel("Layer")
    plt.ylabel("SW-Dropout − Original")
    plt.title("Change in Activation Concentration After Superweight Dropout")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-key", default="olmo-1b")
    parser.add_argument("--original-model-path", default=None)
    parser.add_argument("--redistribution-model-path", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument(
        "--prompt",
        default="If it is winter, then it is cold. It is winter. What follows?",
    )

    parser.add_argument("--max-length", type=int, default=64)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    cfg = MODEL_CONFIGS[args.model_key]

    original_model_path = (
        args.original_model_path
        if args.original_model_path is not None
        else cfg["hf_name"]
    )

    print(f"Loading original model: {original_model_path}")
    original_metrics = run_model(
        model_path=original_model_path,
        model_key=args.model_key,
        prompt=args.prompt,
        max_length=args.max_length,
    )

    print(f"Loading redistribution model: {args.redistribution_model_path}")
    redistributed_metrics = run_model(
        model_path=args.redistribution_model_path,
        model_key=args.model_key,
        prompt=args.prompt,
        max_length=args.max_length,
    )

    json_path = output_dir / "activation_concentration_delta.json"
    plot_path = output_dir / "activation_concentration_delta.png"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_key": args.model_key,
                "original_model": original_model_path,
                "redistribution_model": args.redistribution_model_path,
                "prompt": args.prompt,
                "max_length": args.max_length,
                "original": original_metrics,
                "redistribution": redistributed_metrics,
            },
            f,
            indent=2,
        )

    plot_metrics(
        original=original_metrics,
        redistributed=redistributed_metrics,
        output_path=plot_path,
    )

    print(f"Saved plot to: {plot_path}")
    print(f"Saved json to: {json_path}")


if __name__ == "__main__":
    main()