from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--original-json", required=True)
    parser.add_argument("--redistribution-json", required=True)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.original_json) as f:
        original = json.load(f)

    with open(args.redistribution_json) as f:
        redistributed = json.load(f)

    layers = [x["layer"] for x in original["input"]]

    input_delta = []
    output_delta = []

    for orig, red in zip(original["input"], redistributed["input"]):
        input_delta.append(
            red["max_abs_value"] - orig["max_abs_value"]
        )

    for orig, red in zip(original["output"], redistributed["output"]):
        output_delta.append(
            red["max_abs_value"] - orig["max_abs_value"]
        )

    plt.figure(figsize=(10, 5))

    plt.bar(
        [x - 0.2 for x in layers],
        input_delta,
        width=0.4,
        label="Input Δ"
    )

    plt.bar(
        [x + 0.2 for x in layers],
        output_delta,
        width=0.4,
        label="Output Δ"
    )

    plt.axhline(0, color="black", linewidth=1)

    plt.xlabel("Layer")
    plt.ylabel("SW-Dropout − Original")
    #plt.title(
       # "Change in Maximum Activations After Superweight Dropout"
    #)

    plt.legend()
    plt.tight_layout()

    plot_path = output_dir / "activation_delta.png"

    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved to {plot_path}")


if __name__ == "__main__":
    main()