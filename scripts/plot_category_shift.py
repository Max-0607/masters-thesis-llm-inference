import argparse
import json
import math
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=str, required=True)
    parser.add_argument("--no-sw", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--min-tokens-per-category", type=int, default=2)
    parser.add_argument("--tokens-per-category", type=int, default=5)
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="OLMo-1B",
    )
    return parser.parse_args()


def load_probs(path):
    with open(path, "r") as f:
        data = json.load(f)

    probs = {}
    for item in data["results"]:
        if item["probability"] is not None:
            probs[item["token"]] = item["probability"]
    return probs


def pretty_token(token: str) -> str:
    return token.replace("Ġ", "").strip()


def build_categories():
    return OrderedDict(
        {
            "punctuation": [".", ",", ";", ":", "!", "?"],
            "articles/determiners": [" the", " a", " an", " this", " that", " these", " those"],
            "conjunctions": [" and", " or", " but", " so", " yet"],
            "prepositions": [" of", " in", " to", " for", " with", " on", " at", " by", " from"],
            "pronouns/function": [" I", " you", " he", " she", " they", " we", " it", " is", " not"],
            "content words": [" summer", " winter", " hot", " cold", " warm", " weather", " season", " temperature", " day", " night"],
        }
    )


def compute_log_delta(original_prob, nosw_prob, eps=1e-8):
    delta = (nosw_prob + eps) / (original_prob + eps)
    return math.log10(delta)


def category_color(name):
    palette = {
        "punctuation": "#2ca02c",
        "articles/determiners": "#1f77b4",
        "conjunctions": "#d62728",
        "prepositions": "#ff7f0e",
        "pronouns/function": "#9467bd",
        "content words": "#4d4d4d",
    }
    return palette.get(name, "#444444")


def collect_data(original, no_sw, min_tokens_per_category, tokens_per_category):
    categories = build_categories()

    category_rows = []
    grouped_low_rows = []

    for category, tokens in categories.items():
        vals = []
        kept = []

        for token in tokens:
            if token in original and token in no_sw:
                log_delta = compute_log_delta(original[token], no_sw[token])
                vals.append(log_delta)
                kept.append((token, log_delta))

        if len(vals) >= min_tokens_per_category:
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals)) if len(vals) > 1 else 0.0

            category_rows.append(
                {
                    "category": category,
                    "mean": mean_val,
                    "std": std_val,
                    "n": len(vals),
                    "color": category_color(category),
                }
            )

            # Top N niedrigste Shifts
            kept_sorted_low = sorted(kept, key=lambda x: x[1])[:tokens_per_category]
            grouped_low_rows.append(
                {
                    "category": category,
                    "color": category_color(category),
                    "tokens": [
                        {
                            "token": tok,
                            "pretty": pretty_token(tok),
                            "log_delta": val,
                        }
                        for tok, val in kept_sorted_low
                    ],
                }
            )

    if not category_rows:
        raise ValueError("No categories have enough valid tokens. Regenerate JSONs with more token coverage.")

    category_rows = sorted(category_rows, key=lambda x: x["mean"], reverse=True)
    ordered_categories = [row["category"] for row in category_rows]

    grouped_low_rows = sorted(
        grouped_low_rows,
        key=lambda x: ordered_categories.index(x["category"])
    )

    return category_rows, grouped_low_rows


def plot_category_level(category_rows, output_path, title):
    fig, ax = plt.subplots(figsize=(11, 5.8))

    y_cat = np.arange(len(category_rows))
    means = [row["mean"] for row in category_rows]
    stds = [row["std"] for row in category_rows]
    names = [row["category"] for row in category_rows]
    colors = [row["color"] for row in category_rows]

    ax.barh(
        y_cat,
        means,
        xerr=stds,
        color=colors,
        alpha=0.9,
        edgecolor="none",
        error_kw={"elinewidth": 1, "ecolor": "#444444", "capsize": 3},
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y_cat)
    ax.set_yticklabels(names, fontsize=12)
    ax.set_xlabel("Mean log10(No-SW / Original)", fontsize=13)
    ax.set_title(title, fontsize=17)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.invert_yaxis()

    min_x = min(means[i] - stds[i] for i in range(len(means)))
    max_x = max(means[i] + stds[i] for i in range(len(means)))
    left = min(-1.0, min_x - 0.4)
    right = max(3.0, max_x + 0.8)
    ax.set_xlim(left, right)

    for i, row in enumerate(category_rows):
        x = row["mean"]
        label = f"{x:.2f}  (n={row['n']})"
        if x >= 0:
            text_x = x + 0.06
            ha = "left"
        else:
            text_x = x - 0.06
            ha = "right"

        ax.text(
            text_x,
            i,
            label,
            va="center",
            ha=ha,
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_low_tokens(grouped_low_rows, output_path, title):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axvline(0, color="black", linewidth=1)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    y_positions = []
    current_y = 0
    category_centers = []
    all_x = []

    for group in grouped_low_rows:
        start_y = current_y
        tokens = group["tokens"]
        color = group["color"]

        for token_row in tokens:
            x = token_row["log_delta"]
            y = current_y

            ax.hlines(y=y, xmin=0, xmax=x, color=color, alpha=0.45, linewidth=2.2)
            ax.scatter(x, y, s=75, color=color, zorder=3)

            if x >= 0:
                text_x = x + 0.05
                ha = "left"
            else:
                text_x = x - 0.05
                ha = "right"

            ax.text(
                text_x,
                y,
                f"{token_row['pretty']} ({x:.2f})",
                va="center",
                ha=ha,
                fontsize=10,
            )

            y_positions.append(y)
            all_x.append(x)
            current_y += 1

        end_y = current_y - 1
        center_y = (start_y + end_y) / 2
        category_centers.append((group["category"], center_y))

        ax.axhline(current_y - 0.5, color="#bbbbbb", linewidth=0.8)
        current_y += 1

    ax.set_yticks([])
    ax.set_xlabel("Token log10(No-SW / Original)", fontsize=13)
    ax.set_title(title, fontsize=17)

    left_x = min(-1.2, min(all_x) - 0.6) if all_x else -1.2
    right_x = max(3.0, max(all_x) + 0.6) if all_x else 3.0
    ax.set_xlim(left_x, right_x)
    ax.invert_yaxis()

    for cat, cy in category_centers:
        ax.text(
            left_x + 0.05,
            cy,
            cat,
            va="center",
            ha="left",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    original = load_probs(args.original)
    no_sw = load_probs(args.no_sw)

    category_rows, grouped_low_rows = collect_data(
        original=original,
        no_sw=no_sw,
        min_tokens_per_category=args.min_tokens_per_category,
        tokens_per_category=args.tokens_per_category,
    )

    out1 = os.path.join(args.output_dir, "category_level_shift.png")
    out2 = os.path.join(args.output_dir, "lowest_token_shifts_by_category.png")

    plot_category_level(
        category_rows=category_rows,
        output_path=out1,
        title=f"{args.title_prefix}: category-level shift after superweight removal",
    )

    plot_low_tokens(
        grouped_low_rows=grouped_low_rows,
        output_path=out2,
        title=f"{args.title_prefix}: lowest token shifts per category",
    )

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()