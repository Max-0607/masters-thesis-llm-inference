import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = "outputs/eval/olmo_task_compare"
OUT_DIR = "outputs/plots"

TASKS = [
    "hellaswag",
    "lambada_openai",
    "mgsm_direct_en",
    "sciq",
]

TASK_LABELS = {
    "hellaswag": "Commonsense\nReasoning",
    "lambada_openai": "Language\nModeling",
    "mgsm_direct_en": "Mathematical\nReasoning",
    "sciq": "Scientific\nKnowledge",
}

TOPK_LABELS = ["top1", "top2", "top3", "top4", "top5", "top10"]

TASK_METRIC = {
    "hellaswag": "acc_norm,none",
    "lambada_openai": "acc,none",
    "mgsm_direct_en": "exact_match,flexible-extract",
    "sciq": "acc,none",
}


# -----------------------
# Data Loading
# -----------------------

def load_metric(path, task, metric):
    with open(path, "r") as f:
        data = json.load(f)
    return data["results"][task][metric]


def build_drop_matrix():
    matrix = []

    for task in TASKS:
        metric = TASK_METRIC[task]
        baseline = load_metric(
            os.path.join(BASE_DIR, f"baseline_{task}.json"),
            task,
            metric,
        )

        row = []
        for label in TOPK_LABELS:
            score = load_metric(
                os.path.join(BASE_DIR, f"{label}_{task}.json"),
                task,
                metric,
            )
            row.append(baseline - score)

        matrix.append(row)

    return np.array(matrix)


# -----------------------
# Processing
# -----------------------

def row_normalize(matrix):
    norm = np.zeros_like(matrix)

    for i, row in enumerate(matrix):
        rmin, rmax = row.min(), row.max()
        if abs(rmax - rmin) < 1e-9:
            norm[i] = 1.0
        else:
            norm[i] = (row - rmin) / (rmax - rmin)

    return norm


def sort_by_impact(matrix, labels):
    """
    Sort tasks by overall impact (max drop)
    """
    impact = matrix.max(axis=1)
    order = np.argsort(-impact)

    return matrix[order], [labels[i] for i in order], order


def build_delta_matrix(matrix):
    return np.diff(matrix, axis=1)


# -----------------------
# Plotting
# -----------------------

def plot_main_heatmap(matrix, save_path):
    norm_matrix = row_normalize(matrix)

    # SORTING (important improvement!)
    sorted_matrix, sorted_labels, order = sort_by_impact(
        matrix,
        [TASK_LABELS[t] for t in TASKS],
    )
    norm_matrix = norm_matrix[order]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))

    im = ax.imshow(norm_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(TOPK_LABELS)))
    ax.set_xticklabels(TOPK_LABELS, fontsize=12)

    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels, fontsize=12)

    ax.set_xlabel("Ablation setting (k zeroed)", fontsize=13)
    ax.set_ylabel("Task category", fontsize=13)
    ax.set_title(
        "OLMo-1B: Performance Drop under Top-k Superweight Ablation\n"
        "(color shows relative severity within each task)",
        fontsize=13,
    )

    # Values
    for i in range(sorted_matrix.shape[0]):
        for j in range(sorted_matrix.shape[1]):
            val = sorted_matrix[i, j]
            brightness = norm_matrix[i, j]
            color = "white" if brightness > 0.55 else "black"

            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold")

    # Mark largest jump
    delta = build_delta_matrix(sorted_matrix)
    for i, row in enumerate(delta):
        if row.max() > 1e-6:
            j = int(np.argmax(row)) + 1
            ax.text(j, i - 0.35, "★", ha="center",
                    fontsize=14, color="navy", fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Normalized drop within task", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=240)
    plt.close()
    print(f"Saved: {save_path}")


def plot_delta_heatmap(matrix, save_path):
    delta = build_delta_matrix(matrix)

    fig, ax = plt.subplots(figsize=(10.5, 4.2))

    vmax = np.abs(delta).max()
    if vmax < 1e-9:
        vmax = 1.0

    im = ax.imshow(
        delta,
        cmap="RdBu_r",
        norm=mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
        aspect="auto",
    )

    labels = [TASK_LABELS[t] for t in TASKS]

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=12)

    ax.set_xticks(range(delta.shape[1]))
    ax.set_xticklabels(
        [f"{TOPK_LABELS[i]}→{TOPK_LABELS[i+1]}" for i in range(len(TOPK_LABELS)-1)],
        rotation=20,
        ha="right",
    )

    ax.set_title("Incremental performance change between k steps", fontsize=13)

    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            val = delta[i, j]
            color = "white" if abs(val) > 0.5 * vmax else "black"
            ax.text(j, i, f"{val:+.3f}", ha="center",
                    va="center", fontsize=10, color=color)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=240)
    plt.close()
    print(f"Saved: {save_path}")


# -----------------------
# MAIN
# -----------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    matrix = build_drop_matrix()

    plot_main_heatmap(
        matrix,
        os.path.join(OUT_DIR, "olmo_topk_heatmap.png"),
    )

    plot_delta_heatmap(
        matrix,
        os.path.join(OUT_DIR, "olmo_topk_heatmap_delta.png"),
    )

    print("Done.")


if __name__ == "__main__":
    main()