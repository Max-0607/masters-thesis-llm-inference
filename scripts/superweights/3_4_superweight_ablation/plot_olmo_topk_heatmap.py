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

TOPK_LABELS = [
    "top1",
    "top2",
    "top3",
    "top4",
    "top5",
    "top10",
]

STEP_LABELS = [
    "Baseline→top1",
    "top1→top2",
    "top2→top3",
    "top3→top4",
    "top4→top5",
    "top5→top10",
]

TASK_METRIC = {
    "hellaswag": "acc_norm,none",
    "lambada_openai": "acc,none",
    "mgsm_direct_en": "exact_match,flexible-extract",
    "sciq": "acc,none",
}


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------

def load_metric(path, task, metric):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["results"][task][metric]


def build_drop_matrix():
    """
    Construct the cumulative performance-drop matrix.

    Each value is calculated relative to the unablated baseline:

        baseline score - top-k score
    """
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

    return np.array(matrix, dtype=float)


def build_incremental_drop_matrix():
    """
    Construct the incremental performance-drop matrix.

    Each value is calculated relative to the immediately preceding
    ablation setting:

        baseline - top1
        top1 - top2
        top2 - top3
        top3 - top4
        top4 - top5
        top5 - top10

    Positive values indicate an additional performance decrease.
    Negative values indicate an improvement relative to the
    preceding ablation setting.
    """
    matrix = []

    for task in TASKS:
        metric = TASK_METRIC[task]

        previous_score = load_metric(
            os.path.join(BASE_DIR, f"baseline_{task}.json"),
            task,
            metric,
        )

        row = []

        for label in TOPK_LABELS:
            current_score = load_metric(
                os.path.join(BASE_DIR, f"{label}_{task}.json"),
                task,
                metric,
            )

            incremental_drop = previous_score - current_score
            row.append(incremental_drop)

            previous_score = current_score

        matrix.append(row)

    return np.array(matrix, dtype=float)


# ------------------------------------------------------------
# Processing
# ------------------------------------------------------------

def row_normalize(matrix):
    """
    Normalize every task independently to the interval [0, 1].
    """
    normalized = np.zeros_like(matrix, dtype=float)

    for i, row in enumerate(matrix):
        row_min = row.min()
        row_max = row.max()

        if abs(row_max - row_min) < 1e-9:
            normalized[i] = 1.0
        else:
            normalized[i] = (
                (row - row_min) / (row_max - row_min)
            )

    return normalized


def sort_by_cumulative_impact(matrix, labels):
    """
    Sort tasks according to their largest cumulative drop.
    """
    impact = matrix.max(axis=1)
    order = np.argsort(-impact)

    sorted_matrix = matrix[order]
    sorted_labels = [labels[i] for i in order]

    return sorted_matrix, sorted_labels, order


def sort_by_incremental_impact(matrix, labels):
    """
    Sort tasks according to their largest absolute incremental change.
    """
    impact = np.abs(matrix).max(axis=1)
    order = np.argsort(-impact)

    sorted_matrix = matrix[order]
    sorted_labels = [labels[i] for i in order]

    return sorted_matrix, sorted_labels, order


def build_delta_matrix(matrix):
    """
    Calculate the change between consecutive cumulative drops.
    """
    return np.diff(matrix, axis=1)


def get_symmetric_limit(matrix):
    """
    Return a non-zero symmetric color limit.
    """
    vmax = float(np.abs(matrix).max())

    if vmax < 1e-9:
        vmax = 1.0

    return vmax


# ------------------------------------------------------------
# Existing figure 1:
# Cumulative performance drop relative to baseline
# ------------------------------------------------------------

def plot_main_heatmap(matrix, save_path):
    normalized_matrix = row_normalize(matrix)

    task_labels = [TASK_LABELS[task] for task in TASKS]

    sorted_matrix, sorted_labels, order = (
        sort_by_cumulative_impact(
            matrix,
            task_labels,
        )
    )

    normalized_matrix = normalized_matrix[order]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))

    image = ax.imshow(
        normalized_matrix,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        aspect="auto",
    )

    ax.set_xticks(range(len(TOPK_LABELS)))
    ax.set_xticklabels(
        TOPK_LABELS,
        fontsize=12,
    )

    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(
        sorted_labels,
        fontsize=12,
    )

    ax.set_xlabel(
        "Ablation setting (k zeroed)",
        fontsize=13,
    )
    ax.set_ylabel(
        "Task category",
        fontsize=13,
    )

    # No title is added so that the exported figure has no heading.

    for i in range(sorted_matrix.shape[0]):
        for j in range(sorted_matrix.shape[1]):
            value = sorted_matrix[i, j]
            brightness = normalized_matrix[i, j]

            text_color = (
                "white"
                if brightness > 0.55
                else "black"
            )

            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=11,
                color=text_color,
                fontweight="bold",
            )

    # Mark the largest additional drop within every task.
    delta = build_delta_matrix(sorted_matrix)

    for i, row in enumerate(delta):
        if row.max() > 1e-6:
            j = int(np.argmax(row)) + 1

            ax.text(
                j,
                i - 0.35,
                "★",
                ha="center",
                va="center",
                fontsize=14,
                color="navy",
                fontweight="bold",
            )

    colorbar = plt.colorbar(
        image,
        ax=ax,
        fraction=0.03,
        pad=0.02,
    )
    colorbar.set_label(
        "Normalized drop within task",
        fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=240,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {save_path}")


# ------------------------------------------------------------
# Existing figure 2:
# Consecutive differences derived from the cumulative matrix
# ------------------------------------------------------------

def plot_delta_heatmap(matrix, save_path):
    delta = build_delta_matrix(matrix)
    vmax = get_symmetric_limit(delta)

    fig, ax = plt.subplots(figsize=(10.5, 4.2))

    image = ax.imshow(
        delta,
        cmap="RdBu_r",
        norm=mcolors.TwoSlopeNorm(
            vmin=-vmax,
            vcenter=0,
            vmax=vmax,
        ),
        aspect="auto",
    )

    labels = [TASK_LABELS[task] for task in TASKS]

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(
        labels,
        fontsize=12,
    )

    transition_labels = [
        f"{TOPK_LABELS[i]}→{TOPK_LABELS[i + 1]}"
        for i in range(len(TOPK_LABELS) - 1)
    ]

    ax.set_xticks(range(delta.shape[1]))
    ax.set_xticklabels(
        transition_labels,
        rotation=20,
        ha="right",
        fontsize=10,
    )

    # No title is added so that the exported figure has no heading.

    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            value = delta[i, j]

            text_color = (
                "white"
                if abs(value) > 0.5 * vmax
                else "black"
            )

            ax.text(
                j,
                i,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
            )

    colorbar = plt.colorbar(
        image,
        ax=ax,
        fraction=0.03,
        pad=0.02,
    )
    colorbar.set_label(
        "Change in cumulative performance drop",
        fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=240,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {save_path}")


# ------------------------------------------------------------
# New figure 3:
# Performance drop relative to the preceding ablation step
# ------------------------------------------------------------

def plot_incremental_heatmap(matrix, save_path):
    task_labels = [TASK_LABELS[task] for task in TASKS]

    sorted_matrix, sorted_labels, _ = (
        sort_by_incremental_impact(
            matrix,
            task_labels,
        )
    )

    vmax = get_symmetric_limit(sorted_matrix)

    color_norm = mcolors.TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0,
        vmax=vmax,
    )

    fig, ax = plt.subplots(figsize=(12, 5.2))

    image = ax.imshow(
        sorted_matrix,
        cmap="RdBu_r",
        norm=color_norm,
        aspect="auto",
    )

    ax.set_xticks(range(len(STEP_LABELS)))
    ax.set_xticklabels(
        STEP_LABELS,
        rotation=20,
        ha="right",
        fontsize=10,
    )

    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(
        sorted_labels,
        fontsize=12,
    )

    ax.set_xlabel(
        "Additional superweights removed",
        fontsize=13,
    )
    ax.set_ylabel(
        "Task category",
        fontsize=13,
    )

    # No title is added so that the exported figure has no heading.

    for i in range(sorted_matrix.shape[0]):
        for j in range(sorted_matrix.shape[1]):
            value = sorted_matrix[i, j]
            relative_strength = abs(value) / vmax

            text_color = (
                "white"
                if relative_strength > 0.5
                else "black"
            )

            ax.text(
                j,
                i,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=11,
                color=text_color,
                fontweight="bold",
            )

    colorbar = plt.colorbar(
        image,
        ax=ax,
        fraction=0.03,
        pad=0.02,
    )
    colorbar.set_label(
        "Performance drop relative to preceding step",
        fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=240,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {save_path}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Matrix for the existing baseline-relative figures.
    cumulative_matrix = build_drop_matrix()

    # Separate matrix for the new preceding-step figure.
    incremental_matrix = build_incremental_drop_matrix()

    # Existing heatmap without a title.
    plot_main_heatmap(
        cumulative_matrix,
        os.path.join(
            OUT_DIR,
            "olmo_topk_heatmap_no_title.png",
        ),
    )

    # Existing delta heatmap without a title.
    plot_delta_heatmap(
        cumulative_matrix,
        os.path.join(
            OUT_DIR,
            "olmo_topk_heatmap_delta_no_title.png",
        ),
    )

    # New heatmap relative to the immediately preceding step.
    plot_incremental_heatmap(
        incremental_matrix,
        os.path.join(
            OUT_DIR,
            "olmo_topk_heatmap_incremental_no_title.png",
        ),
    )

    print("Done.")


if __name__ == "__main__":
    main()