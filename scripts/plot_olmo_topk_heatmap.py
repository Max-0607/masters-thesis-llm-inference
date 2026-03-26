import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

BASE_DIR = "outputs/eval/olmo_task_compare"
OUT_DIR = "outputs/plots"

TASKS = [
    "hellaswag",
    "lambada_openai",
    "mgsm_direct_en",
    "sciq",
]

TOPK_LABELS = [
    "top1",
    "top2",
    "top3",
    "top4",
    "top5",
    "top10",
]

TASK_METRIC = {
    "hellaswag": "acc_norm,none",
    "lambada_openai": "acc,none",
    "mgsm_direct_en": "exact_match,flexible-extract",
    "sciq": "acc,none",
}


def load_metric(path: str, task: str, metric: str) -> float:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["results"][task][metric]


def build_drop_matrix():
    matrix = []
    for task in TASKS:
        metric = TASK_METRIC[task]
        baseline_path = os.path.join(BASE_DIR, f"baseline_{task}.json")
        baseline_score = load_metric(baseline_path, task, metric)
        row = []
        for label in TOPK_LABELS:
            ablated_path = os.path.join(BASE_DIR, f"{label}_{task}.json")
            ablated_score = load_metric(ablated_path, task, metric)
            drop = baseline_score - ablated_score
            row.append(drop)
        matrix.append(row)
    return np.array(matrix, dtype=float)


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    """
    Normalize each row independently to [0, 1] so that within each task
    the color reflects where the biggest *change* happens across k values.
    Rows where all values are identical get mapped to 1.0 (full impact at top2).
    """
    norm = np.zeros_like(matrix)
    for i, row in enumerate(matrix):
        row_min = row.min()
        row_max = row.max()
        if row_max - row_min < 1e-9:
            # Flat row: everything is equally impacted -> all max
            norm[i] = np.ones_like(row)
        else:
            norm[i] = (row - row_min) / (row_max - row_min)
    return norm


def build_delta_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the incremental change between consecutive k values.
    Shape: (n_tasks, n_topk - 1)
    delta[i, j] = matrix[i, j+1] - matrix[i, j]
    Positive = drop got larger (bad), negative = drop got smaller (recovered).
    """
    return np.diff(matrix, axis=1)


def save_csv(matrix: np.ndarray, csv_path: str) -> None:
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("task," + ",".join(TOPK_LABELS) + "\n")
        for task, row in zip(TASKS, matrix):
            values = ",".join(f"{x:.6f}" for x in row)
            f.write(f"{task},{values}\n")


def plot_heatmap_rownorm(matrix: np.ndarray, norm_matrix: np.ndarray, save_path: str) -> None:
    """
    Main heatmap with row-wise normalization for color, but raw drop values as annotations.
    A thin vertical marker highlights where the largest single-step jump occurs per row.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    cmap = plt.get_cmap("YlOrRd")  # white=no change, dark red=full impact
    im = ax.imshow(norm_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(TOPK_LABELS)))
    ax.set_xticklabels(TOPK_LABELS, fontsize=11)
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels(TASKS, fontsize=11)
    ax.set_xlabel("Ablation setting (k zeroed)", fontsize=12)
    ax.set_ylabel("Task", fontsize=12)
    ax.set_title("OLMo-1B: Performance Drop – row-normalized per task\n(color shows relative severity within each task)", fontsize=12)

    # Annotate with raw drop values
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            raw = matrix[i, j]
            brightness = norm_matrix[i, j]
            txt_color = "white" if brightness > 0.55 else "black"
            ax.text(j, i, f"{raw:.3f}", ha="center", va="center",
                    fontsize=10, color=txt_color, fontweight="bold")

    # Mark the column where the biggest incremental jump happens per row (star)
    delta = build_delta_matrix(matrix)
    for i, row_delta in enumerate(delta):
        if row_delta.max() > 1e-6:          # only if there is any change at all
            jump_col = int(np.argmax(row_delta)) + 1  # +1 because diff shifts index
            ax.annotate("★", xy=(jump_col, i), xytext=(jump_col, i - 0.38),
                        ha="center", va="center", fontsize=13, color="navy",
                        fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Normalized drop within task (0 = min, 1 = max)", fontsize=10)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["min impact", "mid", "max impact"])

    # Legend for star
    ax.plot([], [], marker="*", color="navy", linestyle="None",
            markersize=10, label="Largest single-step jump")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_delta_heatmap(matrix: np.ndarray, save_path: str) -> None:
    """
    Secondary plot: shows the incremental change Δ between consecutive k values.
    Diverging colormap: red = big additional drop, blue = slight recovery.
    """
    delta = build_delta_matrix(matrix)
    delta_labels = [f"{TOPK_LABELS[j]}→{TOPK_LABELS[j+1]}" for j in range(len(TOPK_LABELS) - 1)]

    abs_max = np.abs(delta).max()
    if abs_max < 1e-9:
        abs_max = 1.0  # avoid degenerate colormap

    fig, ax = plt.subplots(figsize=(10, 4))
    cmap = plt.get_cmap("RdBu_r")   # red = more drop, blue = less
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    im = ax.imshow(delta, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(len(delta_labels)))
    ax.set_xticklabels(delta_labels, fontsize=10, rotation=20, ha="right")
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels(TASKS, fontsize=11)
    ax.set_xlabel("Step between ablation settings", fontsize=12)
    ax.set_ylabel("Task", fontsize=12)
    ax.set_title("OLMo-1B: Incremental Performance Drop Δ between consecutive k values\n"
                 "(red = drop increases, blue = drop decreases / recovery)", fontsize=12)

    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            val = delta[i, j]
            # Choose text color based on cell brightness
            normed = (val + abs_max) / (2 * abs_max)
            txt_color = "white" if (normed < 0.25 or normed > 0.75) else "black"
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    fontsize=10, color=txt_color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Δ drop (positive = bigger drop)", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    matrix = build_drop_matrix()
    norm_matrix = row_normalize(matrix)

    # Plot 1: Row-normalized heatmap (main)
    plot_heatmap_rownorm(
        matrix, norm_matrix,
        save_path=os.path.join(OUT_DIR, "olmo_topk_task_heatmap_rownorm.png"),
    )

    # Plot 2: Incremental delta heatmap (secondary)
    plot_delta_heatmap(
        matrix,
        save_path=os.path.join(OUT_DIR, "olmo_topk_task_heatmap_delta.png"),
    )

    # CSV of raw drops
    save_csv(matrix, os.path.join(OUT_DIR, "olmo_topk_task_heatmap.csv"))
    print("Done.")


if __name__ == "__main__":
    main()