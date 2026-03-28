import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = "outputs/eval/olmo_sw1_scaling"
OUT_DIR = "outputs/plots"

TASKS = [
    "hellaswag",
    "lambada_openai",
    "mgsm_direct_en",
    "sciq",
]

SCALE_LABELS = [
    "0p5",
    "0p8",
    "1p0",
    "1p1",
    "1p2",
    "1p5",
    "2p0",
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


def build_delta_vs_baseline_matrix() -> np.ndarray:
    matrix = []

    for task in TASKS:
        metric = TASK_METRIC[task]
        baseline_path = os.path.join(BASE_DIR, f"baseline_{task}.json")
        baseline_score = load_metric(baseline_path, task, metric)

        row = []
        for scale_label in SCALE_LABELS:
            scaled_path = os.path.join(BASE_DIR, f"scale_{scale_label}_{task}.json")
            scaled_score = load_metric(scaled_path, task, metric)
            delta = scaled_score - baseline_score
            row.append(delta)

        matrix.append(row)

    return np.array(matrix, dtype=float)


def build_absolute_matrix() -> np.ndarray:
    matrix = []

    for task in TASKS:
        metric = TASK_METRIC[task]
        baseline_path = os.path.join(BASE_DIR, f"baseline_{task}.json")
        baseline_score = load_metric(baseline_path, task, metric)

        row = [baseline_score]
        for scale_label in SCALE_LABELS:
            scaled_path = os.path.join(BASE_DIR, f"scale_{scale_label}_{task}.json")
            scaled_score = load_metric(scaled_path, task, metric)
            row.append(scaled_score)

        matrix.append(row)

    return np.array(matrix, dtype=float)


def build_step_change_matrix(delta_matrix: np.ndarray) -> np.ndarray:
    return np.diff(delta_matrix, axis=1)


def save_csv(matrix: np.ndarray, csv_path: str, headers) -> None:
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("task," + ",".join(headers) + "\n")
        for task, row in zip(TASKS, matrix):
            values = ",".join(f"{x:.6f}" for x in row)
            f.write(f"{task},{values}\n")


def plot_scaling_delta_heatmap(matrix: np.ndarray, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))

    abs_max = np.abs(matrix).max()
    if abs_max < 1e-9:
        abs_max = 1.0

    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm)

    xlabels = [f"x{s.replace('p', '.')}" for s in SCALE_LABELS]
    ax.set_xticks(range(len(SCALE_LABELS)))
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels(TASKS, fontsize=11)
    ax.set_xlabel("SW1 scaling factor", fontsize=12)
    ax.set_ylabel("Task", fontsize=12)
    ax.set_title(
        "OLMo-1B: Task-specific score change under SW1 scaling\n"
        "(relative to baseline)",
        fontsize=12,
    )

    for i in range(matrix.shape[0]):
        best_j = int(np.argmax(matrix[i]))
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            txt_color = "white" if abs(val) > 0.5 * abs_max else "black"
            label = f"{val:+.3f}"
            if j == best_j and val > 0:
                label += " ★"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=10,
                color=txt_color,
                fontweight="bold",
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Score change vs baseline", fontsize=10)

    ax.plot([], [], marker="*", color="black", linestyle="None",
            markersize=9, label="Best positive scale per task")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def plot_absolute_heatmap(matrix: np.ndarray, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")

    headers = ["baseline"] + [f"x{s.replace('p', '.')}" for s in SCALE_LABELS]
    ax.set_xticks(range(len(headers)))
    ax.set_xticklabels(headers, fontsize=11)
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels(TASKS, fontsize=11)
    ax.set_xlabel("SW1 scaling factor", fontsize=12)
    ax.set_ylabel("Task", fontsize=12)
    ax.set_title("OLMo-1B: Absolute task scores under SW1 scaling", fontsize=12)

    for i in range(matrix.shape[0]):
        row_min = matrix[i].min()
        row_max = matrix[i].max()
        threshold = row_min + 0.6 * (row_max - row_min + 1e-12)

        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            txt_color = "white" if val >= threshold else "black"
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=10,
                color=txt_color,
                fontweight="bold",
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Task score", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def plot_step_change_heatmap(step_matrix: np.ndarray, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))

    abs_max = np.abs(step_matrix).max()
    if abs_max < 1e-9:
        abs_max = 1.0

    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
    im = ax.imshow(step_matrix, aspect="auto", cmap="RdBu_r", norm=norm)

    step_labels = [
        "x0.5→x0.8",
        "x0.8→x1.0",
        "x1.0→x1.1",
        "x1.1→x1.2",
        "x1.2→x1.5",
        "x1.5→x2.0",
    ]

    ax.set_xticks(range(len(step_labels)))
    ax.set_xticklabels(step_labels, fontsize=10, rotation=20, ha="right")
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels(TASKS, fontsize=11)
    ax.set_xlabel("Step between scaling factors", fontsize=12)
    ax.set_ylabel("Task", fontsize=12)
    ax.set_title(
        "OLMo-1B: Incremental change between consecutive SW1 scales\n"
        "(positive = improvement increases, negative = improvement decreases)",
        fontsize=12,
    )

    for i in range(step_matrix.shape[0]):
        for j in range(step_matrix.shape[1]):
            val = step_matrix[i, j]
            txt_color = "white" if abs(val) > 0.5 * abs_max else "black"
            ax.text(
                j,
                i,
                f"{val:+.3f}",
                ha="center",
                va="center",
                fontsize=10,
                color=txt_color,
                fontweight="bold",
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Δ(score change)", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    delta_matrix = build_delta_vs_baseline_matrix()
    abs_matrix = build_absolute_matrix()
    step_matrix = build_step_change_matrix(delta_matrix)

    plot_scaling_delta_heatmap(
        delta_matrix,
        save_path=os.path.join(OUT_DIR, "olmo_sw1_scaling_delta_heatmap.png"),
    )

    plot_absolute_heatmap(
        abs_matrix,
        save_path=os.path.join(OUT_DIR, "olmo_sw1_scaling_absolute_heatmap.png"),
    )

    plot_step_change_heatmap(
        step_matrix,
        save_path=os.path.join(OUT_DIR, "olmo_sw1_scaling_step_heatmap.png"),
    )

    save_csv(
        delta_matrix,
        os.path.join(OUT_DIR, "olmo_sw1_scaling_delta.csv"),
        SCALE_LABELS,
    )

    save_csv(
        abs_matrix,
        os.path.join(OUT_DIR, "olmo_sw1_scaling_absolute.csv"),
        ["baseline"] + SCALE_LABELS,
    )

    save_csv(
        step_matrix,
        os.path.join(OUT_DIR, "olmo_sw1_scaling_step.csv"),
        [
            "0p5_to_0p8",
            "0p8_to_1p0",
            "1p0_to_1p1",
            "1p1_to_1p2",
            "1p2_to_1p5",
            "1p5_to_2p0",
        ],
    )

    print("Done.")


if __name__ == "__main__":
    main()