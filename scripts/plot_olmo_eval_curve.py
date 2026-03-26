import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metric(path: str, metric_key: str) -> float:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["results"]["hellaswag"][metric_key]


def main() -> None:
    os.makedirs("outputs/eval", exist_ok=True)

    x = [0, 1, 2, 3, 5, 10]

    acc = [
        load_metric("outputs/eval/olmo-1b_hellaswag_baseline.json", "acc,none"),
        load_metric("outputs/eval/olmo-1b_hellaswag_top1_ablated.json", "acc,none"),
        load_metric("outputs/eval/olmo-1b_hellaswag_top2_ablated.json", "acc,none"),
        load_metric("outputs/eval/olmo-1b_hellaswag_top3_ablated.json", "acc,none"),
        load_metric("outputs/eval/olmo-1b_hellaswag_top5_ablated.json", "acc,none"),
        load_metric("outputs/eval/olmo-1b_hellaswag_top10_ablated.json", "acc,none"),
    ]

    acc_norm = [
        load_metric("outputs/eval/olmo-1b_hellaswag_baseline.json", "acc_norm,none"),
        load_metric("outputs/eval/olmo-1b_hellaswag_top1_ablated.json", "acc_norm,none"),
        load_metric("outputs/eval/olmo-1b_hellaswag_top2_ablated.json", "acc_norm,none"),
        load_metric("outputs/eval/olmo-1b_hellaswag_top3_ablated.json", "acc_norm,none"),
        load_metric("outputs/eval/olmo-1b_hellaswag_top5_ablated.json", "acc_norm,none"),
        load_metric("outputs/eval/olmo-1b_hellaswag_top10_ablated.json", "acc_norm,none"),
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(x, acc_norm, marker="o", label="acc_norm")

    for xi, yi in zip(x, acc_norm):
        plt.text(xi, yi, f"{yi:.3f}", ha="center", va="top")

    plt.xticks(x)
    plt.xlabel("Number of removed weights (k)")
    plt.ylabel("HellaSwag score")
    plt.title("OLMo-1B: HellaSwag vs. Number of Ablated Superweights")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = "outputs/eval/olmo_ablation_curve.png"
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved plot to: {save_path}")


if __name__ == "__main__":
    main()