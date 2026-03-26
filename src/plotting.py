import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_category_layer_maxima(category_to_summaries, key, title, save_path):
    """
    category_to_summaries:
        {
            "causal": [...],
            "coding": [...],
            ...
        }

    key:
        "max_abs_value"
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 5))

    for category, summaries in category_to_summaries.items():
        layers = [x["layer"] for x in summaries]
        values = [x[key] for x in summaries]
        plt.plot(layers, values, marker="o", label=category)

    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel(key)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved plot to: {save_path}")