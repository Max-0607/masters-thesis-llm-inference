import argparse
import csv
import json
from pathlib import Path

import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_metric_key(task: str, cfg: dict) -> str:
    overrides = cfg.get("metrics", {}).get("overrides", {})
    default_metric = cfg.get("metrics", {}).get("default", "acc,none")
    return overrides.get(task, default_metric)


def flatten_task_groups(task_groups: dict) -> dict:
    task_to_group = {}
    for group, tasks in task_groups.items():
        for task in tasks:
            task_to_group[task] = group
    return task_to_group


def read_score(result_json: dict, task_name: str, metric_key: str) -> float:
    results = result_json.get("results", {})
    if task_name not in results:
        raise KeyError(f"Task '{task_name}' not found in results JSON.")

    task_result = results[task_name]
    if metric_key not in task_result:
        available = ", ".join(task_result.keys())
        raise KeyError(
            f"Metric '{metric_key}' not found for task '{task_name}'. Available: {available}"
        )

    return float(task_result[metric_key])


def discover_available_ablation_prefixes(input_dir: Path, tasks: list[str]) -> list[str]:
    prefixes = set()
    for task in tasks:
        for path in input_dir.glob(f"*__{task}.json"):
            prefix = path.name.split("__")[0]
            if prefix != "baseline_model" and prefix.endswith("_ablated_model"):
                prefixes.add(prefix)
    return sorted(prefixes)


def prefix_to_superweight_id(prefix: str) -> str:
    return prefix.replace("_ablated_model", "")


def build_superweight_lookup(superweights_cfg: list[dict]) -> dict:
    return {sw["id"]: sw for sw in superweights_cfg}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/task_heatmaps/phi3_tasks.yaml",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="outputs/task_heatmaps/phi3",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/task_heatmaps/phi3_heatmap.csv",
    )
    parser.add_argument(
        "--baseline-prefix",
        type=str,
        default="baseline_model",
        help="Expected file prefix: baseline_model__TASK.json",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    task_to_group = flatten_task_groups(cfg["task_groups"])
    tasks = list(task_to_group.keys())
    model_name = cfg["model"]["name"]

    superweight_lookup = build_superweight_lookup(cfg["superweights"])
    available_ablation_prefixes = discover_available_ablation_prefixes(input_dir, tasks)

    if not available_ablation_prefixes:
        raise FileNotFoundError(
            f"No ablated result files found in {input_dir}. "
            f"Expected files like sw1_ablated_model__TASK.json"
        )

    rows = []

    for task, group in task_to_group.items():
        metric_key = get_metric_key(task, cfg)

        baseline_path = input_dir / f"{args.baseline_prefix}__{task}.json"
        if not baseline_path.exists():
            raise FileNotFoundError(f"Missing baseline result file: {baseline_path}")

        baseline_json = load_json(baseline_path)
        baseline_score = read_score(baseline_json, task, metric_key)

        for ablated_prefix in available_ablation_prefixes:
            ablated_path = input_dir / f"{ablated_prefix}__{task}.json"
            if not ablated_path.exists():
                # Skip incomplete ablation runs instead of failing
                continue

            superweight_id = prefix_to_superweight_id(ablated_prefix)
            if superweight_id not in superweight_lookup:
                print(
                    f"Warning: {superweight_id} found in files but not in config. Skipping."
                )
                continue

            sw = superweight_lookup[superweight_id]
            ablated_json = load_json(ablated_path)
            ablated_score = read_score(ablated_json, task, metric_key)

            delta_abs = baseline_score - ablated_score
            delta_rel = delta_abs / baseline_score if baseline_score != 0 else 0.0

            rows.append(
                {
                    "model": model_name,
                    "task": task,
                    "category": group,
                    "metric": metric_key,
                    "superweight_id": superweight_id,
                    "ablated_model_prefix": ablated_prefix,
                    "layer": sw["layer"],
                    "row": sw["row"],
                    "col": sw["col"],
                    "baseline_score": baseline_score,
                    "ablated_score": ablated_score,
                    "delta_abs": delta_abs,
                    "delta_rel": delta_rel,
                }
            )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "task",
                "category",
                "metric",
                "superweight_id",
                "ablated_model_prefix",
                "layer",
                "row",
                "col",
                "baseline_score",
                "ablated_score",
                "delta_abs",
                "delta_rel",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV to {output_csv}")
    print(f"Included ablations: {', '.join(available_ablation_prefixes)}")


if __name__ == "__main__":
    main()