import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dtype(dtype_name: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def save_model_and_tokenizer(model, tokenizer, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def ablate_single_superweight(model, layer: int, row: int, col: int):
    weight = model.model.layers[layer].mlp.down_proj.weight
    original_value = weight[row, col].item()
    with torch.no_grad():
        weight[row, col] = 0.0
    return original_value


def write_metadata(save_dir: Path, metadata: dict):
    with open(save_dir / "ablation_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/task_heatmaps/phi3_tasks.yaml")
    parser.add_argument("--output-root", type=str, default="outputs/task_heatmaps/phi3")
    parser.add_argument("--save-baseline", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["model"]["hf_path"]
    dtype = get_dtype(cfg["model"].get("dtype", "float16"))
    superweights = cfg["superweights"]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if args.save_baseline:
        print(f"Loading baseline model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=None,
        )
        model.eval()

        baseline_dir = output_root / "baseline_model"
        print(f"Saving baseline model to {baseline_dir}")
        save_model_and_tokenizer(model, tokenizer, baseline_dir)
        write_metadata(
            baseline_dir,
            {
                "type": "baseline",
                "source_model": model_name,
            },
        )
        del model

    for sw in superweights:
        sw_id = sw["id"]
        layer = sw["layer"]
        row = sw["row"]
        col = sw["col"]

        print(f"\nCreating ablated model for {sw_id}: layer={layer}, row={row}, col={col}")

        fresh_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=None,
        )
        fresh_model.eval()

        original_value = ablate_single_superweight(fresh_model, layer, row, col)

        save_dir = output_root / f"{sw_id}_ablated_model"
        print(f"Saving to {save_dir}")
        save_model_and_tokenizer(fresh_model, tokenizer, save_dir)

        write_metadata(
            save_dir,
            {
                "type": "single_superweight_ablation",
                "source_model": model_name,
                "superweight_id": sw_id,
                "layer": layer,
                "row": row,
                "col": col,
                "original_value": original_value,
            },
        )

        del fresh_model

    print("\nDone.")


if __name__ == "__main__":
    main()