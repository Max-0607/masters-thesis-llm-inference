from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.hooks import get_nested_attr
from src.prompts import CATEGORY_PROMPTS


MODEL_SPECS = {
    "mistral-7b": {
        "hf_name": "mistralai/Mistral-7B-v0.1",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [(1, 2070, 7310)],
    },
    "llama-7b": {
        "hf_name": "huggyllama/llama-7b",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [(2, 3968, 7003)],
    },
    "llama-13b": {
        "hf_name": "huggyllama/llama-13b",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [(2, 2231, 2278), (2, 2231, 6939)],
    },
    "llama-30b": {
        "hf_name": "huggyllama/llama-30b",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [
            (3, 5633, 12817),
            (3, 5633, 17439),
            (10, 5633, 14386),
        ],
    },
    "llama3-8b": {
        "hf_name": "meta-llama/Meta-Llama-3-8B",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [
            (1, 788, 2427),
            (1, 1384, 2427),
            (1, 4062, 2427),
        ],
    },
    "olmo1b": {
        "hf_name": "allenai/OLMo-1B-0724-hf",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [
            (1, 1764, 1710),
            (1, 1764, 8041),
        ],
    },
    "olmo7b": {
        "hf_name": "allenai/OLMo-7B-0724-hf",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [
            (1, 269, 7467),
            (2, 269, 8275),
            (7, 269, 453),
            (24, 269, 2300),
        ],
    },
    "phi3-mini": {
        "hf_name": "microsoft/Phi-3-mini-4k-instruct",
        "layer_path": "model.layers",
        "down_proj_path": "mlp.down_proj",
        "superweights": [
            (2, 525, 808),
            (2, 1693, 808),
            (2, 1113, 808),
            (4, 525, 2723),
            (4, 1113, 2723),
            (4, 1693, 2723),
        ],
    },
}


def to_sw_dicts(tuples):
    return [
        {"layer": layer, "row": row, "col": col}
        for layer, row, col in tuples
    ]


def get_device(model):
    return next(model.parameters()).device


def load_model(model_path_or_id):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path_or_id,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path_or_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    model.eval()
    return model, tokenizer


def collect_down_proj_outputs(
    model,
    tokenizer,
    prompts,
    layer_idx,
    layer_path,
    down_proj_path,
    max_length,
):
    layers = get_nested_attr(model, layer_path)
    module = get_nested_attr(layers[layer_idx], down_proj_path)

    outputs = []

    def hook_fn(module, inputs, output):
        outputs.append(output.detach().float().cpu())

    handle = module.register_forward_hook(hook_fn)
    device = get_device(model)

    try:
        with torch.no_grad():
            for prompt in prompts:
                batch = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )

                batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                }

                model(**batch, use_cache=False)
    finally:
        handle.remove()

    return torch.cat(outputs, dim=1)


def compute_metrics(y, channel, top_k):
    flat = y.reshape(-1, y.shape[-1])

    abs_y = flat.abs()
    energy_y = flat.pow(2)

    target_abs = abs_y[:, channel]
    total_abs = abs_y.sum(dim=-1)

    target_energy = energy_y[:, channel]
    total_energy = energy_y.sum(dim=-1)

    dominance_abs = target_abs / (total_abs + 1e-12)
    dominance_energy = target_energy / (total_energy + 1e-12)

    p_abs = abs_y / (total_abs.unsqueeze(-1) + 1e-12)
    entropy_abs = -(p_abs * torch.log(p_abs + 1e-12)).sum(dim=-1)

    mean_abs_by_channel = abs_y.mean(dim=0)
    mean_energy_by_channel = energy_y.mean(dim=0)

    k = min(top_k, mean_abs_by_channel.shape[0])

    top_abs_vals, top_abs_idx = torch.topk(mean_abs_by_channel, k=k)
    top_energy_vals, top_energy_idx = torch.topk(mean_energy_by_channel, k=k)

    return {
        "dominance_abs_mean": dominance_abs.mean().item(),
        "dominance_abs_std": dominance_abs.std().item(),
        "dominance_energy_mean": dominance_energy.mean().item(),
        "dominance_energy_std": dominance_energy.std().item(),
        "entropy_abs_mean": entropy_abs.mean().item(),
        "entropy_abs_std": entropy_abs.std().item(),
        "channel_abs_mean": mean_abs_by_channel[channel].item(),
        "channel_energy_mean": mean_energy_by_channel[channel].item(),
        "top_abs_channels": [
            {"channel": int(i), "mean_abs": float(v)}
            for i, v in zip(top_abs_idx, top_abs_vals)
        ],
        "top_energy_channels": [
            {"channel": int(i), "mean_energy": float(v)}
            for i, v in zip(top_energy_idx, top_energy_vals)
        ],
    }


def analyze_one_superweight(
    original_model,
    original_tokenizer,
    dropout_model,
    dropout_tokenizer,
    prompts,
    sw,
    layer_path,
    down_proj_path,
    max_length,
    top_k,
):
    layer = sw["layer"]
    row = sw["row"]

    result = {
        "superweight": sw,
        "original": {},
        "sw_dropout": {},
        "comparison": {},
    }

    print(
        f"\nCollecting outputs for layer={layer}, "
        f"row/channel={row}, col={sw['col']}"
    )

    y_original = collect_down_proj_outputs(
        model=original_model,
        tokenizer=original_tokenizer,
        prompts=prompts,
        layer_idx=layer,
        layer_path=layer_path,
        down_proj_path=down_proj_path,
        max_length=max_length,
    )

    y_dropout = collect_down_proj_outputs(
        model=dropout_model,
        tokenizer=dropout_tokenizer,
        prompts=prompts,
        layer_idx=layer,
        layer_path=layer_path,
        down_proj_path=down_proj_path,
        max_length=max_length,
    )

    result["original"] = compute_metrics(
        y=y_original,
        channel=row,
        top_k=top_k,
    )

    result["sw_dropout"] = compute_metrics(
        y=y_dropout,
        channel=row,
        top_k=top_k,
    )

    result["comparison"] = {
        "dominance_abs_delta": (
            result["sw_dropout"]["dominance_abs_mean"]
            - result["original"]["dominance_abs_mean"]
        ),
        "dominance_energy_delta": (
            result["sw_dropout"]["dominance_energy_mean"]
            - result["original"]["dominance_energy_mean"]
        ),
        "entropy_abs_delta": (
            result["sw_dropout"]["entropy_abs_mean"]
            - result["original"]["entropy_abs_mean"]
        ),
        "channel_abs_mean_delta": (
            result["sw_dropout"]["channel_abs_mean"]
            - result["original"]["channel_abs_mean"]
        ),
        "channel_energy_mean_delta": (
            result["sw_dropout"]["channel_energy_mean"]
            - result["original"]["channel_energy_mean"]
        ),
    }

    del y_original
    del y_dropout

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-key",
        choices=list(MODEL_SPECS.keys()),
        required=True,
    )

    parser.add_argument(
        "--original-model",
        default=None,
        help="Optional model path/id. If omitted, MODEL_SPECS hf_name is used.",
    )

    parser.add_argument(
        "--dropout-model",
        required=True,
        help="Path to saved SW-dropout / redistribution model.",
    )

    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=20)

    parser.add_argument(
        "--output-json",
        required=True,
    )

    args = parser.parse_args()

    spec = MODEL_SPECS[args.model_key]

    original_model_path = (
        args.original_model
        if args.original_model is not None
        else spec["hf_name"]
    )

    layer_path = spec["layer_path"]
    down_proj_path = spec["down_proj_path"]
    superweights = to_sw_dicts(spec["superweights"])

    prompts = []
    for prompt_list in CATEGORY_PROMPTS.values():
        prompts.extend(prompt_list)

    print(f"Model key: {args.model_key}")
    print(f"Original model: {original_model_path}")
    print(f"Dropout model: {args.dropout_model}")
    print(f"Superweights: {superweights}")

    print("\nLoading original model...")
    original_model, original_tokenizer = load_model(original_model_path)

    print("\nLoading dropout model...")
    dropout_model, dropout_tokenizer = load_model(args.dropout_model)

    all_results = {
        "model_key": args.model_key,
        "original_model": original_model_path,
        "dropout_model": args.dropout_model,
        "layer_path": layer_path,
        "down_proj_path": down_proj_path,
        "max_length": args.max_length,
        "top_k": args.top_k,
        "superweights": superweights,
        "analyses": [],
    }

    for sw in superweights:
        result = analyze_one_superweight(
            original_model=original_model,
            original_tokenizer=original_tokenizer,
            dropout_model=dropout_model,
            dropout_tokenizer=dropout_tokenizer,
            prompts=prompts,
            sw=sw,
            layer_path=layer_path,
            down_proj_path=down_proj_path,
            max_length=args.max_length,
            top_k=args.top_k,
        )

        all_results["analyses"].append(result)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== Results saved ===")
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()