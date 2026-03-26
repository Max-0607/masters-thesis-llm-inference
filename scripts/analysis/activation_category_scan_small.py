import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


PROMPT_SETS = {
    "reasoning": [
        "A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the ball. How much does the ball cost?"
    ],
    "math": [
        "27 * 14 = ?"
    ],
    "causal": [
        "Why does dropping a glass on the floor make it shatter?"
    ],
    "knowledge": [
        "What is the capital of France?"
    ],
    "coding": [
        "Write a Python function that returns the factorial of a number."
    ],
}


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Could not find transformer layers.")


def get_down_proj_module(layer):
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
        return layer.mlp.down_proj
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "fc2"):
        return layer.mlp.fc2
    raise ValueError("Could not find down projection module.")


def get_model_device(model):
    return next(model.parameters()).device


def make_store(layer_indices):
    return {l: {"in_max": None, "out_max": None} for l in layer_indices}


def collect_for_layer(layer_idx, store):
    def hook_fn(module, inputs, outputs):
        x = inputs[0].detach().float()
        y = outputs.detach().float()

        x_abs = x.abs().reshape(-1, x.shape[-1])
        y_abs = y.abs().reshape(-1, y.shape[-1])

        x_max = x_abs.max(dim=0).values
        y_max = y_abs.max(dim=0).values

        if store[layer_idx]["in_max"] is None:
            store[layer_idx]["in_max"] = x_max
            store[layer_idx]["out_max"] = y_max
        else:
            store[layer_idx]["in_max"] = torch.maximum(store[layer_idx]["in_max"], x_max)
            store[layer_idx]["out_max"] = torch.maximum(store[layer_idx]["out_max"], y_max)

    return hook_fn


def analyze_category_small(model, tok, texts, layer_indices, max_len=32):
    layers = get_transformer_layers(model)
    store = make_store(layer_indices)
    hooks = []

    for l in layer_indices:
        mod = get_down_proj_module(layers[l])
        hooks.append(mod.register_forward_hook(collect_for_layer(l, store)))

    device = get_model_device(model)

    try:
        with torch.no_grad():
            for t in texts:
                inp = tok(
                    t,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_len,
                )
                inp = {k: v.to(device) for k, v in inp.items()}

                _ = model(
                    **inp,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
    finally:
        for h in hooks:
            h.remove()

    results = {}

    for l in layer_indices:
        if store[l]["in_max"] is None or store[l]["out_max"] is None:
            raise RuntimeError(f"No activations collected for layer {l}")

        in_peak_val, in_peak_ch = store[l]["in_max"].max(dim=0)
        out_peak_val, out_peak_ch = store[l]["out_max"].max(dim=0)

        W = get_down_proj_module(layers[l]).weight.detach().float()

        results[l] = {
            "in_peak_val": float(in_peak_val.cpu()),
            "in_peak_ch": int(in_peak_ch.cpu()),
            "out_peak_val": float(out_peak_val.cpu()),
            "out_peak_ch": int(out_peak_ch.cpu()),
            "candidate_coord": (int(out_peak_ch.cpu()), int(in_peak_ch.cpu())),
            "candidate_abs_weight": float(W[int(out_peak_ch.cpu()), int(in_peak_ch.cpu())].abs().cpu()),
        }

    best_layer = max(layer_indices, key=lambda l: results[l]["out_peak_val"])

    return {
        "best_layer": best_layer,
        "layer_results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--category", type=str, required=True, choices=list(PROMPT_SETS.keys()))
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 4])
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    result = analyze_category_small(
        model,
        tok,
        PROMPT_SETS[args.category],
        layer_indices=args.layers,
        max_len=args.max_len,
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
