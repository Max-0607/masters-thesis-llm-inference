import json
from pathlib import Path

import torch


SUPERWEIGHTS_OLMO1B = [
    (1, 1764, 1710),
    (1, 1764, 8041),
]


def read_fp16_superweights(model, coords=SUPERWEIGHTS_OLMO1B):
    values = {}
    for layer, row, col in coords:
        w = model.model.layers[layer].mlp.down_proj.weight
        values[f"{layer}:{row}:{col}"] = float(w[row, col].detach().cpu())
    return values


def save_fp16_superweights(model, output_json, coords=SUPERWEIGHTS_OLMO1B):
    values = read_fp16_superweights(model, coords)
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(values, f, indent=2)
    return values


def _get_dequant_weight(linear):
    """
    Works for normal Linear and many quantized Linear wrappers.
    If AutoAWQ exposes .weight as dequantized/pseudo weight, this is enough.
    """
    w = linear.weight
    if callable(getattr(w, "dequantize", None)):
        w = w.dequantize()
    return w.detach()


def compute_awq_deltas(model, fp16_values, coords=SUPERWEIGHTS_OLMO1B):
    deltas = {}

    for layer, row, col in coords:
        key = f"{layer}:{row}:{col}"
        down_proj = model.model.layers[layer].mlp.down_proj

        w_awq = _get_dequant_weight(down_proj)
        awq_value = float(w_awq[row, col].detach().cpu())
        fp16_value = float(fp16_values[key])

        deltas[key] = {
            "layer": layer,
            "row": row,
            "col": col,
            "fp16_value": fp16_value,
            "awq_dequant_value": awq_value,
            "delta": fp16_value - awq_value,
        }

    return deltas


def save_awq_deltas(model, fp16_json, output_json, coords=SUPERWEIGHTS_OLMO1B):
    with open(fp16_json, "r", encoding="utf-8") as f:
        fp16_values = json.load(f)

    deltas = compute_awq_deltas(model, fp16_values, coords)

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(deltas, f, indent=2)

    return deltas


def apply_superweight_awq_forward_patch(model, delta_json):
    with open(delta_json, "r", encoding="utf-8") as f:
        deltas = json.load(f)

    grouped = {}
    for item in deltas.values():
        grouped.setdefault(item["layer"], []).append(item)

    hooks = []

    for layer, items in grouped.items():
        down_proj = model.model.layers[int(layer)].mlp.down_proj

        def make_hook(items_for_layer):
            def hook(module, inputs, output):
                x = inputs[0]
                y = output.clone()

                for item in items_for_layer:
                    row = int(item["row"])
                    col = int(item["col"])
                    delta = torch.tensor(
                        item["delta"],
                        device=y.device,
                        dtype=y.dtype,
                    )

                    y[..., row] += x[..., col] * delta

                return y

            return hook

        hooks.append(down_proj.register_forward_hook(make_hook(items)))

    return hooks