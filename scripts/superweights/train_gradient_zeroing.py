from __future__ import annotations

import torch


OLMO1B_SUPERWEIGHTS = [
    {"layer": 1, "row": 1764, "col": 1710},
    {"layer": 1, "row": 1764, "col": 8041},
]


def get_olmo_down_proj_weight(model, layer: int) -> torch.nn.Parameter:
    """
    Returns the down projection weight for OLMo-style HuggingFace models.
    """
    return model.model.layers[layer].mlp.down_proj.weight


def zero_superweight_gradients(model, superweights) -> None:
    """
    Sets gradients of selected superweight coordinates to zero.
    Call this after loss.backward() and before optimizer.step().
    """
    for sw in superweights:
        layer = sw["layer"]
        row = sw["row"]
        col = sw["col"]

        weight = get_olmo_down_proj_weight(model, layer)

        if weight.grad is None:
            raise RuntimeError(f"No gradient found for layer {layer} down_proj.weight")

        weight.grad[row, col] = 0.0


def print_superweight_gradients(model, superweights, title: str) -> None:
    print(f"\n{title}")
    for sw in superweights:
        layer = sw["layer"]
        row = sw["row"]
        col = sw["col"]

        weight = get_olmo_down_proj_weight(model, layer)

        grad_value = None
        if weight.grad is not None:
            grad_value = weight.grad[row, col].item()

        print(
            f"layer={layer}, row={row}, col={col}, grad={grad_value}"
        )
