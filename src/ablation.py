from typing import List, Tuple
import torch


def zero_out_weight(model, layer_path: str, down_proj_path: str, layer_idx: int, row: int, col: int):
    layers = model
    for attr in layer_path.split("."):
        layers = getattr(layers, attr)

    module = layers[layer_idx]
    for attr in down_proj_path.split("."):
        module = getattr(module, attr)

    original_value = module.weight.data[row, col].item()
    module.weight.data[row, col] = 0.0
    return original_value


def restore_weight(model, layer_path: str, down_proj_path: str, layer_idx: int, row: int, col: int, value: float):
    layers = model
    for attr in layer_path.split("."):
        layers = getattr(layers, attr)

    module = layers[layer_idx]
    for attr in down_proj_path.split("."):
        module = getattr(module, attr)

    module.weight.data[row, col] = value