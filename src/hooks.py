from typing import Dict, List
import torch


def get_nested_attr(obj, attr_path: str):
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


def summarize_activation(tensor: torch.Tensor):
    """
    Expected tensor shape: [batch, seq_len, hidden_dim]
    Stores only summary stats, not the full tensor.
    """
    x = tensor.detach()

    if x.dim() != 3:
        return None

    abs_x = x.abs()
    max_abs_value = abs_x.max().item()
    flat_idx = abs_x.view(-1).argmax().item()

    batch_size, seq_len, hidden_dim = abs_x.shape
    b = flat_idx // (seq_len * hidden_dim)
    rem = flat_idx % (seq_len * hidden_dim)
    s = rem // hidden_dim
    h = rem % hidden_dim

    signed_value = x[b, s, h].item()

    return {
        "max_abs_value": max_abs_value,
        "signed_value": signed_value,
        "batch_idx": b,
        "token_idx": s,
        "channel_idx": h,
    }


class ActivationRecorder:
    def __init__(self):
        self.inputs: Dict[int, dict] = {}
        self.outputs: Dict[int, dict] = {}
        self.handles: List = []

    def register_on_layers(self, layers, module_path: str):
        for layer_idx, layer in enumerate(layers):
            module = get_nested_attr(layer, module_path)

            def hook_fn(idx):
                def hook(module, inputs, outputs):
                    self.inputs[idx] = summarize_activation(inputs[0])
                    self.outputs[idx] = summarize_activation(outputs)
                return hook

            handle = module.register_forward_hook(hook_fn(layer_idx))
            self.handles.append(handle)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()