from __future__ import annotations

import torch
from typing import List
from src.hooks import get_nested_attr


def asymmetric_rtn_per_token(x: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """
    Fake quantization per token over the last dimension.
    Input shape expected: [batch, seq, hidden]
    """
    if not torch.is_tensor(x):
        return x

    x_f = x.float()
    shape = x_f.shape
    hidden = shape[-1]
    flat = x_f.reshape(-1, hidden)

    qmin = 0
    qmax = (1 << n_bits) - 1

    mins = flat.min(dim=1, keepdim=True).values
    maxs = flat.max(dim=1, keepdim=True).values
    scales = (maxs - mins) / max(qmax - qmin, 1)
    scales = torch.where(scales < 1e-12, torch.ones_like(scales), scales)

    zero_points = torch.round(qmin - mins / scales).clamp(qmin, qmax)
    q = torch.round(flat / scales + zero_points).clamp(qmin, qmax)
    deq = (q - zero_points) * scales

    return deq.reshape(shape).to(x.dtype)


class ActivationQuantHook:
    def __init__(
        self,
        layers,
        module_path: str,
        layer_indices,
        n_bits: int = 8,
        mode: str = "naive",   # "naive" or "super"
    ):
        self.handles: List = []
        self.n_bits = n_bits
        self.mode = mode

        for layer_idx in layer_indices:
            module = get_nested_attr(layers[layer_idx], module_path)
            handle = module.register_forward_hook(self._make_hook())
            self.handles.append(handle)

    def _make_hook(self):
        def hook(module, inputs, output):
            if not torch.is_tensor(output):
                return output

            if self.mode == "naive":
                return asymmetric_rtn_per_token(output, n_bits=self.n_bits)

            if self.mode == "super":
                out = output.detach().clone().float()

                flat = out.view(-1)
                max_idx = flat.abs().argmax()
                saved_val = flat[max_idx].clone()

                flat[max_idx] = out.median()

                out_q = asymmetric_rtn_per_token(out, n_bits=self.n_bits).float()
                out_q.view(-1)[max_idx] = saved_val

                return out_q.to(output.dtype)

            raise ValueError(f"Unknown mode: {self.mode}")

        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()