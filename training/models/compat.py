"""Small adapters for Pointcept's standalone SpUNet on native Windows.

The segmentation architecture is unchanged. Registry wiring and optional
classification pooling do not require the full Pointcept training engine.
"""
import torch
from torch.nn.init import trunc_normal_


class _Registry:
    def register_module(self, name=None):
        def register(cls):
            return cls
        return register


MODELS = _Registry()


def offset2batch(offset):
    counts = torch.diff(offset, prepend=offset.new_zeros(1))
    return torch.repeat_interleave(
        torch.arange(len(counts), device=offset.device), counts.long()
    )


def scatter(src, index, dim=0, reduce="sum"):
    if dim != 0 or reduce not in ("sum", "mean"):
        raise NotImplementedError("Only dim=0 sum/mean is required by SpUNet.")
    size = int(index.max().item()) + 1 if index.numel() else 0
    out = src.new_zeros((size,) + tuple(src.shape[1:]))
    out.index_add_(0, index.long(), src)
    if reduce == "mean":
        count = torch.bincount(index.long(), minlength=size).clamp_min(1)
        out = out / count.reshape((-1,) + (1,) * (src.ndim - 1))
    return out
