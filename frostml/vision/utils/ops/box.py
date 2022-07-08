import torch

from torch import Tensor


__all__ = [
    'box_cxcywh_to_xyxy',
    'box_xyxy_to_cxcywh',
]


def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    cx, cy, ws, hs = x.unbind(-1)
    b = [(cx - 0.5 * ws), (cy - 0.5 * hs), (cx + 0.5 * ws), (cy + 0.5 * hs)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
    return torch.stack(b, dim=-1)
