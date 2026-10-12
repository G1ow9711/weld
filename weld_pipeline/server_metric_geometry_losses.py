"""Complete geometry losses measured against the existing three-mm tolerance.

The original masks and all process/frame/classification losses are retained.
No input, reference coordinate, geometry gate, or inference threshold changes.
"""
import math

import torch
from torch.nn import functional as F

from .server_full_weld_model import complete_child_losses, masked_mean


def point_distance_loss(predicted, target):
    # Euclidean distance is invariant to the choice of orthonormal query basis.
    normalized = torch.linalg.vector_norm(predicted - target, dim=-1) / .003
    return F.smooth_l1_loss(normalized, torch.zeros_like(normalized), beta=1., reduction='none')


def arc_points(shape, fractions):
    center = shape[:, :3]
    radius = shape[:, 3].exp()
    normal = F.normalize(shape[:, 4:7], dim=1, eps=1e-8)
    start = shape[:, 7:10]
    start = F.normalize(start - (start * normal).sum(1, keepdim=True) * normal, dim=1, eps=1e-8)
    side = torch.linalg.cross(normal, start)
    angle = shape[:, 10, None] * math.pi * fractions[None]
    return center[:, None] + radius[:, None, None] * (
        start[:, None] * angle.cos()[:, :, None] + side[:, None] * angle.sin()[:, :, None])


def metric_child_losses(prediction, targets, geometry):
    losses = complete_child_losses(prediction, targets)
    line = targets['kind'] == 0
    arc = ~line
    error = point_distance_loss(prediction['line_points'], targets['line_points'])
    losses['line_points'] = masked_mean(error, line[:, None].expand_as(error))
    if arc.any():
        fractions = torch.linspace(0, 1, 65, device=geometry.device, dtype=geometry.dtype)
        predicted = arc_points(prediction['arc_shape'][arc], fractions)
        reference = arc_points(targets['arc_shape'][arc], fractions)
        losses['arc_shape'] = point_distance_loss(predicted, reference).mean()
    else:
        losses['arc_shape'] = prediction['arc_shape'].sum() * 0
    length_mm = torch.expm1(geometry[:, 2])
    interval_error = (prediction['interval_ratios'] - targets['interval_ratios']) * length_mm[:, None] / 3.
    values = F.smooth_l1_loss(interval_error, torch.zeros_like(interval_error),
                            beta=1., reduction='none')
    losses['interval'] = masked_mean(values, targets['interval_known'][:, None].expand_as(values))
    return losses
