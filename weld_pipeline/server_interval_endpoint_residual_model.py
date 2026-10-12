"""Warm-preserving endpoint corrections in metres on the finite input curve.

The existing full local query model and its free geometry remain. A new zero
head moves each carrier endpoint directly, avoiding the sigmoid's scale in the
old prior. The forward endpoints still enclose the real input query. The
projection uses an identity backward path so an overshoot can move back inside.
This experimental architecture needs independent complete training admission.
"""
import math

import torch
from torch import nn

from .server_local_query_weld_model import LocalQueryWeldModel


ARCHITECTURE = 'LOCAL_XYZ_METRE_INTERVAL_ENDPOINT_RESIDUAL_V5'


class InputEndpointProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, query_ratio):
        return torch.stack((
            torch.minimum(values[:, 0].clamp_min(0), query_ratio),
            torch.maximum(values[:, 1].clamp_max(1), query_ratio)), dim=1)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def input_geometry_anchor(geometry, bounds, reverse, query_ratio):
    """Same analytic anchors as the preserved legacy implementation."""
    ordered = torch.where(reverse[:, None], bounds.flip(1), bounds)
    length_m = torch.expm1(geometry[:, 2]) / 1000
    line = geometry.new_zeros((len(geometry), 2, 3))
    line[:, :, 0] = (ordered - query_ratio[:, None]) * length_m[:, None]
    arc = geometry[:, 0].bool()
    default_normal = geometry.new_tensor([0., 0., 1.]).expand(len(geometry), 3)
    default_start = geometry.new_tensor([1., 0., 0.]).expand(len(geometry), 3)
    center = torch.where(arc[:, None], geometry[:, 3:6], torch.zeros_like(default_normal))
    normal = torch.where(arc[:, None], geometry[:, 7:10], default_normal)
    start = torch.where(arc[:, None], geometry[:, 10:13], default_start)
    log_radius = torch.where(arc, geometry[:, 6], torch.log(length_m))
    sweep = torch.where(arc, geometry[:, 13] * math.pi, torch.ones_like(length_m) * math.pi)
    angle = ordered * sweep[:, None]
    rotated = (start[:, None] * torch.cos(angle)[:, :, None]
               + torch.linalg.cross(normal, start)[:, None] * torch.sin(angle)[:, :, None])
    arc_points = center[:, None] + torch.exp(log_radius)[:, None, None] * rotated
    points = torch.where(arc[:, None, None], arc_points, line)
    sweep_out = (bounds[:, 1] - bounds[:, 0]) * sweep
    sweep_out = torch.where(reverse, -sweep_out, sweep_out)
    shape = torch.cat((center, log_radius[:, None], normal,
                       rotated[:, 0], (sweep_out / math.pi)[:, None]), dim=1)
    return points, shape


class IntervalEndpointResidualModel(LocalQueryWeldModel):
    def __init__(self):
        super().__init__()
        self.interval_endpoint_offset = nn.Linear(128, 2)
        nn.init.zeros_(self.interval_endpoint_offset.weight)
        nn.init.zeros_(self.interval_endpoint_offset.bias)

    def inherit_complete_local(self, weights):
        loaded = self.load_state_dict(weights, strict=False)
        expected = {'interval_endpoint_offset.weight', 'interval_endpoint_offset.bias'}
        if set(loaded.missing_keys) != expected or loaded.unexpected_keys:
            raise ValueError('Every complete warm parameter must be inherited')

    def predict(self, hidden, query_ratio):
        result = super().predict(hidden, query_ratio)
        geometry = hidden[:, 256:]
        length_m = torch.expm1(geometry[:, 2]) / 1000
        if (not torch.isfinite(length_m).all() or not (length_m > 0).all()
                or not torch.isfinite(query_ratio).all()
                or not ((query_ratio >= 0) & (query_ratio <= 1)).all()):
            raise ValueError('Finite positive input lengths and actual query ratios are required')
        previous = result['interval_ratios']
        offsets_m = self.interval_endpoint_offset(hidden[:, :128])
        if not torch.isfinite(offsets_m).all():
            raise ValueError('Nonfinite learned input endpoint correction')
        corrected = InputEndpointProjection.apply(previous + offsets_m / length_m[:, None], query_ratio)
        reverse = result['direction'].argmax(1).bool()
        old_points, old_shape = input_geometry_anchor(geometry, previous, reverse, query_ratio)
        new_points, new_shape = input_geometry_anchor(geometry, corrected, reverse, query_ratio)
        result['interval_ratios'] = corrected
        result['line_points'] = result['line_points'] + (new_points - old_points)
        result['arc_shape'] = result['arc_shape'] + (new_shape - old_shape)
        return result
