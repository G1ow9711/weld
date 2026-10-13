"""Warm-preserving nonlinear corrections to complete input-relative geometry.

The complete endpoint model, its point features and all process/frame heads
remain. A zero output layer initially preserves every named prediction. Its
encoder receives only the learned input context and the existing 15-column
input geometry descriptor. Experimental weights need complete-fit admission.
"""
import torch
from torch import nn

from .server_interval_endpoint_residual_model import (
    IntervalEndpointResidualModel, InputEndpointProjection, input_geometry_anchor)


ARCHITECTURE = 'LOCAL_XYZ_NONLINEAR_GEOMETRY_RESIDUAL_V6'


class NonlinearGeometryResidualModel(IntervalEndpointResidualModel):
    def __init__(self):
        super().__init__()
        self.geometry_refinement_encoder = nn.Sequential(
            nn.Linear(143, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU())
        # Two carrier endpoint corrections in metres, six free line endpoint
        # corrections in metres and the original eleven arc shape channels.
        self.geometry_refinement_output = nn.Linear(128, 19)
        nn.init.zeros_(self.geometry_refinement_output.weight)
        nn.init.zeros_(self.geometry_refinement_output.bias)

    def inherit_complete_endpoint(self, weights):
        loaded = self.load_state_dict(weights, strict=False)
        expected = {name for name in self.state_dict()
                    if name.startswith('geometry_refinement_')}
        if set(loaded.missing_keys) != expected or loaded.unexpected_keys:
            raise ValueError('Every complete warm endpoint parameter must be inherited')

    def predict(self, hidden, query_ratio):
        result = super().predict(hidden, query_ratio)
        geometry = hidden[:, 256:]
        context = torch.cat((hidden[:, :128], geometry), dim=1)
        correction = self.geometry_refinement_output(
            self.geometry_refinement_encoder(context))
        if not torch.isfinite(correction).all():
            raise ValueError('Nonfinite nonlinear geometry correction')
        length_m = torch.expm1(geometry[:, 2]) / 1000
        previous = result['interval_ratios']
        corrected = InputEndpointProjection.apply(
            previous + correction[:, :2] / length_m[:, None], query_ratio)
        reverse = result['direction'].argmax(1).bool()
        old_points, old_shape = input_geometry_anchor(
            geometry, previous, reverse, query_ratio)
        new_points, new_shape = input_geometry_anchor(
            geometry, corrected, reverse, query_ratio)
        result['interval_ratios'] = corrected
        result['line_points'] = (result['line_points'] + (new_points - old_points)
                                 + correction[:, 2:8].reshape(-1, 2, 3))
        result['arc_shape'] = (result['arc_shape'] + (new_shape - old_shape)
                               + correction[:, 8:])
        return result
