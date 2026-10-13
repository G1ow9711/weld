"""Preserve hard geometry outputs and let geometry train their direction head.

The old argmax-selected anchors have no direction gradient. A zero-valued
surrogate adds only the derivative of the two analytic orientation anchors.
Its difference is detached, so the original interval and free geometry
gradient paths are retained. This experimental recipe needs full admission.
"""
import torch

from .server_interval_endpoint_residual_model import (
    IntervalEndpointResidualModel, input_geometry_anchor)


ARCHITECTURE = 'LOCAL_XYZ_GEOMETRY_DIRECTION_SURROGATE_V6'


class GeometryDirectionSurrogateModel(IntervalEndpointResidualModel):
    def inherit_complete_endpoint(self, weights):
        self.load_state_dict(weights, strict=True)

    def predict(self, hidden, query_ratio):
        result = super().predict(hidden, query_ratio)
        geometry = hidden[:, 256:]
        forward = torch.zeros(len(hidden), dtype=torch.bool, device=hidden.device)
        reverse = ~forward
        forward_points, forward_shape = input_geometry_anchor(
            geometry, result['interval_ratios'], forward, query_ratio)
        reverse_points, reverse_shape = input_geometry_anchor(
            geometry, result['interval_ratios'], reverse, query_ratio)
        probability = result['direction'].softmax(1)[:, 1]
        zero_forward = probability - probability.detach()
        result['line_points'] = result['line_points'] + zero_forward[:, None, None] * (
            reverse_points - forward_points).detach()
        result['arc_shape'] = result['arc_shape'] + zero_forward[:, None] * (
            reverse_shape - forward_shape).detach()
        return result
