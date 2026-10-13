"""Train the existing direction head from line endpoints without arc feedback.

Only the predicted line branch receives the zero-valued geometry surrogate.
Its input context and anchor difference are detached. The added derivative
therefore trains only the direction head, preserving the other direct paths.
All forward predictions and parameters initially equal the complete warm model.
"""
import torch

from .server_interval_endpoint_residual_model import (
    IntervalEndpointResidualModel, input_geometry_anchor)


ARCHITECTURE = 'LOCAL_XYZ_LINE_DIRECTION_HEAD_SURROGATE_V6'


class LineDirectionSurrogateModel(IntervalEndpointResidualModel):
    def inherit_complete_endpoint(self, weights):
        self.load_state_dict(weights, strict=True)

    def line_direction_correction(self, hidden, result, query_ratio):
        geometry = hidden[:, 256:]
        forward = torch.zeros(len(hidden), dtype=torch.bool, device=hidden.device)
        first, _ = input_geometry_anchor(
            geometry, result['interval_ratios'], forward, query_ratio)
        second, _ = input_geometry_anchor(
            geometry, result['interval_ratios'], ~forward, query_ratio)
        probability = self.direction(hidden[:, :128].detach()).softmax(1)[:, 1]
        zero_forward = probability - probability.detach()
        line = result['kind'].argmax(1) == 0
        return (zero_forward * line)[:, None, None] * (second - first).detach()

    def predict(self, hidden, query_ratio):
        result = super().predict(hidden, query_ratio)
        result['line_points'] = result['line_points'] + self.line_direction_correction(
            hidden, result, query_ratio)
        return result
