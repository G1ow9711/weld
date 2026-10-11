"""Full geometry residuals anchored to a frozen, column-compatible old query.

The inherited point features and old query remain fixed. New complete-task
geometry can leave the carrier, while its zero residual starts on the existing
finite geometry instead of random endpoints. This is not the final run446
526-column selector, which still needs complete deployment validation.
"""
import math

import torch
from torch import nn

from .server_full_weld_model import FullWeldModel
from .server_joint_reference_query_model import JointReferenceModel


class LegacyAnchoredFullWeldModel(FullWeldModel):
    def __init__(self):
        super().__init__()
        self.legacy = JointReferenceModel()
        self.interval_delta = nn.Linear(128, 4)
        self.legacy.requires_grad_(False)
        self.interval.requires_grad_(False)
        self.reference.requires_grad_(False)
        self.reset_geometry_residuals()

    def reset_geometry_residuals(self):
        for head in (self.interval_delta, self.line_points, self.arc_shape):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def initialize(self, warm_state, legacy_state):
        result = self.load_state_dict(warm_state, strict=False)
        expected = {name for name in self.state_dict()
                    if name.startswith(('legacy.', 'interval_delta.'))}
        if set(result.missing_keys) != expected or result.unexpected_keys:
            raise ValueError('Full warm-start architecture is incompatible')
        self.legacy.load_state_dict(legacy_state)
        self.reset_geometry_residuals()

    def encode_queries(self, features, profile_ids, token_sequences,
                       normalize_tokens, geometry):
        main = super().encode_queries(features, profile_ids, token_sequences,
                                      normalize_tokens, geometry)
        with torch.no_grad():
            old_hidden = self.legacy.trunk(features)
        # Input-only geometry accompanies the two feature branches. Output
        # child embeddings remain the first 128 columns, not reference data.
        return torch.cat((main, old_hidden, geometry), dim=1)

    def predict(self, hidden, query_ratio):
        if hidden.shape[1] != 271:
            raise ValueError('Anchored query requires 128+128+15 input columns')
        main, old, geometry = hidden[:, :128], hidden[:, 128:256], hidden[:, 256:]
        result = super().predict(main, query_ratio)
        logits = self.legacy.interval(old) + self.interval_delta(main)
        distances = torch.sigmoid(logits[:, :2])
        bounds = torch.stack((query_ratio * (1 - distances[:, 0]),
                   query_ratio + (1 - query_ratio) * distances[:, 1]), dim=1)
        result['interval_ratios'] = bounds
        result['weld_type'] = logits[:, 2:]
        result['reference_score'] = self.legacy.reference(old).squeeze(1)
        reverse = result['direction'].argmax(1).bool()
        ordered = torch.where(reverse[:, None], bounds.flip(1), bounds)
        length_m = torch.expm1(geometry[:, 2]) / 1000
        line_anchor = torch.zeros_like(result['line_points'])
        line_anchor[:, :, 0] = (ordered - query_ratio[:, None]) * length_m[:, None]
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
        arc_anchor = center[:, None] + torch.exp(log_radius)[:, None, None] * rotated
        anchor = torch.where(arc[:, None, None], arc_anchor, line_anchor)
        result['line_points'] = anchor + result['line_points']
        sweep_out = (bounds[:, 1] - bounds[:, 0]) * sweep
        sweep_out = torch.where(reverse, -sweep_out, sweep_out)
        arc_shape = torch.cat((center, log_radius[:, None], normal,
                               rotated[:, 0], (sweep_out / math.pi)[:, None]), dim=1)
        result['arc_shape'] = arc_shape + result['arc_shape']
        result['hidden'] = main
        return result

    def parent_logits(self, hidden, children):
        return super().parent_logits(hidden[:, :128], children)

    def link_logits(self, hidden, left, right, relative):
        return super().link_logits(hidden[:, :128], left, right, relative)
