"""Whole input-event intervals with inherited full weld fields and parent heads.

Native queries retain the completed model's genuine legacy inputs. Additional
visual curves use a separate raw-XYZ/image encoder. No fake legacy rows or free
endpoint/circle regression are used for emitted geometry. Training/runtime use
the same endpoint and whole-pair scoring functions.
"""
from copy import deepcopy

import torch
from torch import nn

from .server_interval_endpoint_residual_model import IntervalEndpointResidualModel, input_geometry_anchor
from .server_full_weld_model import FullWeldModel, complete_child_losses
from .server_stable_endpoint_pairs import endpoint_pairs
from .server_stable_joint_interval_residual import JointIntervalResidual
from .server_supported_directed_parent_pool import DirectedLinkModel

ARCHITECTURE = 'SUPPORTED_INPUT_EVENT_INTERVALS_WITH_COMPLETE_WELD_FIELDS_V2'


class SupportedIntervalWeldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = IntervalEndpointResidualModel()
        self.native_event = deepcopy(self.query.legacy.trunk)
        self.native_event.requires_grad_(True)
        # 72 genuine local XYZ + 60 multiview image + 15 input curve columns.
        self.visual_query = nn.Sequential(nn.Linear(147, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.visual_event = nn.Sequential(nn.Linear(160, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.native_visual_residual = nn.Linear(60, 128)
        self.endpoint_score = nn.Sequential(nn.Linear(272, 128), nn.ReLU(), nn.Linear(128, 2))
        self.whole_pair = JointIntervalResidual()
        # Interval-conditioned process fields distinguish multiple complete alternatives.
        self.interval_context = nn.Linear(390, 128)
        self.visual_type = nn.Linear(128, 2)
        self.visual_reference = nn.Linear(128, 1)
        self.directed_link = DirectedLinkModel()
        for layer in (self.native_visual_residual, self.interval_context):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        self._freeze_unused_geometry()

    def _freeze_unused_geometry(self):
        for name in ('line_points', 'arc_shape', 'interval_endpoint_offset'):
            getattr(self.query, name).requires_grad_(False)

    def inherit_complete_query(self, state_dict):
        """Caller must authenticate the complete TRAIN checkpoint before loading."""
        self.query.load_state_dict(state_dict, strict=True)
        self.native_event.load_state_dict(self.query.legacy.trunk.state_dict(), strict=True)
        self._freeze_unused_geometry()

    def encode_native(self, batch, image_features):
        if image_features.shape != (len(batch['features']), 60) or not torch.isfinite(image_features).all():
            raise ValueError('Genuine image features at each native query are required')
        hidden = self.query.encode_queries(**batch)
        return torch.cat((hidden[:, :128] + self.native_visual_residual(image_features), hidden[:, 128:]), 1)

    def encode_visual(self, raw_context):
        if raw_context.ndim != 2 or raw_context.shape[1] != 147 or not torch.isfinite(raw_context).all():
            raise ValueError('Visual curves need 72 raw XYZ, 60 image and 15 input geometry columns')
        return self.visual_query(raw_context)

    def encode_events(self, *, native_features=None, visual_context=None, provenance=None):
        if (native_features is None) == (visual_context is None):
            raise ValueError('Exactly one genuine event feature route must be specified')
        if native_features is not None:
            if native_features.shape[-1] != 457 or not torch.isfinite(native_features).all():
                raise ValueError('Original normalized native event columns required')
            return self.native_event(native_features)
        if (visual_context.shape[-1] != 147 or provenance is None or provenance.shape[-1] != 13
                or visual_context.shape[:-1] != provenance.shape[:-1]
                or not torch.isfinite(visual_context).all() or not torch.isfinite(provenance).all()):
            raise ValueError('Actual raw/visual event context and explicit input provenance required')
        return self.visual_event(torch.cat((visual_context, provenance), -1))

    def score_events(self, query_embedding, events, provenance, positions_mm, query_mm, length_mm):
        q = query_embedding[:, None, :].expand_as(events)
        delta = positions_mm - query_mm[:, None]
        relative = torch.stack((delta / length_mm[:, None], delta / 1000.,
                                length_mm[:, None].expand_as(delta) / 1000.), -1)
        values = torch.cat((q, events, provenance, relative.to(events.dtype)), -1)
        if values.shape[-1] != 272 or not torch.isfinite(values).all():
            raise ValueError('Finite input-only endpoint score features required')
        return self.endpoint_score(values)

    def score_pairs(self, query_embedding, events, provenance, positions_mm, query_mm, length_mm, logits, legal):
        pairs = endpoint_pairs(logits, legal, positions_mm)
        li, ri = pairs['left_ids'], pairs['right_ids']
        gather = lambda values, ids: values.gather(1, ids[:, :, None].expand(-1, -1, values.shape[-1]))
        scores = self.whole_pair(query_embedding, gather(events, li), gather(events, ri),
            gather(provenance, li), gather(provenance, ri), positions_mm.gather(1, li),
            positions_mm.gather(1, ri), query_mm[:, None], length_mm[:, None],
            pairs['base_logits'], pairs['valid'])
        return scores, pairs

    def predict_interval(self, query_hidden, left_embedding, right_embedding, geometry,
                         bounds, query_ratio, *, native, reverse=None):
        """Predict fields on a selected whole input interval; preserve analytic geometry."""
        if (bounds.shape != (len(query_hidden), 2) or not torch.isfinite(bounds).all()
                or (bounds[:, 0] < 0).any() or (bounds[:, 1] > 1).any()
                or (bounds[:, 1] <= bounds[:, 0]).any()):
            raise ValueError('A finite strictly ordered pair of input endpoints is required')
        main = query_hidden[:, :128]
        if query_hidden.shape[1] != (271 if native else 128):
            raise ValueError('Native and visual encodings cannot be substituted for each other')
        if native and not torch.equal(query_hidden[:, 256:], geometry):
            raise ValueError('Native query and selected interval must share input geometry')
        if ((bounds[:, 0] > query_ratio).any() or (bounds[:, 1] < query_ratio).any()):
            raise ValueError('Selected whole interval must enclose its input query')
        # Direction is decided before the supervised orientation conditions
        # process fields. Otherwise teacher orientation leaks its own answer.
        direction_logits = self.query.direction(main)
        if reverse is None:
            reverse = direction_logits.argmax(1).bool()
        length_m = torch.expm1(geometry[:, 2]) / 1000.
        lo, hi = bounds.unbind(1); width = hi - lo
        context = torch.stack(((lo - query_ratio) * length_m, (hi - query_ratio) * length_m,
            width * length_m, width, (query_ratio - lo) / width, reverse.to(main.dtype)), -1)
        main = main + self.interval_context(torch.cat((main, left_embedding, right_embedding, context.to(main.dtype)), 1))
        if native:
            prediction = self.query.predict(torch.cat((main, query_hidden[:, 128:]), 1), query_ratio)
        else:
            prediction = FullWeldModel.predict(self.query, main, query_ratio)
            prediction['weld_type'] = self.visual_type(main)
            correction = self.visual_reference(main.detach()).squeeze(1)
            prediction.update(reference_score=correction, frozen_reference_score=torch.zeros_like(correction),
                              reference_delta_logit=correction)
        points, shape = input_geometry_anchor(geometry.to(bounds.dtype), bounds, reverse, query_ratio)
        prediction.update(direction=direction_logits, line_points=points, arc_shape=shape, interval_ratios=bounds, hidden=main,
                          selected_direction_reversed=reverse)
        return prediction

    def parent_logits(self, hidden, children):
        return self.query.parent_logits(hidden[:, :128], children)

    def reference_link_logits(self, hidden, left, right, relative):
        return self.query.link_logits(hidden[:, :128], left, right, relative)


def known_field_losses(prediction, targets):
    # Frames are located at actual emitted endpoints, not the reference endpoints.
    adjusted = dict(targets, frame_offsets=targets['frame_offsets']
                    + targets['line_points'] - prediction['line_points'].detach())
    losses = complete_child_losses(prediction, adjusted)
    for name in ('line_points', 'arc_shape', 'interval'):
        del losses[name]
    return losses
