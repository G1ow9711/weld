"""Shared full-shard geometry, process, frame and ordered-parent warm-start.

Input point embeddings remain those of the frozen upstream models. This adds
a trainable downstream head; it does not reconstruct run446's final selector.
Teacher parent order is used only for supervised warm-start. Deployment must
construct candidate parents from predicted children and evaluate whole outputs.
"""
import torch
from torch import nn
from torch.nn import functional as F


class FullWeldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(457, 256), nn.ReLU(),
                                   nn.Linear(256, 128), nn.ReLU())
        self.interval = nn.Linear(128, 4)
        self.reference = nn.Linear(128, 1)
        # All fragment tokens, including their finite arc-length order.
        self.token_encoder = nn.Sequential(nn.Linear(459, 128), nn.ReLU())
        self.query_context = nn.MultiheadAttention(128, 4, batch_first=True)
        self.context_residual = nn.Linear(128, 128)
        nn.init.zeros_(self.context_residual.weight)
        nn.init.zeros_(self.context_residual.bias)
        self.geometry_encoder = nn.Sequential(nn.Linear(15, 128), nn.ReLU())
        self.geometry_residual = nn.Linear(128, 128)
        nn.init.zeros_(self.geometry_residual.weight)
        nn.init.zeros_(self.geometry_residual.bias)
        self.direction = nn.Linear(128, 2)
        self.geometric_kind = nn.Linear(128, 2)
        self.closed = nn.Linear(128, 2)
        # Endpoints relative to an input-derived query frame, in metres.
        # Circle parameters are expressed in that same frame, with raw signs.
        self.line_points = nn.Linear(128, 6)
        self.arc_shape = nn.Linear(128, 11)
        self.attribute_states = nn.Linear(128, 32)
        self.attribute_mm = nn.Linear(128, 8)
        self.frame_axes = nn.Linear(128, 18)
        self.frame_offsets = nn.Linear(128, 6)
        self.next_child = nn.Sequential(nn.Linear(264, 128), nn.ReLU(),
                                        nn.Linear(128, 2))
        layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0,
                                           batch_first=True)
        self.parent_encoder = nn.TransformerEncoder(layer, 2,
                                                    enable_nested_tensor=False)
        self.parent_closed = nn.Linear(128, 2)

    def inherit_query(self, state):
        """Exact column-compatible legacy query experiment; not full run446."""
        self.trunk.load_state_dict({k.removeprefix('trunk.'): v
                                    for k, v in state.items()
                                    if k.startswith('trunk.')})
        self.interval.load_state_dict({k.removeprefix('interval.'): v
                                       for k, v in state.items()
                                       if k.startswith('interval.')})
        self.reference.load_state_dict({k.removeprefix('reference.'): v
                                        for k, v in state.items()
                                        if k.startswith('reference.')})

    def encode_queries(self, features, profile_ids, token_sequences,
                       normalize_tokens, geometry):
        hidden = self.trunk(features)
        context = torch.zeros_like(hidden)
        for pid in torch.unique(profile_ids).tolist():
            indices = torch.where(profile_ids == pid)[0]
            raw = token_sequences[pid]
            if len(raw) == 0:
                raise ValueError('A training query cannot have an empty profile')
            tokens = self.token_encoder(normalize_tokens(raw))[None]
            attended, _ = self.query_context(hidden[indices][None], tokens,
                                             tokens, need_weights=False)
            context[indices] = attended[0]
        return (hidden + self.context_residual(context)
                + self.geometry_residual(self.geometry_encoder(geometry)))

    def predict(self, hidden, query_ratio):
        interval = self.interval(hidden)
        distances = torch.sigmoid(interval[:, :2])
        return dict(hidden=hidden,
                    interval_ratios=torch.stack((query_ratio * (1 - distances[:, 0]),
                        query_ratio + (1 - query_ratio) * distances[:, 1]), 1),
                    weld_type=interval[:, 2:],
                    reference_score=self.reference(hidden).squeeze(1),
                    direction=self.direction(hidden), kind=self.geometric_kind(hidden),
                    closed=self.closed(hidden),
                    line_points=self.line_points(hidden).reshape(-1, 2, 3),
                    arc_shape=self.arc_shape(hidden),
                    attribute_states=self.attribute_states(hidden).reshape(-1, 8, 4),
                    attribute_mm=F.softplus(self.attribute_mm(hidden)) * 50,
                    frame_axes=self.frame_axes(hidden).reshape(-1, 2, 3, 3),
                    frame_offsets=self.frame_offsets(hidden).reshape(-1, 2, 3))

    def link_logits(self, hidden, left, right, relative):
        if relative.shape != (len(left), 8):
            raise ValueError('Input-only directional link features must have eight columns')
        return self.next_child(torch.cat((hidden[left], hidden[right], relative), 1))

    def parent_logits(self, hidden, children):
        if not children:
            return hidden.new_empty((0, 2))
        length = max(len(indices) for indices in children)
        sequences = hidden.new_zeros((len(children), length, 128))
        valid = torch.zeros((len(children), length), device=hidden.device,
                             dtype=torch.bool)
        for i, indices in enumerate(children):
            if not indices:
                raise ValueError('Empty teacher parent cannot enter the encoder')
            ordered = hidden[indices]
            # Explicit sequence order, not a permutation-invariant reduction.
            position = torch.arange(len(indices), device=hidden.device,
                                     dtype=hidden.dtype)[:, None]
            scale = torch.exp(torch.arange(0, 128, 2, device=hidden.device,
                                            dtype=hidden.dtype) * (-9.210340371976184 / 128))
            encoding = torch.zeros_like(ordered)
            encoding[:, 0::2] = torch.sin(position * scale)
            encoding[:, 1::2] = torch.cos(position * scale)
            sequences[i, :len(indices)] = ordered + encoding
            valid[i, :len(indices)] = True
        encoded = self.parent_encoder(sequences, src_key_padding_mask=~valid)
        return self.parent_closed((encoded * valid[:, :, None]).sum(1)
                                  / valid.sum(1)[:, None])


def masked_mean(values, mask):
    """Unknown labels have no term, and are never converted to zero labels."""
    if mask.dtype != torch.bool or values.shape != mask.shape:
        raise ValueError('A same-shape explicit boolean supervision mask is required')
    return values[mask].mean() if mask.any() else values.sum() * 0


def complete_child_losses(prediction, targets):
    """No reference child is dropped for lack of a D3-positive carrier."""
    p, t = prediction, targets
    line = t['kind'] == 0
    arc = ~line
    losses = dict(kind=F.cross_entropy(p['kind'], t['kind']),
                  weld_type=F.cross_entropy(p['weld_type'], t['weld_type']),
                  direction=F.cross_entropy(p['direction'], t['direction']),
                  closed=F.cross_entropy(p['closed'], t['closed']))
    # Geometric supervision on free output points can express targets outside
    # the finite carrier pool. Distances are in metres, not truncated to D3.
    point_error = F.smooth_l1_loss(p['line_points'], t['line_points'], reduction='none', beta=.003)
    losses['line_points'] = masked_mean(point_error, line[:, None, None].expand_as(point_error))
    shape_error = F.smooth_l1_loss(p['arc_shape'], t['arc_shape'], reduction='none', beta=.01)
    losses['arc_shape'] = masked_mean(shape_error, arc[:, None].expand_as(shape_error))
    interval_error = F.smooth_l1_loss(p['interval_ratios'], t['interval_ratios'], reduction='none', beta=.01)
    losses['interval'] = masked_mean(interval_error, t['interval_known'][:, None].expand_as(interval_error))
    # Preserve the four export encodings. -1 never enters cross entropy.
    known = t['state_known'] & t['attribute_applicable']
    safe_state = t['attribute_state'].masked_fill(~known, 0)
    state_error = F.cross_entropy(p['attribute_states'].reshape(-1, 4),
                                  safe_state.reshape(-1), reduction='none').reshape(-1, 8)
    losses['attribute_state'] = masked_mean(state_error, known)
    exact = t['numeric_known'] & t['attribute_applicable']
    safe_mm = t['attribute_mm'].masked_fill(~exact, 0)
    numeric_error = F.smooth_l1_loss(p['attribute_mm'] / 50, safe_mm / 50,
                                    reduction='none', beta=.02)
    losses['attribute_mm'] = masked_mean(numeric_error, exact)
    frame_known = t['frame_known']
    safe_axes = t['frame_axes'].masked_fill(~frame_known[:, :, None, None], 0)
    axis_error = F.smooth_l1_loss(p['frame_axes'], safe_axes, reduction='none', beta=.02)
    losses['frame_axes'] = masked_mean(axis_error, frame_known[:, :, None, None].expand_as(axis_error))
    safe_offsets = t['frame_offsets'].masked_fill(~frame_known[:, :, None], 0)
    offset_error = F.smooth_l1_loss(p['frame_offsets'], safe_offsets, reduction='none', beta=.003)
    losses['frame_offsets'] = masked_mean(offset_error, frame_known[:, :, None].expand_as(offset_error))
    return losses
