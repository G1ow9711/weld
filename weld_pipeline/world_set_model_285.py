"""B: every original XYZ row -> pooled tokens -> parent/child sequence.

forward accepts only RawPointSource. Labels, STEP, names and GT query positions
are absent. The fixed global capacities belong to the registered model.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .contract_attribute_heads_283 import ContractAttributeHeads
from .raw_point_groups_285 import RawPointSource, TOKEN_CAPACITY


def observed_reference_rows(positions, query_capacity):
    """Deterministic farthest observed group centres; never GT locations."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise ValueError("Nonempty pure-source group centres required")
    center = positions.mean(axis=0)
    next_row = int(np.argmax(np.sum((positions - center) ** 2, axis=1)))
    distances = np.full(len(positions), np.inf)
    selected = []
    for _ in range(min(query_capacity, len(positions))):
        selected.append(next_row)
        distances = np.minimum(distances, np.sum((positions - positions[next_row]) ** 2, axis=1))
        distances[selected] = -1
        next_row = int(np.argmax(distances))
    return np.asarray([selected[i % len(selected)] for i in range(query_capacity)], np.int64)


class WorldSetModel(nn.Module):
    def __init__(self, parent_queries, child_capacity, hidden=128, point_block_size=16384):
        super().__init__()
        if (type(parent_queries) is not int or parent_queries < 1 or type(child_capacity) is not int
                or child_capacity < 1 or hidden % 4 or point_block_size < 1):
            raise ValueError("Registered finite global model capacities required")
        self.parent_queries, self.child_capacity = parent_queries, child_capacity
        self.hidden, self.point_block_size = hidden, point_block_size
        self.point_encoder = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, 64), nn.SiLU())
        self.token_encoder = nn.Sequential(nn.Linear(64 * 2 + 4, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.query_positions = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.parent_queries_embedding = nn.Embedding(parent_queries, hidden)
        decoder_layer = nn.TransformerDecoderLayer(hidden, 4, dim_feedforward=hidden * 2,
            dropout=0., activation="gelu", batch_first=True)
        self.parent_decoder = nn.TransformerDecoder(decoder_layer, 2)
        self.presence = nn.Linear(hidden, 1)
        self.parent_mode = nn.Linear(hidden, 3)
        self.parent_closed = nn.Linear(hidden, 2)
        self.parent_direction = nn.Linear(hidden, 2)
        self.parent_turns = nn.Linear(hidden, 2)
        self.line_endpoints = nn.Linear(hidden, 6)
        self.circle_parameters = nn.Linear(hidden, 10)  # centre3, logRadius1, quaternion4, phase1, sweep1
        self.sequence_positions = nn.Embedding(child_capacity + 1, hidden)
        self.sequence_encoder = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.sequence_kind = nn.Linear(hidden, 3)  # STOP, line, arc
        self.sequence_weld_type = nn.Linear(hidden, 2)  # H, V
        self.sequence_gap = nn.Linear(hidden, 2)  # shared, positive
        self.sequence_interval = nn.Linear(hidden, 2)  # positive child and gap lengths
        self.mixed_endpoints = nn.Linear(hidden, 6)
        self.mixed_arc = nn.Linear(hidden, 4)  # raw normal3, sweep logit
        self.mixed_arc_direction = nn.Linear(hidden, 2)
        self.contract_attributes = ContractAttributeHeads(hidden)
        with torch.no_grad():
            self.presence.bias.fill_(-2.197224577)
            self.line_endpoints.weight.zero_()
            self.line_endpoints.bias.copy_(torch.tensor([-.05, 0., 0., .05, 0., 0.]))
            self.circle_parameters.bias.zero_()
            self.circle_parameters.bias[3] = math.log(math.expm1(.1))
            self.circle_parameters.bias[4] = 1.
            self.mixed_arc.bias.zero_()
            self.mixed_arc.bias[2] = 1.

    def _point_pool(self, features, group_ids):
        encoded = self.point_encoder(features)
        sums = encoded.new_zeros((TOKEN_CAPACITY, 64)).index_add(0, group_ids, encoded)
        maxima = encoded.new_full((TOKEN_CAPACITY, 64), -torch.inf)
        maxima = maxima.scatter_reduce(0, group_ids[:, None].expand(-1, 64), encoded,
                                        reduce="amax", include_self=True)
        return sums, maxima

    def encode_original_source(self, source):
        if type(source) is not RawPointSource:
            raise ValueError("Only the pure original XYZ source reader is accepted")
        device = next(self.parameters()).device
        if device.type != "cpu":
            raise ValueError("First registered B round uses the measured CPU model")
        sums = torch.zeros((TOKEN_CAPACITY, 64), dtype=torch.float32, device=device)
        maxima = torch.full_like(sums, -torch.inf)
        consumed = 0
        for first, features, groups in source.blocks(self.point_block_size):
            if first != consumed:
                raise ValueError("Original point row sequence omitted or repeated")
            values = torch.from_numpy(features).to(device)
            indices = torch.from_numpy(groups).to(device)
            if self.training and torch.is_grad_enabled():
                # Reentrant checkpoint needs a grad-bearing input. The source
                # copy is never written back; gradients update network weights.
                values.requires_grad_(True)
                added, peak = checkpoint(self._point_pool, values, indices, preserve_rng_state=False)
            else:
                added, peak = self._point_pool(values, indices)
            sums = sums + added
            maxima = torch.maximum(maxima, peak)
            consumed += len(features)
        if consumed != len(source.xyz_mm):
            raise ValueError("All original XYZ must contribute to this forward")
        counts = torch.from_numpy(source.counts.copy()).to(device)
        available = counts > 0
        mean = sums / counts.clamp(min=1).to(sums.dtype)[:, None]
        maxima = torch.where(available[:, None], maxima, torch.zeros_like(maxima))
        positions = (source.mean_xyz_mm[available.numpy()] - source.center_mm) / 1000.
        positions_tensor = torch.tensor(positions, dtype=torch.float32, device=device)
        relative_count = counts[available].to(torch.float32).log1p() / math.log1p(consumed)
        token_features = torch.cat((mean[available], maxima[available], positions_tensor, relative_count[:, None]), dim=1)
        tokens = self.token_encoder(token_features)
        rows = observed_reference_rows(positions, self.parent_queries)
        reference = positions_tensor[torch.from_numpy(rows).to(device)]
        origin = torch.tensor(source.center_mm.copy(), dtype=torch.float64, device=device)
        return tokens, reference, origin, dict(original_points_consumed=consumed,
            original_points=len(source.xyz_mm), nonempty_source_tokens=len(tokens), truth_point_queries=0,
            STEP_reads=0, group_centers_are_final_coordinates=False)

    def forward(self, source):
        tokens, reference, origin, input_receipt = self.encode_original_source(source)
        query = self.parent_queries_embedding.weight + self.query_positions(reference)
        parents = self.parent_decoder(query[None], tokens[None])[0]
        positions = self.sequence_positions.weight
        children = self.sequence_encoder(parents[:, None] + positions[None])
        line = self.line_endpoints(parents).reshape(-1, 2, 3) + reference[:, None]
        circle = self.circle_parameters(parents)
        circle_center = circle[:, :3] + reference
        mixed = self.mixed_endpoints(children).reshape(self.parent_queries, self.child_capacity + 1, 2, 3)
        mixed = mixed + reference[:, None, None]
        return dict(version="PURE_RAW_WORLD_PARENT_QUERY_CHILD_SEQUENCE_285_v1", origin_mm=origin,
            input_receipt=input_receipt, parent_embeddings=parents, child_embeddings=children,
            presence_logits=self.presence(parents).squeeze(-1), mode_logits=self.parent_mode(parents),
            closed_logits=self.parent_closed(parents), direction_logits=self.parent_direction(parents),
            turns_logits=self.parent_turns(parents), line_endpoints_m=line,
            circle_center_m=circle_center, circle_radius_m=F.softplus(circle[:, 3]),
            circle_quaternion=circle[:, 4:8], circle_phase_rad=circle[:, 8],
            circle_open_sweep_rad=2 * math.pi * torch.sigmoid(circle[:, 9]),
            sequence_kind_logits=self.sequence_kind(children), weld_type_logits=self.sequence_weld_type(children),
            gap_logits=self.sequence_gap(children), interval_weights=F.softplus(self.sequence_interval(children)),
            mixed_endpoints_m=mixed, mixed_arc_parameters=self.mixed_arc(children),
            mixed_direction_logits=self.mixed_arc_direction(children),
            global_parent_query_capacity=self.parent_queries, global_child_sequence_capacity=self.child_capacity)
