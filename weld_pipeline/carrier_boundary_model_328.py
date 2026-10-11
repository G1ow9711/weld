"""Fresh shared 128 sequence encoder; finite boundary/child/pointer heads.

Full original XYZ form differentiable context. Local inputs are computed along
all pure carriers, not predicted/GT neighborhoods. No world-sized endpoint
regression and no names, target counts, CAD or target files enter this API.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .raw_point_groups_285 import RawPointSource, TOKEN_CAPACITY
from .carrier_observation_sequences_328 import FEATURE_WIDTH

VERSION = "SHARED128_FINITE_CELL_BOUNDARY_POINTER_MODEL_328_v1"
SEQUENCE_HALO_ROWS = 2
ROLE_FIELDS = ("boundary", "child_start", "child_end", "parent_start", "parent_end")


class AllOriginalXYZContext(nn.Module):
    def __init__(self, point_block=16384):
        super().__init__()
        self.point_block = point_block
        self.point_encoder = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, 64), nn.SiLU())
        self.token_encoder = nn.Sequential(nn.Linear(132, 128), nn.SiLU(), nn.Linear(128, 128))

    def _pool(self, values, groups):
        encoded = self.point_encoder(values)
        sums = encoded.new_zeros((TOKEN_CAPACITY, 64)).index_add(0, groups, encoded)
        peaks = encoded.new_full((TOKEN_CAPACITY, 64), -torch.inf)
        peaks = peaks.scatter_reduce(0, groups[:, None].expand(-1, 64), encoded, reduce="amax", include_self=True)
        return sums, peaks

    def forward(self, source):
        if type(source) is not RawPointSource or next(self.parameters()).device.type != "cpu":
            raise ValueError("Verified original XYZ on the registered CPU required")
        sums = torch.zeros((TOKEN_CAPACITY, 64))
        peaks = torch.full_like(sums, -torch.inf)
        consumed = 0
        for first, values, groups in source.blocks(self.point_block):
            if first != consumed:
                raise ValueError("Omitted or repeated original source rows")
            features, group_ids = torch.from_numpy(values), torch.from_numpy(groups)
            if self.training and torch.is_grad_enabled():
                features.requires_grad_(True)
                add, maximum = checkpoint(self._pool, features, group_ids, preserve_rng_state=False)
            else:
                add, maximum = self._pool(features, group_ids)
            sums, peaks = sums + add, torch.maximum(peaks, maximum)
            consumed += len(values)
        if consumed != len(source.xyz_mm):
            raise ValueError("Full original XYZ denominator must contribute")
        counts = torch.from_numpy(source.counts.copy())
        available = counts > 0
        mean = sums / counts.clamp(min=1).float()[:, None]
        peaks = torch.where(available[:, None], peaks, torch.zeros_like(peaks))
        centers = torch.tensor((source.mean_xyz_mm[available.numpy()] - source.center_mm) / 1000., dtype=torch.float32)
        relative_counts = counts[available].float().log1p() / math.log1p(consumed)
        tokens = self.token_encoder(torch.cat((mean[available], peaks[available], centers, relative_counts[:, None]), dim=1))
        return tokens, dict(original_points_consumed=consumed, original_points=len(source.xyz_mm),
            nonempty_original_groups=len(tokens), target_reads=0, STEP_reads=0, original_rows_downsampled=False)


class CarrierBoundaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = AllOriginalXYZContext()
        self.local_context = nn.Linear(128, 128, bias=False)
        self.global_context = nn.Linear(128, 128, bias=False)
        self.first = nn.Linear(FEATURE_WIDTH, 128)
        self.first_sequence = nn.Conv1d(128, 128, 3, padding=1, groups=128)
        self.second = nn.Linear(128, 128)
        self.second_sequence = nn.Conv1d(128, 128, 3, padding=1, groups=128)
        self.activation = nn.SiLU()
        self.subslot_embedding = nn.Parameter(torch.empty(2, 128))
        nn.init.normal_(self.subslot_embedding, std=.02)
        self.roles = nn.Linear(128, len(ROLE_FIELDS))
        self.weld_type = nn.Linear(128, 2)
        self.within_cell = nn.Linear(128, 1)
        self.residual = nn.Linear(128, 7)
        self.terminal_key = nn.Linear(128, 32)
        self.terminal_query = nn.Linear(128, 32)
        self.relationship = nn.Sequential(nn.Linear(128 * 4 + 4, 128), nn.SiLU(), nn.Linear(128, 3))
        with torch.no_grad():
            self.roles.bias.fill_(-4.)
            self.residual.weight.zero_(); self.residual.bias.zero_()
            self.within_cell.weight.zero_(); self.within_cell.bias.zero_()

    def encode_sequence(self, features, nearby_context, global_context):
        if features.ndim != 2 or features.shape[1] != FEATURE_WIDTH or nearby_context.shape != (len(features), 128) or global_context.shape != (128,):
            raise ValueError("Pure full sequence observations and original context required")
        values = self.activation(self.first(features) + self.local_context(nearby_context) + self.global_context(global_context)[None])
        values = self.activation(values + self.first_sequence(values.T[None])[0].T)
        values = self.activation(self.second(values))
        return self.activation(values + self.second_sequence(values.T[None])[0].T)

    def slot_embeddings(self, sequence):
        return sequence[:, None, :] + self.subslot_embedding[None]

    def heads(self, sequence):
        embeddings = self.slot_embeddings(sequence)
        return dict(embeddings=embeddings, role_logits=self.roles(embeddings), weld_type_logits=self.weld_type(embeddings),
            within_cell_fraction=self.within_cell(embeddings).squeeze(-1).sigmoid(), residual=self.residual(embeddings),
            terminal_keys=self.terminal_key(embeddings), terminal_queries=self.terminal_query(embeddings))

    def pointer_logits(self, start_embedding, terminal_keys):
        return terminal_keys @ self.terminal_query(start_embedding) / math.sqrt(32.)

    def relation_logits(self, start, end, next_start, next_end, geometric_features):
        return self.relationship(torch.cat((start, end, next_start, next_end, geometric_features), dim=-1))
