"""Preserve complete query weights and add a zero-initialized local XYZ branch."""
import torch
from torch import nn

from .server_reference_recall_weld_model import ReferenceRecallWeldModel


ARCHITECTURE = 'LOCAL_XYZ_CONTEXT_ANCHORED_REFERENCE_RECALL_V4'


class LocalQueryWeldModel(ReferenceRecallWeldModel):
    def __init__(self):
        super().__init__()
        self.query_local_encoder = nn.Sequential(nn.Linear(72, 128), nn.LayerNorm(128),
            nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
        self.query_local_residual = nn.Linear(128, 128)
        nn.init.zeros_(self.query_local_residual.weight)
        nn.init.zeros_(self.query_local_residual.bias)

    def inherit_complete_query(self, weights):
        result = self.load_state_dict(weights, strict=False)
        expected = {name for name in self.state_dict()
                    if name.startswith(('query_local_encoder.', 'query_local_residual.'))}
        if set(result.missing_keys) != expected or result.unexpected_keys:
            raise ValueError('Every original query parameter must be inherited without omission')

    def encode_queries(self, features, profile_ids, token_sequences,
                       normalize_tokens, geometry, query_local_features):
        if (query_local_features.shape != (len(features), 72)
                or not torch.isfinite(query_local_features).all()):
            raise ValueError('Actual query-local XYZ features are required')
        old = super().encode_queries(features, profile_ids, token_sequences, normalize_tokens, geometry)
        residual = self.query_local_residual(self.query_local_encoder(query_local_features))
        return torch.cat((old[:, :128] + residual, old[:, 128:]), dim=1)
