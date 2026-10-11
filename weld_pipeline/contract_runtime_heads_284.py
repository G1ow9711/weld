"""Prediction-only cardinalities for the complete source graph model.

Counts are targets only inside cardinality_loss. Source IDs, truth counts,
selected truth slots and teacher domain pools are not forward inputs.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .contract_graph_heads_283 import ContractGraphHeads


def cardinality_loss(prediction, counts):
    """-1 stays unknown. Positive counts learn log(count), zero uses a gate."""
    log_count = prediction["positive_log_count"]
    gate = prediction["nonzero_logits"]
    if counts.shape != log_count.shape or gate.shape != (*counts.shape, 2):
        raise ValueError("Cardinality target shape differs from predicted fields")
    if counts.dtype != torch.long or (counts < -1).any():
        raise ValueError("Counts must be integer labels; -1 means unknown")
    if not torch.isfinite(log_count).all() or not torch.isfinite(gate).all():
        raise ValueError("Nonfinite cardinality prediction")
    known, positive = counts >= 0, counts > 0
    gate_loss = F.cross_entropy(gate[known], positive[known].long()) if known.any() else gate.sum() * 0
    positive_loss = (F.smooth_l1_loss(log_count[positive], counts[positive].to(log_count.dtype).log())
                     if positive.any() else log_count.sum() * 0)
    return gate_loss + positive_loss, dict(known_counts=int(known.sum()), positive_counts=int(positive.sum()),
                                          unknown_counts=int((~known).sum()))


def discrete_cardinalities(prediction, capacity):
    """Explicit model quantization. Overflow fails rather than deleting output."""
    if type(capacity) is not int or capacity < 0:
        raise ValueError("Observed source capacity must be a nonnegative integer")
    logs = prediction["positive_log_count"].detach()
    gates = prediction["nonzero_logits"].detach()
    if gates.shape != (*logs.shape, 2) or not torch.isfinite(logs).all() or not torch.isfinite(gates).all():
        raise ValueError("Missing or nonfinite cardinality output")
    if (logs < 0).any():
        raise ValueError("A positive count cannot have a negative logarithm")
    nonzero = gates.argmax(dim=-1).bool()
    # For count n, log(n) -> 1 + nearest(n - 1). Ties round upward.
    # Check the exponential in floating point before casting to int64.
    positive = 1 + torch.floor(torch.expm1(logs.to(torch.float64)) + 0.5)
    if not torch.isfinite(positive).all() or (positive[nonzero] > capacity).any():
        raise ValueError("Predicted cardinality exceeds unchanged source capacity")
    if (positive[nonzero] < 1).any():
        raise ValueError("Predicted positive cardinality is invalid")
    result = torch.zeros_like(logs, dtype=torch.long)
    result[nonzero] = positive[nonzero].long()
    return result


class ContractRuntimeHeads(ContractGraphHeads):
    """All14 graph heads plus learned source-role and allowed-domain counts."""
    def __init__(self, hidden=64):
        super().__init__(hidden)
        self.source_cardinality = nn.Sequential(nn.Linear(19 + 36 + 1, hidden), nn.SiLU(), nn.Linear(hidden, 18))
        self.allowed_domain_cardinality = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.SiLU(), nn.Linear(hidden, 3))

    @staticmethod
    def _prediction(values, count):
        return dict(positive_log_count=F.softplus(values[..., :count]),
                    nonzero_logits=values[..., count:].reshape(*values.shape[:-1], count, 2))

    def predict_source_role_counts(self, source_context, raw_triangle_context, anchor_capacity):
        if source_context.shape != (19,) or raw_triangle_context.shape != (36,):
            raise ValueError("Complete original source contexts required")
        if type(anchor_capacity) is not int or anchor_capacity < 1:
            raise ValueError("Original canonical anchor capacity is required")
        size = source_context.new_tensor([math.log1p(anchor_capacity)])
        features = torch.cat((source_context, raw_triangle_context, size))
        return self._prediction(self.source_cardinality(features), 6)

    def predict_allowed_domain_counts(self, parent_queries, domain_capacity):
        if parent_queries.ndim != 2 or parent_queries.shape[1] != self.hidden:
            raise ValueError("Predicted parent domain queries required")
        if type(domain_capacity) is not int or domain_capacity < 1:
            raise ValueError("Unchanged complete source-domain capacity required")
        sizes = parent_queries.new_full((len(parent_queries), 1), math.log1p(domain_capacity))
        prediction = self._prediction(self.allowed_domain_cardinality(torch.cat((parent_queries, sizes), dim=1)), 1)
        return {name: value.squeeze(1) for name, value in prediction.items()}
