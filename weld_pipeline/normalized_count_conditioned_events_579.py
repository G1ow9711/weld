"""Bounded new inputs and count/role-conditioned legal path increments.

The frozen325 feature/prior modules retain their exact original forward.
Requested counts cover the complete registered domain; no identity or target
table is an inference input. Training passes this same event module to the
count-aware loss so teacher-count geometry cannot reuse free-count increments.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from weld_pipeline.legacy_legal_allocation_model_518 import (
    LegalCandidateEvents, LegalAllocationContainer,
    legal_positions_for_counts as prior_legal_positions_for_counts,
)
from weld_pipeline.legacy_frozen_incremental_model_392 import tensor_state_digest

VERSION = "U000_BOUNDED_COUNT_ROLE_LEGAL_EVENTS_ONE65_v1"


def bounded_new_features(h):
    """Fixed float32 convention; frozen h itself is not changed."""
    h = h.float()
    length = torch.linalg.vector_norm(h, dim=-1, keepdim=True)
    return torch.cat((h / length.clamp_min(1.), length / (1. + length),
                      torch.ones_like(length)), dim=-1)


class FixedCountRoleFeatures(nn.Module):
    def __init__(self, feature_count, child_capacity, width=128, seed=20261006579):
        super().__init__()
        self.child_capacity = child_capacity
        generator = torch.Generator(device="cpu").manual_seed(seed)
        size = feature_count + child_capacity + 1 + 4
        self.register_buffer("projection", torch.randn(size, width, generator=generator) / math.sqrt(size))
        self.register_buffer("offset", torch.randn(width, generator=generator) / math.sqrt(size))

    def forward(self, phi, counts):
        b, q = counts.shape
        count_features = F.one_hot(counts.long(), self.child_capacity + 1).to(phi.dtype)
        roles = torch.eye(4, device=phi.device, dtype=phi.dtype)
        inputs = torch.cat((phi[:, None, None, :].expand(b, q, 4, -1),
            count_features[:, :, None, :].expand(b, q, 4, -1),
            roles[None, None].expand(b, q, 4, -1)), dim=-1)
        return torch.tanh(inputs @ self.projection + self.offset)


class CountRoleAllocation(nn.Module):
    """Zero readout; count and role enter a fixed bounded nonlinear feature."""
    def __init__(self, q, c, feature_count=130, width=128, seed=20261006579):
        super().__init__()
        self.q, self.c = q, c
        self.fixed = FixedCountRoleFeatures(feature_count, c, width, seed)
        self.weight = nn.Parameter(torch.zeros(q, 2 * c + 1, width))
        self.bias = nn.Parameter(torch.zeros(q, 2 * c + 1))

    def forward(self, phi, counts):
        assert counts.shape == (len(phi), self.q)
        assert bool(((counts >= 0) & (counts <= self.c)).all())
        h = self.fixed(phi, counts)
        by_role = torch.einsum("bqrh,qkh->bqrk", h, self.weight) + self.bias[None, :, None, :]
        # 0=start margin, 1=end margin, 2=length, 3=gap. The end-margin
        # component moves to 2*N; inactive components remain in the full bank.
        index = torch.arange(2 * self.c + 1, device=phi.device)
        role = torch.where(index % 2 == 1, 2, 3).expand(len(phi), self.q, -1).clone()
        role[..., 0] = 0
        role.scatter_(-1, (2 * counts.clamp_min(1)).long()[..., None], 1)
        return by_role.gather(2, role[:, :, None, :])[..., 0, :]


class BoundedTypeIncrement(nn.Module):
    def __init__(self, q, c, width=130, seed=20261006579):
        super().__init__()
        for name, size in (("presence", q), ("count", q * (c + 1)), ("gap", q * (c - 1)),
                           ("direction", q), ("closed", q), ("margin", q * 2)):
            layer = nn.Linear(width, size)
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            setattr(self, name, layer)
        self.allocation = CountRoleAllocation(q, c, width, seed=seed)

    def classification_parameters(self):
        return [p for name in ("presence", "count", "gap", "direction", "closed", "margin")
                for p in getattr(self, name).parameters()]


class NormalizedCountConditionedEvents(LegalCandidateEvents):
    def __init__(self, features, parent_capacity, child_capacity):
        super().__init__(features, parent_capacity, child_capacity)
        self.H = BoundedTypeIncrement(parent_capacity, child_capacity, seed=20261006579)
        self.V = BoundedTypeIncrement(parent_capacity, child_capacity, seed=20261006580)

    def allocation_for_requested_counts(self, output, counts):
        phi, group = output["new_head_phi"], output["allocation_group_ids"]
        if isinstance(counts, int):
            counts = torch.full_like(output["frozen325_counts"], counts, dtype=torch.long)
        a, v = self.H.allocation(phi, counts), self.V.allocation(phi, counts)
        return torch.where((group == 0)[:, None, None], a,
                           torch.where((group == 1)[:, None, None], v, torch.zeros_like(a)))

    def positions_for_requested_counts(self, output, counts):
        requested = dict(output)
        requested["allocation_increment_logits"] = self.allocation_for_requested_counts(output, counts)
        return prior_legal_positions_for_counts(requested, counts)

    def forward_cached(self, h, prior):
        b, q, c = len(h), self.parent_capacity, self.child_capacity
        phi, group = bounded_new_features(h), prior["allocation_group_ids"]

        def inc(name, shape):
            a = getattr(self.H, name)(phi).reshape(shape)
            v = getattr(self.V, name)(phi).reshape(shape)
            mask_shape = (b,) + (1,) * (len(shape) - 1)
            return torch.where((group == 0).reshape(mask_shape), a,
                               torch.where((group == 1).reshape(mask_shape), v, torch.zeros_like(a)))

        out = dict(prior, new_head_phi=phi)
        for name, shape in (("presence", (b,q)), ("count", (b,q,c+1)), ("gap", (b,q,c-1)),
                            ("direction", (b,q)), ("closed", (b,q))):
            out[name+"_logits"] = prior[name+"_logits"] + inc(name, shape)
        out["margin_increment_logits"] = inc("margin", (b,q,2))
        counts = out["count_logits"].argmax(-1)
        out["allocation_increment_logits"] = self.allocation_for_requested_counts(out, counts)
        out["positions_mm"] = prior_legal_positions_for_counts(out, counts)
        out["full_candidate_domain_mm"] = out["allocation_domain_mm"]
        return out


class PhysicalTransactionContainer(LegalAllocationContainer):
    def __init__(self, original115, original_hybrid, config):
        super().__init__(original115, original_hybrid, config)
        self.events = NormalizedCountConditionedEvents(config["candidate_feature_count"],
            config["parent_capacity"], config["child_capacity"])
        self._count_feature_digests = None

    def verify_frozen(self):
        super().verify_frozen()
        if self._count_feature_digests is not None:
            assert {g:tensor_state_digest(getattr(self.events,g).allocation.fixed) for g in ("H","V")} == self._count_feature_digests

    @classmethod
    def from_bundle(cls, bundle):
        config = dict(bundle["config"], physical_transaction_stage=VERSION)
        model = cls(bundle["original115"], bundle["original_hybrid"], config)
        already_new = "physical_transaction_stage" in bundle["config"]
        if not already_new:
            assert bundle["branch_accepted_optimizer_updates"] == 0
            assert bundle["actual_native_optimizer_step_calls_this_stage"] == 0
            assert all(bool((value == 0).all()) for value in bundle["incremental_state"]["events"].values())
        for name,state in bundle["incremental_state"].items():
            if name != "events" or already_new:
                getattr(model,name).load_state_dict(state,strict=True)
        for name,state in bundle["frozen325_reference_state"].items():
            getattr(model,name).load_state_dict(state,strict=True)
            getattr(model,name).eval().requires_grad_(False)
        model.horizontal_residual.eval().requires_grad_(False)
        model.events.attach_reference(model.frozen325_events)
        model.attributes.attach_reference(model.frozen325_attributes)
        model.path_attributes.attach_reference(model.frozen325_path_attributes)
        model._reference_digests = model.reference_digests()
        model._count_feature_digests = {g:tensor_state_digest(getattr(model.events,g).allocation.fixed) for g in ("H","V")}
        model.verify_frozen()
        return model
