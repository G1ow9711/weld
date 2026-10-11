"""Frozen325 prior, independent H/V increments, legal full-count paths.

Every argument at inference comes from the current XYZ factory. A TRAIN
protection cache is never a forward argument or a serialized runtime table.
"""
import torch
from torch import nn
from weld_pipeline.legacy_frozen_incremental_model_392 import (
    CandidateEventResidual, FrozenLegacyIncrementalContainer,
    PointwiseAttributeHead, tensor_state_digest,
)
from weld_pipeline.legacy_incremental_effective_nodes_422 import effective_nodes_torch
from weld_pipeline.finite_plate_attribute_head_523 import FinitePlateAttributeHead

VERSION = "RECOVER325_LEGAL_PATH_FINITE_PLATE_U130_v1"


def add_current_domain(base, sequences):
    """Keep the old float32 prior; separate exact decoder domain in float64."""
    import numpy as np
    result = dict(base)
    result["allocation_domain_mm"] = np.asarray([
        [0., (len(s["xyz"]) - 1) * float(s["sample_step_mm"])]
        for s in sequences], dtype=np.float64)
    result["allocation_group_ids"] = np.asarray([
        2 if s.get("curve_kind") == "arc" else
        0 if s["type"] == "horizontal" else 1 for s in sequences], dtype=np.int64)
    assert np.isfinite(result["allocation_domain_mm"]).all()
    assert (result["allocation_domain_mm"][:, 1] > 0).all()
    return result


def reference_allocation(prior):
    """Extract effective old path, preserving exact shared nodes."""
    p = effective_nodes_torch(prior["positions_mm"].double(), prior["gap_logits"])
    n = prior["count_logits"].argmax(-1)
    b, q, c, _ = p.shape
    index = torch.arange(c, device=p.device)[None, None]
    chosen = index < n[..., None]
    lo, hi = prior["allocation_domain_mm"][:, 0, None], prior["allocation_domain_mm"][:, 1, None]
    end = p[..., 1].gather(-1, (n - 1).clamp_min(0)[..., None])[..., 0]
    valid = (n > 0) & torch.isfinite(p).flatten(-2).all(-1)
    valid &= (((p[..., 0] >= lo[..., None]) & (p[..., 1] <= hi[..., None]) &
               (p[..., 1] > p[..., 0])) | ~chosen).all(-1)
    if c > 1:
        gap = p[..., 1:, 0] - p[..., :-1, 1]
        live = index[..., :-1] < (n - 1)[..., None]
        required = prior["gap_logits"] >= 0.
        valid &= (((gap > 0.) & required) | ((gap == 0.) & ~required) | ~live).all(-1)
    return p, n, valid, torch.stack((p[..., 0, 0] - lo, hi - end), -1)


def margin_logits_for_counts(output, counts):
    same = (counts == output["frozen325_counts"]) & output["reference_path_legal"]
    prior = torch.where(same[..., None] & (output["reference_margins_mm"] > 0.), 1., -1.)
    return prior + output["margin_increment_logits"].double()


def legal_positions_for_counts(output, counts):
    """Full requested n; one float64 cumulative total and actual full span.

    Active lengths/gaps are strictly positive. Inactive margins/shared gaps
    are exact zero. Zero count stays zero for the typed invalid-positive ledger.
    There is no coordinate clamp, sorting, industrial length floor or GT snap.
    """
    old = output["reference_effective_positions_mm"]
    b, q, c, _ = old.shape
    if isinstance(counts, int):
        counts = torch.full((b, q), counts, dtype=torch.long, device=old.device)
    assert counts.shape == (b, q) and bool(((counts >= 0) & (counts <= c)).all())
    n = counts.clamp_min(1)
    index = torch.arange(c, device=old.device)[None, None]
    same = (counts == output["frozen325_counts"]) & output["reference_path_legal"]
    margins = margin_logits_for_counts(output, counts) >= 0.
    positive_gaps = output["gap_logits"] >= 0.
    child_active = index < n[..., None]
    gap_active = (index[..., :-1] < (n - 1)[..., None]) & positive_gaps
    active = torch.zeros((b, q, 2 * c + 1), dtype=torch.bool, device=old.device)
    active[..., 0] = margins[..., 0]
    active[..., 1::2] = child_active
    if c > 1:
        active[..., 2:-1:2] = gap_active
    active.scatter_(-1, (2 * n)[..., None], margins[..., 1, None])
    # A new count uses a neutral whole-domain allocation. A valid same-count
    # path uses every old effective length/gap/margin, never its visible prefix.
    weights = torch.ones((b, q, 2 * c + 1), dtype=torch.float64, device=old.device)
    lengths = old[..., 1] - old[..., 0]
    weights[..., 1::2] = torch.where(same[..., None] & (lengths > 0.), lengths, 1.)
    if c > 1:
        gaps = old[..., 1:, 0] - old[..., :-1, 1]
        weights[..., 2:-1:2] = torch.where(same[..., None] & (gaps > 0.), gaps, 1.)
    weights[..., 0] = torch.where(same & (output["reference_margins_mm"][..., 0] > 0.),
                                output["reference_margins_mm"][..., 0], 1.)
    end_weight = torch.where(same & (output["reference_margins_mm"][..., 1] > 0.),
                             output["reference_margins_mm"][..., 1], 1.)
    weights.scatter_(-1, (2 * n)[..., None], end_weight[..., None])
    z = weights.log() + output["allocation_increment_logits"].double()
    assert bool(torch.isfinite(z).all()), "Nonfinite allocation parameter"
    maximum = z.masked_fill(~active, -torch.inf).max(-1, keepdim=True).values
    positive = torch.exp((z - maximum).masked_fill(~active, -torch.inf))
    assert bool(((positive > 0.) | ~active).all()), "Required length underflow"
    w = torch.where(active, positive, torch.zeros_like(positive))
    cumulative = w.cumsum(-1)
    total = cumulative[..., -1:]
    domain = output["allocation_domain_mm"].double()
    nodes = domain[:, 0, None, None] + (cumulative / total) * (domain[:, 1] - domain[:, 0])[:, None, None]
    starts, ends = nodes[..., 0:-1:2], nodes[..., 1::2]
    required_length = ends - starts
    assert bool(((required_length > 0.) | ~child_active).all()), "Required child lost to float64 rounding"
    if c > 1:
        actual_gap = starts[..., 1:] - ends[..., :-1]
        assert bool(((actual_gap > 0.) | ~gap_active).all()), "Positive gap lost to float64 rounding"
        shared = (index[..., :-1] < (n - 1)[..., None]) & ~positive_gaps
        assert bool(((actual_gap == 0.) | ~shared).all()), "Shared node disagreement"
    positions = torch.stack((starts, ends), -1)
    positions = torch.where(child_active[..., None], positions, torch.zeros_like(positions))
    return torch.where((counts > 0)[..., None, None], positions, torch.zeros_like(positions))


class TypeIncrement(nn.Module):
    def __init__(self, q, c, width=128):
        super().__init__()
        self.presence = nn.Linear(width, q)
        self.count = nn.Linear(width, q * (c + 1))
        self.gap = nn.Linear(width, q * (c - 1))
        self.direction = nn.Linear(width, q)
        self.closed = nn.Linear(width, q)
        self.margin = nn.Linear(width, q * 2)
        self.allocation = nn.Linear(width, q * (2 * c + 1))
        for layer in self.children():
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def classification_parameters(self):
        return [p for name, layer in self.named_children() if name != "allocation" for p in layer.parameters()]


class LegalCandidateEvents(nn.Module):
    def __init__(self, features, parent_capacity, child_capacity):
        super().__init__()
        self.parent_capacity, self.child_capacity = parent_capacity, child_capacity
        self.H = TypeIncrement(parent_capacity, child_capacity)
        self.V = TypeIncrement(parent_capacity, child_capacity)

    def attach_reference(self, reference):
        object.__setattr__(self, "reference_module", reference)

    def current_frozen_features(self, features, base):
        with torch.no_grad():
            prior = self.reference_module(features, base)
            h = self.reference_module.shared(features)
            prior = dict(prior, allocation_domain_mm=base["allocation_domain_mm"],
                         allocation_group_ids=base["allocation_group_ids"])
            p, n, valid, margins = reference_allocation(prior)
            prior.update(reference_effective_positions_mm=p, frozen325_counts=n,
                         reference_path_legal=valid, reference_margins_mm=margins)
        return h, prior

    def forward_cached(self, h, prior):
        """Same forward for frozen current features; cached use is TRAIN-only."""
        b, q, c = len(h), self.parent_capacity, self.child_capacity
        group = prior["allocation_group_ids"]
        def inc(name, shape):
            a = getattr(self.H, name)(h).reshape(shape)
            v = getattr(self.V, name)(h).reshape(shape)
            mshape = (b,) + (1,) * (len(shape) - 1)
            return torch.where((group == 0).reshape(mshape), a,
                               torch.where((group == 1).reshape(mshape), v, torch.zeros_like(a)))
        out = dict(prior)
        for name, shape in (("presence", (b, q)), ("count", (b, q, c + 1)),
                            ("gap", (b, q, c - 1)), ("direction", (b, q)), ("closed", (b, q))):
            out[name + "_logits"] = prior[name + "_logits"] + inc(name, shape)
        out["margin_increment_logits"] = inc("margin", (b, q, 2))
        out["allocation_increment_logits"] = inc("allocation", (b, q, 2 * c + 1))
        out["positions_mm"] = legal_positions_for_counts(out, out["count_logits"].argmax(-1))
        out["full_candidate_domain_mm"] = out["allocation_domain_mm"]
        return out

    def forward(self, features, base):
        h, prior = self.current_frozen_features(features, base)
        return self.forward_cached(h, prior)


class LegalAllocationContainer(FrozenLegacyIncrementalContainer):
    def __init__(self, original115, original_hybrid, config):
        super().__init__(original115, original_hybrid, config)
        self.events = LegalCandidateEvents(config["candidate_feature_count"], config["parent_capacity"], config["child_capacity"])
        self.frozen325_events = CandidateEventResidual(config["candidate_feature_count"], config["parent_capacity"], config["child_capacity"])
        self.frozen325_attributes = PointwiseAttributeHead()
        self.frozen325_path_attributes = PointwiseAttributeHead(features=84 * 17)
        self.attributes = FinitePlateAttributeHead()
        self.path_attributes = FinitePlateAttributeHead(features=84 * 17)
        self._reference_digests = None

    def train(self, mode=True):
        super().train(mode)
        self.frozen325_events.eval()
        self.frozen325_attributes.eval()
        self.frozen325_path_attributes.eval()
        self.horizontal_residual.eval()
        return self

    def reference_digests(self):
        return {name: tensor_state_digest(getattr(self, name)) for name in
                ("frozen325_events", "frozen325_attributes", "frozen325_path_attributes")}

    def verify_frozen(self):
        super().verify_frozen()
        if self._reference_digests is not None:
            assert self.reference_digests() == self._reference_digests
        assert all(not p.requires_grad and p.grad is None for module in
                   (self.frozen325_events, self.frozen325_attributes,
                    self.frozen325_path_attributes, self.horizontal_residual) for p in module.parameters())

    def incremental_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def reference_state_cpu(self):
        return {name: {k: v.detach().cpu() for k, v in getattr(self, name).state_dict().items()}
                for name in ("frozen325_events", "frozen325_attributes", "frozen325_path_attributes")}

    @classmethod
    def from_bundle(cls, bundle):
        config = dict(bundle["config"], legal_allocation_stage=VERSION,
                      pointwise_H_residual_training_enabled=False, classification_increment_gain=1.)
        model = cls(bundle["original115"], bundle["original_hybrid"], config)
        if "legal_allocation_stage" in bundle["config"]:
            for name, state in bundle["incremental_state"].items():
                getattr(model, name).load_state_dict(state, strict=True)
            for name, state in bundle["frozen325_reference_state"].items():
                getattr(model, name).load_state_dict(state, strict=True)
        else:
            assert bundle["actual_optimizer_updates"] == 325
            model.frozen325_events.load_state_dict(bundle["incremental_state"]["events"], strict=True)
            for name in ("attributes", "path_attributes"):
                getattr(model, "frozen325_" + name).load_state_dict(bundle["incremental_state"][name], strict=True)
            model.horizontal_residual.load_state_dict(bundle["incremental_state"]["horizontal_residual"], strict=True)
        for name in ("frozen325_events", "frozen325_attributes", "frozen325_path_attributes"):
            getattr(model, name).eval().requires_grad_(False)
        model.horizontal_residual.eval().requires_grad_(False)
        model.events.attach_reference(model.frozen325_events)
        model.attributes.attach_reference(model.frozen325_attributes)
        model.path_attributes.attach_reference(model.frozen325_path_attributes)
        model._reference_digests = model.reference_digests()
        model.verify_frozen()
        return model
