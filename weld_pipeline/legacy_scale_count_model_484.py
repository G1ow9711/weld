"""Output-preserving325 reference with scaled increments and count-aware paths.

One global function for every current input. Frozen reference is model weights,
not per-case prediction data. A4/E4 publication remains fixed by the predictor.
"""
from copy import deepcopy
import torch
from torch import nn
from weld_pipeline.legacy_frozen_incremental_model_392 import (
    CandidateEventResidual, PointwiseAttributeHead, FrozenLegacyIncrementalContainer,
    tensor_state_digest,
)
from weld_pipeline.contract_field_acceptance_297 import FIELDS
from weld_pipeline.legacy_incremental_metric_context_485 import CONTEXT_COUNT

VERSION = "SCALE_COUNT_PROTECTED325_V1"
LOGIT_INCREMENT_GAIN = 4.
DISTANCE_SCALES = {"front_wall": 50., "back_wall": 50., "start_base_edge": 50.,
                   "end_base_edge": 50., "start_height": 1000., "end_height": 1000.,
                   "start_hole_height": 1000., "end_hole_height": 1000.}
NUMERIC_SCALES = tuple(DISTANCE_SCALES.get(name, 1.) for name in FIELDS)


def positions_for_counts(output, counts):
    """Complete raw path for requested counts; TRAIN or predicted free counts."""
    reference = output["frozen325_positions_mm"]
    delta = output["position_increment_dimensionless"]
    b, q, c, two = reference.shape
    assert two == 2 and delta.shape == reference.shape
    if isinstance(counts, int):
        counts = torch.full((b, q), counts, dtype=torch.long, device=reference.device)
    assert counts.shape == (b, q) and bool(((counts >= 0) & (counts <= c)).all())
    old_count = output["frozen325_counts"]
    domain = output["full_candidate_domain_mm"]
    span = domain[:, 1] - domain[:, 0]
    assert bool((span > 0).all())
    end_index = (old_count - 1).clamp(0, c - 1)[..., None, None].expand(b, q, 1, 2)
    last = reference.gather(2, end_index)[..., 0, 1]
    first = reference[:, :, 0, 0]
    valid = (old_count > 0) & torch.isfinite(first) & torch.isfinite(last) & (last > first)
    valid = valid & (first >= domain[:, 0, None]) & (last <= domain[:, 1, None])
    lo = torch.where(valid, first, domain[:, 0, None])
    hi = torch.where(valid, last, domain[:, 1, None])
    child = torch.arange(c, dtype=reference.dtype, device=reference.device)
    denominator = counts.clamp_min(1).to(reference.dtype)
    left = lo[..., None] + (hi - lo)[..., None] * child / denominator[..., None]
    right = lo[..., None] + (hi - lo)[..., None] * (child + 1.) / denominator[..., None]
    divided = torch.stack((left, right), dim=-1)
    same = (counts == old_count) & (old_count > 0)
    base = torch.where(same[..., None, None], reference, divided)
    # No clamp, sorting, nearest-GT projection, or removal of invalid positives.
    return base + span[:, None, None, None] * delta


class ScaledCandidateEvents(CandidateEventResidual):
    def attach_reference(self, reference):
        # Container registers and freezes this module once. Avoid registering it
        # twice below events or including its parameters in the active optimizer.
        object.__setattr__(self, "reference_module", reference)

    def forward(self, features, base):
        h = self.shared(features)
        reference = self.reference_module
        with torch.no_grad():
            rh = reference.shared(features)
            r_presence = reference.presence(rh)
            r_count = reference.count(rh)
            r_position = reference.positions(rh)
            r_direction = reference.direction(rh)
            r_closed = reference.closed(rh)
            r_gap = reference.gap(rh) if reference.gap is not None else None
        b, q, c = len(features), self.parent_capacity, self.child_capacity
        def classification(layer, old, prior, shape):
            initial = prior + old.reshape(shape)
            return initial + LOGIT_INCREMENT_GAIN * (layer(h) - old).reshape(shape)
        count = classification(self.count, r_count, base["count_logits"], (b, q, c + 1))
        old_count = (base["count_logits"] + r_count.reshape(b, q, c + 1)).argmax(-1)
        output = dict(
            presence_logits=base["presence_logits"] + self.presence(h),
            count_logits=count,
            direction_logits=classification(self.direction, r_direction, base["direction_logits"], (b, q)),
            closed_logits=classification(self.closed, r_closed, base["closed_logits"], (b, q)),
            gap_logits=classification(self.gap, r_gap, base["gap_logits"], (b, q, c - 1)) if c > 1 else base["gap_logits"],
            full_candidate_domain_mm=base["full_candidate_domain_mm"],
            frozen325_positions_mm=base["positions_mm"] + r_position.reshape(b, q, c, 2),
            position_increment_dimensionless=(self.positions(h) - r_position).reshape(b, q, c, 2),
            frozen325_counts=old_count,
        )
        output["positions_mm"] = positions_for_counts(output, count.argmax(-1))
        return output


class ScaledMetricAttributeHead(PointwiseAttributeHead):
    def __init__(self, features=84, width=96):
        super().__init__(features=features, width=width)
        self.metric_projection = nn.Linear(CONTEXT_COUNT, width, bias=False)
        nn.init.zeros_(self.metric_projection.weight)

    def attach_reference(self, reference):
        object.__setattr__(self, "reference_module", reference)

    def forward(self, features, metric_context):
        assert metric_context.shape == (len(features), CONTEXT_COUNT)
        assert bool(torch.isfinite(metric_context).all())
        h = self.shared(features) + self.metric_projection(metric_context)
        with torch.no_grad():
            baseline = self.reference_module(features)
        scalar = self.scalars(h)
        scale = scalar.new_tensor(NUMERIC_SCALES)[None]
        physical = baseline["scalar_values"] + scale * (scalar - baseline["scalar_values"])
        physical = torch.where(scale == 1., scalar, physical)
        raw_state = self.states(h).reshape(-1, 16, 5)
        state = baseline["state_logits"] + LOGIT_INCREMENT_GAIN * (raw_state - baseline["state_logits"])
        raw_normal = self.normals(h).reshape(-1, 2, 3)
        length = torch.linalg.vector_norm(raw_normal, dim=-1, keepdim=True)
        return dict(state_logits=state, scalar_values=physical,
                    numeric_increment_dimensionless=scalar - baseline["scalar_values"],
                    raw_signed_normals=raw_normal, signed_unit_normals=raw_normal / length.clamp_min(1e-8),
                    normal_lengths=length[..., 0],
                    output_scope="CURRENT_METRIC_CONTEXT_ESTIMATE_REQUIRES_WHOLE_CURVE_AND_SIDE_REVIEW")


class ScaleCountProtectedContainer(FrozenLegacyIncrementalContainer):
    def __init__(self, original115, original_hybrid, config):
        super().__init__(original115, original_hybrid, config)
        self.events = ScaledCandidateEvents(config["candidate_feature_count"], config["parent_capacity"], config["child_capacity"])
        self.attributes = ScaledMetricAttributeHead()
        self.path_attributes = ScaledMetricAttributeHead(features=84 * 17)
        self.frozen325_events = CandidateEventResidual(config["candidate_feature_count"], config["parent_capacity"], config["child_capacity"])
        self.frozen325_attributes = PointwiseAttributeHead()
        self.frozen325_path_attributes = PointwiseAttributeHead(features=84 * 17)
        self._reference_digests = None
        self.attach_references()

    def attach_references(self):
        for live, frozen in ((self.events, self.frozen325_events),
                             (self.attributes, self.frozen325_attributes),
                             (self.path_attributes, self.frozen325_path_attributes)):
            frozen.eval().requires_grad_(False)
            live.attach_reference(frozen)
        self.horizontal_residual.eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        for name in ("frozen325_events", "frozen325_attributes", "frozen325_path_attributes", "horizontal_residual"):
            getattr(self, name).eval()
        return self

    def reference_digests(self):
        return {name: tensor_state_digest(getattr(self, name)) for name in (
            "frozen325_events", "frozen325_attributes", "frozen325_path_attributes")}

    def verify_frozen(self):
        super().verify_frozen()
        if self._reference_digests is not None:
            assert self.reference_digests() == self._reference_digests, "Frozen325 reference changed"
        assert all(p.grad is None and not p.requires_grad for name in (
            "frozen325_events", "frozen325_attributes", "frozen325_path_attributes", "horizontal_residual")
            for p in getattr(self, name).parameters())

    @classmethod
    def from_bundle(cls, bundle):
        config = dict(bundle["config"])
        config.update(scale_count_stage=VERSION, scaled_reference_optimizer_updates=325,
                      pointwise_H_residual_training_enabled=False)
        model = cls(bundle["original115"], bundle["original_hybrid"], config)
        for name, state in bundle["incremental_state"].items():
            status = getattr(model, name).load_state_dict(state, strict=False)
            expected = ["metric_projection.weight"] if name in ("attributes", "path_attributes") and "metric_projection.weight" not in state else []
            assert status.missing_keys == expected and not status.unexpected_keys, (name, status)
        if "frozen325_reference_state" in bundle:
            for name, state in bundle["frozen325_reference_state"].items():
                getattr(model, name).load_state_dict(state, strict=True)
        else:
            assert bundle["actual_optimizer_updates"] == 325
            for name in ("events", "attributes", "path_attributes"):
                getattr(model, "frozen325_" + name).load_state_dict(bundle["incremental_state"][name], strict=True)
        model.attach_references()
        model._reference_digests = model.reference_digests()
        model.verify_frozen()
        return model

    def reference_state_cpu(self):
        return {name: {key: value.detach().cpu() for key, value in getattr(self, name).state_dict().items()}
                for name in ("frozen325_events", "frozen325_attributes", "frozen325_path_attributes")}
