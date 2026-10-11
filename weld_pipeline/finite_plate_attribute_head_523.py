"""Learn a physical plate hypothesis; unresolved association stays unknown."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from weld_pipeline.finite_plate_support_522 import HYPOTHESIS_FEATURE_COUNT
from weld_pipeline.contract_field_acceptance_297 import FIELDS


class FinitePlateAttributeHead(nn.Module):
    def __init__(self, features=84):
        super().__init__()
        self.states = nn.Linear(96, 16 * 5)
        self.numeric = nn.Linear(96, 16)
        self.normals = nn.Linear(96, 6)
        self.association = nn.Sequential(nn.Linear(96 + HYPOTHESIS_FEATURE_COUNT, 32), nn.GELU(), nn.Linear(32, 1))
        for layer in (self.states, self.numeric, self.normals, self.association[-1]):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def attach_reference(self, reference):
        object.__setattr__(self, "reference_module", reference)

    def forward(self, features, hypotheses):
        assert len(features) == len(hypotheses)
        with torch.no_grad():
            h = self.reference_module.shared(features)
            prior = self.reference_module(features)
        raw_states = prior["state_logits"] + self.states(h).reshape(-1, 16, 5)
        correction = self.numeric(h)
        rows, valid_rows, has_rows = [], [], []
        for ri, record in enumerate(hypotheses):
            values, valid, has = [], [], []
            for fi, candidates in enumerate(record["fields"]):
                allowed = [row for row in candidates if row["field_valid"]]
                # Invalid/unresolved hypotheses remain in the public ragged
                # record. They cannot supply an asserted physical field.
                has.append(bool(allowed))
                if not allowed:
                    values.append(correction[ri, fi] * 0.)
                    valid.append(False)
                    continue
                x = features.new_tensor([row["features"] for row in allowed])
                score = self.association(torch.cat((h[ri:ri + 1].expand(len(x), -1), x), -1))[:, 0]
                base = score.new_tensor([row["physical_value_mm"] for row in allowed])
                assert bool((base >= 0.).all()), "Negative physical hypothesis"
                selected = int(score.detach().argmax())
                soft = (score.softmax(0) * base).sum()
                # Numerical output is one actual hypothesis in TRAIN and free
                # inference. Straight-through soft derivative trains ranking;
                # it never emits a mixture of different plate heights.
                hard = base[selected]
                physical = hard + (soft - soft.detach())
                values.append(physical * correction[ri, fi].exp())
                compatible = float((base.max() - base.min()).detach()) <= min(3., min(row["uncertainty_mm"] for row in allowed))
                if len(allowed) == 1:
                    certain = True
                else:
                    best = score.detach().topk(2).values
                    certain = float(best[0] - best[1]) >= math.log(4.)
                valid.append(bool(compatible or certain))
            rows.append(torch.stack(values))
            valid_rows.append(valid); has_rows.append(has)
        scalar = torch.stack(rows) if rows else correction.new_empty((0, 16))
        assert bool(torch.isfinite(scalar).all()), "Physical attribute overflow"
        association_valid = torch.as_tensor(valid_rows, dtype=torch.bool, device=features.device).reshape(-1, 16)
        raw = prior["raw_signed_normals"] + self.normals(h).reshape(-1, 2, 3)
        length = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
        # Runtime state is identical to TRAIN output state; losses can use the
        # unmasked classifier surrogate. No target is a masking/forward input.
        unknown = torch.zeros_like(raw_states); unknown[..., 1:] = -80.
        states = torch.where(association_valid[..., None], raw_states, unknown)
        return dict(state_logits=states, training_state_logits_unmasked=raw_states,
            scalar_values=scalar, numeric_increment_dimensionless=correction,
            association_valid_mask=association_valid,
            association_has_valid_candidates=torch.as_tensor(has_rows, dtype=torch.bool, device=features.device).reshape(-1, 16),
            raw_signed_normals=raw, signed_unit_normals=raw / length.clamp_min(1e-8),
            normal_lengths=length[..., 0], output_scope="FINITE_ASSOCIATION_UNRESOLVED_FIELDS_UNKNOWN",
            runtime_uses_one_physical_hypothesis=True, plate_order_signed_normals_unresolved=True)


def attribute_loss(output, target, totals):
    losses = []
    states = output["training_state_logits_unmasked"]
    for j, field in enumerate(FIELDS):
        terms = []
        known = target["state_known"][:, j].bool()
        if int(totals["state_known"][j]) > 0:
            terms.append(F.cross_entropy(states[known, j], target["state_class"][known, j].long(), reduction="sum") /
                         int(totals["state_known"][j]))
        numeric = target["scalar_known"][:, j].bool()
        if int(totals["scalar_known"][j]) > 0:
            tolerance = 1. if field.startswith("thickness_") else 3.
            error = (output["scalar_values"][numeric, j] - target["scalar_values"][numeric, j]) / tolerance
            terms.append(F.smooth_l1_loss(error, torch.zeros_like(error), beta=1., reduction="sum") /
                         int(totals["scalar_known"][j]))
        if terms:
            losses.append(sum(terms))
    for j in range(2):
        known = target["normal_known"][:, j].bool()
        if int(totals["normal_known"][j]) > 0:
            # Available local signed labels remain local supervised records.
            # Runtime does not assert export orientation without association.
            wanted = target["signed_normals"][known, j]
            error = (output["signed_unit_normals"][known, j] - wanted).square().sum(-1)
            losses.append(error.sum() / int(totals["normal_known"][j]))
    return sum(losses) / max(1, len(losses)) if losses else states.sum() * 0.
