"""Separate field policies, set association and bounded physical corrections.

Sampling uncertainty is an estimated support budget, not a certified accuracy.
Whole-path missing fields remain unknown; local labels stay local. TRAIN labels
enter attribute_loss only, never the forward or its hypothesis construction.
"""
import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from weld_pipeline.contract_field_acceptance_297 import FIELDS
from weld_pipeline.finite_plate_support_522 import HYPOTHESIS_FEATURE_COUNT
from weld_pipeline.legacy_scale_count_model_484 import NUMERIC_SCALES

VERSION = "BOUNDED_PHYSICAL_COMPATIBLE_SET_AND_FIELD_POLICY_v1"
PHYSICAL_FIELDS = (6, 7, 8, 9)
BOOLEAN_FIELDS = (12, 14)
MISSING_WHOLE_FIELDS = (0, 1, 2, 3, 13, 15)


def bounded_attribute_features(h):
    h = h.float()
    length = torch.linalg.vector_norm(h,dim=-1,keepdim=True)
    return torch.cat((h/length.clamp_min(1.),length/(1.+length),torch.ones_like(length)),-1)


def retained_hypotheses(record, field):
    """Deduplicate proven identical physical sources; values alone prove none."""
    result, seen = [], set()
    for row in record["fields"][field]:
        key = row.get("physical_source_key")
        if key is not None:
            assert isinstance(key,str) and key
            if key in seen:
                continue
            seen.add(key)
        result.append(row)
    return result


def support_interval(row):
    value, uncertainty = float(row["physical_value_mm"]), float(row["uncertainty_mm"])
    interval = row.get("support_interval_mm",[max(0.,value-uncertainty),value+uncertainty])
    lo, hi = map(float,interval)
    assert all(math.isfinite(v) for v in (value,uncertainty,lo,hi))
    assert value>=0. and uncertainty>=0. and lo<=value<=hi
    return value, uncertainty, lo, hi


def common_value_group(values, intervals, selected, tolerance=3.):
    """Every member contains the emitted anchor, with a common interval.

    No transitive proximity merge. A 100/103/106 chain cannot absorb the whole
    chain around the selected 100 merely because consecutive pairs are close.
    """
    anchor = float(values[selected])
    result = [j for j,(v,(lo,hi)) in enumerate(zip(values,intervals))
              if lo<=anchor<=hi and abs(float(v)-anchor)<=tolerance]
    assert selected in result
    assert max(intervals[j][0] for j in result)<=min(intervals[j][1] for j in result)
    return result


class BoundedSetAttributeHead(nn.Module):
    def __init__(self,features=84,*,whole_path=False):
        super().__init__()
        self.whole_path = bool(whole_path)
        self.states = nn.Linear(98,16*5)
        self.numeric = nn.Linear(98,16)
        self.normals = nn.Linear(98,6)
        self.binary = nn.Linear(98,16)
        self.no_support = nn.Linear(98,16)
        self.association = nn.Sequential(nn.Linear(98+HYPOTHESIS_FEATURE_COUNT,32),nn.GELU(),nn.Linear(32,1))
        for layer in (self.states,self.numeric,self.normals,self.binary,self.no_support,self.association[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def attach_reference(self,reference):
        object.__setattr__(self,"reference_module",reference)

    def forward(self,features,hypotheses):
        assert len(features)==len(hypotheses)
        with torch.no_grad():
            h = self.reference_module.shared(features)
            prior = self.reference_module(features)
        phi = bounded_attribute_features(h)
        raw_states = prior["state_logits"]+self.states(phi).reshape(-1,16,5)
        correction = self.numeric(phi)
        scalar_logits = prior["scalar_values"]+self.binary(phi)
        estimates = prior["scalar_values"]+correction*correction.new_tensor(NUMERIC_SCALES)[None]
        no_support = self.no_support(phi)
        rows, valid_rows, has_rows, reasons, ragged = [],[],[],[],[]
        for ri,record in enumerate(hypotheses):
            assert len(record["fields"])==16
            values, valid, has, why = [],[],[],[]
            for fi in range(16):
                if fi not in PHYSICAL_FIELDS:
                    # Existing decode_fields consumes a SIGNED boolean logit.
                    # Passing 0/1 here would incorrectly decode both as true.
                    value = scalar_logits[ri,fi] if fi in BOOLEAN_FIELDS else estimates[ri,fi]
                    values.append(value)
                    enabled = not(self.whole_path and fi in MISSING_WHOLE_FIELDS)
                    valid.append(enabled)
                    has.append(False)
                    why.append("MISSING_WHOLE_TRUTH" if not enabled else "MODEL_ESTIMATE")
                    continue
                all_rows = retained_hypotheses(record,fi)
                allowed = [row for row in all_rows if row["field_valid"] and
                           float(row["uncertainty_mm"])<=3.]
                has.append(bool(allowed))
                if not allowed:
                    values.append(correction[ri,fi]*0.)
                    valid.append(False)
                    why.append("NO_CANDIDATES" if not all_rows else "CANDIDATE_SUPPORT_INSUFFICIENT")
                    ragged.append(dict(row=ri,field=fi,scores=no_support[ri,fi:fi+1],
                        real_candidate_count=0,candidates=[],coverage_miss_possible=True))
                    continue
                packed = np.ascontiguousarray(np.stack([row["features"] for row in allowed]), dtype=np.float32)
                assert features.dtype == torch.float32
                x = torch.as_tensor(packed, dtype=features.dtype, device=features.device)
                score = self.association(torch.cat((phi[ri:ri+1].expand(len(x),-1),torch.tanh(x)),-1))[:,0]
                supports = [support_interval(row) for row in allowed]
                base = score.new_tensor([r[0] for r in supports])
                uncertainty = score.new_tensor([r[1] for r in supports])
                lo,hi = score.new_tensor([r[2] for r in supports]),score.new_tensor([r[3] for r in supports])
                remaining = (3.-uncertainty).clamp_min(0.)
                positive = torch.minimum(hi-base,remaining)
                negative = torch.minimum(base-lo,remaining)
                r = correction[ri,fi]
                delta = torch.where(r>=0.,positive,negative)*torch.tanh(r)
                corrected = base+delta
                post_budget = uncertainty+delta.abs()
                post_valid = (corrected>=lo)&(corrected<=hi)&(post_budget<=3.)&torch.isfinite(corrected)
                selected = int(score.detach().argmax())
                soft = (score.softmax(0)*corrected).sum()
                value = corrected[selected]+(soft-soft.detach())
                group = common_value_group(corrected.detach().cpu().tolist(),
                    [(r[2],r[3]) for r in supports],selected)
                all_scores = torch.cat((score,no_support[ri,fi:fi+1]))
                mass = float(all_scores.detach().softmax(0)[group].sum())
                scalar_valid = bool(post_valid[selected]) and mass>=.8
                values.append(value)
                valid.append(scalar_valid)
                why.append(("SCALAR_CONSISTENT_PLATE_IDENTITY_UNCONFIRMED" if scalar_valid else
                    "CORRECTION_EXCEEDS_SUPPORT_BUDGET" if not bool(post_valid[selected]) else
                    "SCALAR_VALUE_AMBIGUOUS" if len(group)<len(allowed) else "ASSOCIATION_CONFIDENCE_INSUFFICIENT"))
                ragged.append(dict(row=ri,field=fi,scores=all_scores,candidates=allowed,
                    real_candidate_count=len(allowed),selected=selected,value_group=group,
                    scalar_group_probability=mass,physical_plate_identity_confirmed=False,
                    corrected_values=corrected,post_valid=post_valid,post_budget_mm=post_budget,
                    sampling_uncertainty_is_not_a_certified_error_bound=True))
            rows.append(torch.stack(values))
            valid_rows.append(valid)
            has_rows.append(has)
            reasons.append(why)
        scalar = torch.stack(rows) if rows else estimates.new_empty((0,16))
        assert bool(torch.isfinite(scalar).all())
        association_valid = torch.as_tensor(valid_rows,dtype=torch.bool,device=features.device).reshape(-1,16)
        states = raw_states.clone()
        for fi in PHYSICAL_FIELDS:
            # Special state encodings remain in their classifier. Only a
            # numeric physical assertion requires support and association.
            states[:,fi,1] = torch.where(association_valid[:,fi],states[:,fi,1],states[:,fi,1]*0.-80.)
        if self.whole_path:
            for fi in MISSING_WHOLE_FIELDS:
                states[:,fi] = states[:,fi]*0.-80.
                states[:,fi,0] = 0.
        raw = prior["raw_signed_normals"]+self.normals(phi).reshape(-1,2,3)
        length = torch.linalg.vector_norm(raw,dim=-1,keepdim=True)
        return dict(state_logits=states,training_state_logits_unmasked=raw_states,
            scalar_values=scalar,scalar_logits=scalar_logits,
            scalar_values_boolean_slots_are_logits=True,
            numeric_increment_dimensionless=correction,association_valid_mask=association_valid,
            association_has_valid_candidates=torch.as_tensor(has_rows,dtype=torch.bool,device=features.device).reshape(-1,16),
            association_reasons=reasons,association_rows_TRAIN_differentiable=ragged,
            raw_signed_normals=raw,signed_unit_normals=raw/length.clamp_min(1e-8),normal_lengths=length[...,0],
            output_scope="WHOLE_PATH_FIELD_POLICY_MODEL_ESTIMATES_AND_PHYSICAL_SUPPORT" if self.whole_path else "LOCAL_POINTWISE_MODEL_ESTIMATE_NOT_WHOLE_TRUTH",
            missing_whole_truth_fields_retained_unknown=self.whole_path,
            runtime_uses_one_physical_hypothesis=True,plate_order_signed_normals_unresolved=True,
            physical_corrections_bounded_and_postchecked=True,
            estimated_sampling_budget_not_formal_accuracy_certificate=True)


def attribute_loss(output,target,totals):
    losses, coverage, associated = [],{fi:0 for fi in PHYSICAL_FIELDS},{fi:0 for fi in PHYSICAL_FIELDS}
    states = output["training_state_logits_unmasked"]
    compatible_numeric_mask = torch.zeros_like(target["scalar_known"],dtype=torch.bool)
    by_field = {fi:[] for fi in PHYSICAL_FIELDS}
    for row in output["association_rows_TRAIN_differentiable"]:
        fi,ri = row["field"],row["row"]
        if not bool(target["scalar_known"][ri,fi]):
            continue
        value = float(target["scalar_values"][ri,fi])
        compatible = []
        for j,candidate in enumerate(row["candidates"]):
            _,_,lo,hi = support_interval(candidate)
            role_ok = candidate.get("known_role_constraints_consistent",
                candidate.get("order_status")=="ROLE_UPRIGHT_BASE_GEOMETRIC")
            if role_ok and lo<=value+3. and hi>=value-3.:
                compatible.append(j)
        if not compatible:
            # Known truth stays in full denominators, without false negative
            # labels for every candidate or a fabricated physical target.
            coverage[fi] += 1
            continue
        compatible_numeric_mask[ri,fi] = True
        scores = row["scores"]
        remainder = [j for j in range(len(scores)) if j not in compatible]
        positive = torch.logsumexp(scores[compatible],0)
        others = torch.logsumexp(scores[remainder],0)
        term = torch.logsumexp(scores,0)-positive+F.relu(scores.new_tensor(math.log(4.))-(positive-others))
        by_field[fi].append(term)
        associated[fi] += 1
    for j,field in enumerate(FIELDS):
        terms = []
        known = target["state_known"][:,j].bool()
        if int(totals["state_known"][j])>0:
            terms.append(F.cross_entropy(states[known,j],target["state_class"][known,j].long(),reduction="sum")/int(totals["state_known"][j]))
        numeric = target["scalar_known"][:,j].bool()
        if j in PHYSICAL_FIELDS:
            numeric = numeric & compatible_numeric_mask[:,j]
        if int(totals["scalar_known"][j])>0:
            if j in BOOLEAN_FIELDS:
                terms.append(F.binary_cross_entropy_with_logits(output["scalar_logits"][numeric,j],target["scalar_values"][numeric,j],reduction="sum")/int(totals["scalar_known"][j]))
            else:
                tolerance = 1. if field.startswith("thickness_") else 3.
                error = (output["scalar_values"][numeric,j]-target["scalar_values"][numeric,j])/tolerance
                terms.append(F.smooth_l1_loss(error,torch.zeros_like(error),beta=1.,reduction="sum")/int(totals["scalar_known"][j]))
        if j in by_field and by_field[j]:
            terms.append(torch.stack(by_field[j]).sum()/max(1,int(totals["scalar_known"][j])))
        if terms:
            losses.append(sum(terms))
    for j in range(2):
        known = target["normal_known"][:,j].bool()
        if int(totals["normal_known"][j])>0:
            error = (output["signed_unit_normals"][known,j]-target["signed_normals"][known,j]).square().sum(-1)
            losses.append(error.sum()/int(totals["normal_known"][j]))
    loss = sum(losses)/max(1,len(losses)) if losses else states.sum()*0.
    assert bool(torch.isfinite(loss))
    return loss,dict(physical_candidate_coverage_miss=coverage,
        compatible_set_supervised_rows=associated,
        full_scalar_known_denominators=[int(v) for v in totals["scalar_known"]],
        unique_plate_identity_targets_created=0,coverage_miss_not_taught_all_negative=True,
        effective_loss_terms=len(losses))


def legal_attribute_output_check(output):
    finite = all(bool(torch.isfinite(output[key]).all()) for key in
        ("state_logits","scalar_values","scalar_logits","raw_signed_normals","signed_unit_normals"))
    bounded = all(bool(row["post_valid"].all()) for row in output["association_rows_TRAIN_differentiable"]
                  if row["real_candidate_count"])
    return dict(pass_all=finite and bounded,finite=finite,all_physical_corrections_in_support_and_budget=bounded,
                sampling_budget_not_formal_error_certificate=True)

