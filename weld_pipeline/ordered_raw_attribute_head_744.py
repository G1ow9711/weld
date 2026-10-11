# Original637 forward SHA256 eae4ed9d2aef2b0ef5765336088fa62ee890c21a2bbab331364823519fa951ef
"""Ordered raw feature bypass with separate whole/local parameters and field policies.

The 637 forward is preserved except additive, initially zero raw readouts.
TRAIN channel statistics are fixed buffers; labels and source IDs are not inputs.
"""
import math
import numpy as np
import torch
from torch import nn
from weld_pipeline.reachable_set_attribute_head_637 import (
    bounded_attribute_features,retained_hypotheses,support_interval,common_value_group,
    PHYSICAL_FIELDS,BOOLEAN_FIELDS,MISSING_WHOLE_FIELDS,NUMERIC_SCALES,HYPOTHESIS_FEATURE_COUNT,
)

VERSION="ORDERED_RAW_ATTRIBUTE_ZERO_READOUT_WITH_METRIC_SUPPORT_V1"


class OrderedRawAdapter(nn.Module):
    def __init__(self,statistics,whole_path):
        super().__init__()
        self.whole_path=bool(whole_path);self.points=17 if whole_path else 1
        assert statistics["channels"]==84 and statistics["statistics_fit_uses_input_features_only"]
        mean=torch.tensor(statistics["mean"],dtype=torch.float32)
        scale=torch.tensor(statistics["safe_scale"],dtype=torch.float32)
        assert mean.shape==scale.shape==(84,) and bool(torch.isfinite(mean).all())
        assert bool(torch.isfinite(scale).all()) and bool((scale>0).all())
        self.register_buffer("input_mean",mean)
        self.register_buffer("input_scale",scale)
        self.input_projection=nn.Linear(self.points*84,128)
        # Fixed dimensional units, without whole-vector norm normalization:
        # candidate span (m), ordered local coordinates (m), path length (m),
        # then support counts and admissible support counts for all16 fields.
        self.metric_projection=nn.Linear(1+self.points*3+1+32,128)
        self.states=nn.Linear(128,80);self.binary=nn.Linear(128,16)
        self.numeric=nn.Linear(128,16);self.no_support=nn.Linear(128,16)
        self.association=nn.Linear(128,16*HYPOTHESIS_FEATURE_COUNT)
        for head in (self.states,self.binary,self.numeric,self.no_support,self.association):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def metric_inputs(self,x,hypotheses):
        # Original84 schema:55/56 candidate distances in m;49:52 centered XYZ/m.
        span=(x[:,:,55]+x[:,:,56]).mean(1,keepdim=True)
        if self.whole_path:
            local=x[:,:,49:52]-x[:,:1,49:52]
            length=(x[:,-1,55]-x[:,0,55]).abs()[:,None]
        else:
            local=x[:,:,49:52]
            length=x.new_zeros((len(x),1))
        support=[]
        for record in hypotheses:
            assert len(record["fields"])==16
            counts=[];valid=[]
            for fi in range(16):
                rows=retained_hypotheses(record,fi)
                counts.append(math.log1p(len(rows)))
                valid.append(math.log1p(sum(bool(r["field_valid"]) and float(r["uncertainty_mm"])<=3. for r in rows)))
            support.append(counts+valid)
        support=x.new_tensor(support).reshape(len(x),32)
        return torch.cat((span,local.reshape(len(x),-1),length,support),-1)

    def forward(self,features,hypotheses):
        assert features.ndim==2 and features.shape[1]==self.points*84
        assert len(features)==len(hypotheses) and features.dtype==torch.float32
        x=features.reshape(-1,self.points,84)
        standardized=((x-self.input_mean)/self.input_scale).reshape(len(x),-1)
        metric=self.metric_inputs(x,hypotheses)
        assert bool(torch.isfinite(standardized).all()) and bool(torch.isfinite(metric).all())
        h=torch.nn.functional.gelu(self.input_projection(standardized)+self.metric_projection(metric))
        return dict(states=self.states(h),binary=self.binary(h),numeric=self.numeric(h),
            no_support=self.no_support(h),
            association=self.association(h).reshape(-1,16,HYPOTHESIS_FEATURE_COUNT))


class RawBoundedAttributeHead(nn.Module):
    def __init__(self,existing,statistics):
        super().__init__()
        self.whole_path=existing.whole_path
        # Preserve Parameter objects, names, values and their existing Adam bindings.
        for name in ("states","numeric","normals","binary","no_support","association"):
            setattr(self,name,getattr(existing,name))
        if hasattr(existing,"reference_module"):self.attach_reference(existing.reference_module)
        self.raw_adapter=OrderedRawAdapter(statistics,self.whole_path)

    def attach_reference(self,reference):
        object.__setattr__(self,"reference_module",reference)

    def forward(self,features,hypotheses):
        assert len(features)==len(hypotheses)
        with torch.no_grad():
            h = self.reference_module.shared(features)
            prior = self.reference_module(features)
        phi = bounded_attribute_features(h)
        raw_path = self.raw_adapter(features,hypotheses)
        raw_states = prior["state_logits"]+self.states(phi).reshape(-1,16,5)+raw_path["states"].reshape(-1,16,5)
        correction = self.numeric(phi)+raw_path["numeric"]
        scalar_logits = prior["scalar_values"]+self.binary(phi)+raw_path["binary"]
        estimates = prior["scalar_values"]+correction*correction.new_tensor(NUMERIC_SCALES)[None]
        no_support = self.no_support(phi)+raw_path["no_support"]
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
                score = score+(torch.tanh(x)*raw_path["association"][ri,fi]).sum(-1)/math.sqrt(HYPOTHESIS_FEATURE_COUNT)
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
                value = corrected[selected]
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
            # An unsupported numeric assertion becomes UNKNOWN. Re-running
            # argmax with just VALUE masked can fabricate NO_OBJECT.
            unresolved_value = (raw_states[:,fi].argmax(-1)==1) & ~association_valid[:,fi]
            unknown = torch.full_like(states[:,fi], -80.)
            unknown[:,0] = 0.
            states[:,fi] = torch.where(unresolved_value[:,None], unknown, states[:,fi])
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

