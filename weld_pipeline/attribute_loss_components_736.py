"""Exact 637 objective with differentiable per-field diagnostics; no weight changes."""
import math
import torch
from torch.nn import functional as F
from weld_pipeline.reachable_set_attribute_head_637 import PHYSICAL_FIELDS,BOOLEAN_FIELDS,FIELDS,reachable_interval
def decomposed_attribute_loss(output,target,totals,*,frozen_compatible_sets=None):
    components = {}
    losses, coverage, associated = [],{fi:0 for fi in PHYSICAL_FIELDS},{fi:0 for fi in PHYSICAL_FIELDS}
    states = output["training_state_logits_unmasked"]
    compatible_numeric_mask = torch.zeros_like(target["scalar_known"],dtype=torch.bool)
    by_field = {fi:[] for fi in PHYSICAL_FIELDS}
    numeric_by_field = {fi:[] for fi in PHYSICAL_FIELDS}
    compatible_sets = {}
    for row in output["association_rows_TRAIN_differentiable"]:
        fi,ri = row["field"],row["row"]
        if not bool(target["scalar_known"][ri,fi]):
            continue
        value = float(target["scalar_values"][ri,fi])
        compatible = []
        for j,candidate in enumerate(row["candidates"]):
            lo,hi = reachable_interval(candidate)
            role_ok = candidate.get("known_role_constraints_consistent",
                candidate.get("order_status")=="ROLE_UPRIGHT_BASE_GEOMETRIC")
            if role_ok and lo<=value+3. and hi>=value-3.:
                compatible.append(j)
        key = (ri,fi)
        if frozen_compatible_sets is not None:
            assert compatible == frozen_compatible_sets[key], "Physical set changed inside one transaction"
            compatible = frozen_compatible_sets[key]
        compatible_sets[key] = compatible
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
        error = (row["corrected_values"][compatible] - value)/3.
        numeric_by_field[fi].append(F.smooth_l1_loss(error,torch.zeros_like(error),beta=1.,reduction="mean"))
        associated[fi] += 1
    for j,field in enumerate(FIELDS):
        terms = []
        known = target["state_known"][:,j].bool()
        if int(totals["state_known"][j])>0:
            # Global per-source class totals make chunked objectives exact.
            # Caller must supply all known classes, including absent classes.
            class_counts = totals["state_class_counts"][j]
            active_classes = sum(int(n)>0 for n in class_counts)
            for state_class,count in enumerate(class_counts):
                selected = known & (target["state_class"][:,j]==state_class)
                if int(count)>0:
                    terms.append(F.cross_entropy(states[selected,j],target["state_class"][selected,j].long(),reduction="sum")/int(count)/active_classes)
        numeric = target["scalar_known"][:,j].bool()
        if j in PHYSICAL_FIELDS:
            numeric = numeric & compatible_numeric_mask[:,j]
        if int(totals["scalar_known"][j])>0:
            if j in BOOLEAN_FIELDS:
                class_counts = totals["boolean_class_counts"][j]
                active_classes = sum(int(n)>0 for n in class_counts)
                for label,count in enumerate(class_counts):
                    selected = numeric & (target["scalar_values"][:,j]==label)
                    if int(count)>0:
                        terms.append(F.binary_cross_entropy_with_logits(output["scalar_logits"][selected,j],target["scalar_values"][selected,j],reduction="sum")/int(count)/active_classes)
            elif j not in PHYSICAL_FIELDS:
                tolerance = 1. if field.startswith("thickness_") else 3.
                error = (output["scalar_values"][numeric,j]-target["scalar_values"][numeric,j])/tolerance
                terms.append(F.smooth_l1_loss(error,torch.zeros_like(error),beta=1.,reduction="sum")/int(totals["scalar_known"][j]))
            elif numeric_by_field[j]:
                terms.append(torch.stack(numeric_by_field[j]).sum()/int(totals["scalar_known"][j]))
        if j in by_field and by_field[j]:
            terms.append(torch.stack(by_field[j]).sum()/max(1,int(totals["scalar_known"][j])))
        if terms:
            losses.append(sum(terms))
            components["field_%02d"%j] = losses[-1]
    for j in range(2):
        known = target["normal_known"][:,j].bool()
        if int(totals["normal_known"][j])>0:
            error = (output["signed_unit_normals"][known,j]-target["signed_normals"][known,j]).square().sum(-1)
            losses.append(error.sum()/int(totals["normal_known"][j]))
            components["normal_%d"%j] = losses[-1]
    loss = sum(losses)/max(1,len(losses)) if losses else states.sum()*0.
    components = {key:value/max(1,len(losses)) for key,value in components.items()}
    assert bool(torch.isfinite(loss))
    return loss,dict(physical_candidate_coverage_miss=coverage,
        compatible_set_supervised_rows=associated,
        full_scalar_known_denominators=[int(v) for v in totals["scalar_known"]],
        unique_plate_identity_targets_created=0,coverage_miss_not_taught_all_negative=True,
        compatible_sets_TRAIN_only=compatible_sets,
        effective_loss_terms=len(losses),components=components)

