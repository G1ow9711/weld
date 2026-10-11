"""Corrective loss on decoded shared nodes and full-G child error costs."""
import numpy as np
import torch
from torch.nn import functional as F
from weld_pipeline.legacy_incremental_loss_396 import event_loss as original_event_loss,positive_negative_mean,world_positions
from weld_pipeline.legacy_incremental_effective_nodes_422 import effective_nodes_torch


def presence_targets(output,parents,background_known,assignments):
    presence=output['presence_logits'];yy=torch.zeros_like(presence)
    known=background_known.to(device=presence.device,dtype=torch.bool)[:,None].expand_as(presence).clone()
    children=torch.zeros_like(presence)
    for item in assignments:
        si,qi=item['candidate'],item['predicted_parent_query']
        yy[si,qi]=1.;known[si,qi]=True;children[si,qi]=item['original_child_count']
    return yy,known,children


def corrective_presence(output,yy,known,positive_children,hard_negative_weights,original_G):
    z=output['presence_logits'];counts=output['count_logits'].detach().argmax(-1).clamp_min(1).to(z.dtype)
    hard=hard_negative_weights.to(device=z.device,dtype=z.dtype)
    assert hard.shape==z.shape and original_G>0 and torch.all((hard==1.)|(hard==4.))
    positive=known&(yy>.5);negative=known&~positive
    value=(F.softplus(-z[positive])*positive_children[positive]).sum()
    value=value+(F.softplus(z[negative])*hard[negative]*counts[negative]).sum()
    return value/float(original_G),dict(
        background_expected_child_mass=(counts[negative]*z[negative].sigmoid()).sum(),
        positive_missing_expected_child_mass=(positive_children[positive]*(1.-z[positive].sigmoid())).sum(),
        confirmed_hard_negative_query_cells=int((negative&(hard==4.)).sum()),
        positive_query_cells=int(positive.sum()),known_negative_query_cells=int(negative.sum()))


def event_loss(output,parents_by_candidate,base,background_known,group_ids,hard_negative_weights,original_G):
    effective=dict(output,positions_mm=effective_nodes_torch(output['positions_mm'],output['gap_logits']))
    total,detail=original_event_loss(effective,parents_by_candidate,base,background_known,group_ids)
    # Unselected future child slots must not make sharing add a huge penalty
    # to the actual first-child endpoint. Regularize the learned raw offsets.
    raw_regularization=(output['positions_mm']-base['positions_mm']).square().mean()
    total=total+.001*(raw_regularization-detail['residual_regularization'])
    detail['residual_regularization']=raw_regularization
    yy,known,children=presence_targets(output,parents_by_candidate,background_known,detail['assignments'])
    binary=F.binary_cross_entropy_with_logits(output['presence_logits'],yy,reduction='none')
    parts=[]
    for group in torch.unique(group_ids).tolist():
        selected=(group_ids==group)[:,None].expand_as(known)&known
        parts.append(positive_negative_mean(binary,yy,selected))
    old_presence=torch.stack(parts).mean() if parts else binary.sum()*0.
    new_presence,costs=corrective_presence(output,yy,known,children,hard_negative_weights,original_G)
    total=total-old_presence+new_presence
    keep_terms=[];maximum_terms=[];failures=dict(count=0,gap=0,endpoint_visible_prefix=0,positive_out_of_domain=0)
    for assigned in detail['assignments']:
        si,qi=assigned['candidate'],assigned['predicted_parent_query'];n=assigned['original_child_count']
        record=next(r for r in parents_by_candidate[si] if r['original_parent_index']==assigned['original_parent_index'])
        got=world_positions(effective['positions_mm'][si,qi,:n],record['candidate_geometry'])
        mask=record.get('corrective_keep_child_mask')
        if mask is not None and any(mask):
            chosen=torch.as_tensor(mask,dtype=torch.bool,device=got.device)
            ref=torch.as_tensor(record['corrective_keep_world_endpoints_mm'],dtype=got.dtype,device=got.device)
            assert torch.isfinite(ref[chosen]).all()
            keep_terms.append(F.smooth_l1_loss(got[chosen],ref[chosen],beta=1.))
        predicted_count=int(output['count_logits'][si,qi].detach().argmax())
        failures['count']+=int(predicted_count!=n)
        visible=min(predicted_count,n)
        truth=torch.as_tensor(record['canonical_original_world_endpoints_mm'],dtype=got.dtype,device=got.device)
        if visible:
            error=torch.linalg.vector_norm(got[:visible]-truth[:visible],dim=-1)
            maximum_terms.append(F.relu(error.max()-3.))
            failures['endpoint_visible_prefix']+=int(float(error.detach().max())>3.)
        if predicted_count==n and n>1:
            wanted=record['original_gap_status']
            if record['parent_chart_direction']==-1:wanted=wanted[::-1]
            actual=(output['gap_logits'][si,qi,:n-1].detach()>=0.).cpu().tolist()
            failures['gap']+=int(actual!=[s=='POSITIVE_GAP' for s in wanted])
        domain=output['full_candidate_domain_mm'][si]
        actual_positions=effective['positions_mm'][si,qi,:predicted_count]
        failures['positive_out_of_domain']+=int(bool(((actual_positions<domain[0])|(actual_positions>domain[1])).any()))
    zero=output['presence_logits'].sum()*0.
    keep=torch.stack(keep_terms).mean() if keep_terms else zero
    maximum=torch.stack(maximum_terms).mean() if maximum_terms else zero
    total=total+keep+maximum
    assert torch.isfinite(total)
    detail.update(presence_original_balanced=old_presence,presence_child_error_cost=new_presence,
        correct_current_V_world_keep=keep,visible_prefix_endpoint_maximum_hinge_mm=maximum,
        **costs,training_correspondence_failure_counts=failures,
        geometry_uses_effective_shared_nodes=True,
        count_mismatch_geometry_has_explicit_teacher_forced_extension=True,
        diagnostics_are_training_correspondences_not_acceptance=True)
    detail['event']=detail['event']-old_presence+new_presence
    return total,detail
