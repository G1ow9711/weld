"""TRAIN-only event and graded attribute losses. No free-runtime import.

Assignment is a training operation; truth never chooses runtime parent
queries, direction, count or coordinates. NaN denotes unknown numeric labels
and is indexed out before arithmetic, never replaced by a zero target.
"""
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from weld_pipeline.contract_field_acceptance_297 import FIELDS
from weld_pipeline.legacy_incremental_correspondence_414 import target_variants

VERSION='SOURCE_BALANCED_FROZEN_PARENT_INCREMENTAL_LOSS_v1'

def canonical_target(record):
 p=np.asarray(record['child_parameter_endpoints_mm'],float)
 assert p.shape==(record['original_child_count'],2) and np.isfinite(p).all()
 negative=record['parent_chart_direction']==-1
 # Store orientation explicitly. Inversion is only a parameter-chart change;
 # the free decoder restores the predicted signed order before evaluation.
 return p[::-1,::-1].copy() if negative else p.copy(),negative

def world_positions(position,geometry):
 if geometry['kind']=='line':
  origin=torch.as_tensor(geometry['origin_mm'],dtype=position.dtype,device=position.device)
  direction=torch.as_tensor(geometry['direction'],dtype=position.dtype,device=position.device)
  return origin+position[...,None]*direction
 curve=geometry['curve'];angle=curve['sequence_start_angle_rad']+position/curve['radius_mm']
 return torch.stack((curve['center_xy_mm'][0]+curve['radius_mm']*torch.cos(angle),
  curve['center_xy_mm'][1]+curve['radius_mm']*torch.sin(angle),torch.zeros_like(angle)+curve['height_mm']),-1)

def positive_negative_mean(values,target,known):
 terms=[]
 for positive in [False,True]:
  selected=known & ((target>.5)==positive)
  if bool(selected.any()):terms.append(values[selected].mean())
 return torch.stack(terms).mean() if terms else values.sum()*0.

def event_loss(output,parents_by_candidate,base,background_known=None,group_ids=None):
 presence=output['presence_logits'];positions=output['positions_mm'];counts=output['count_logits']
 b,q=presence.shape;c=positions.shape[2];device=presence.device
 assert len(parents_by_candidate)==b
 yy=torch.zeros_like(presence);known=torch.ones_like(presence,dtype=torch.bool)
 if background_known is not None:
  known=background_known.to(device=device,dtype=torch.bool)[:,None].expand(b,q).clone()
 count_terms=[];geometry_terms=[];relation_terms=[];physical_terms=[];keep_terms=[];assignments=[]
 for si,records in enumerate(parents_by_candidate):
  assert len(records)<=q,'Global parent capacity omits registered supervision'
  if not records:continue
  options=[list(target_variants(r)) for r in records]
  assert all(options) and all(r['original_child_count']<=c for r in records),'Global child capacity omits original supervision'
  candidate=positions[si].detach().float().cpu().numpy()
  count_log=counts[si].detach().float().log_softmax(-1).cpu().numpy()
  world_cost={}
  for ti,record in enumerate(records):
   if 'candidate_geometry' not in record:continue
   n=record['original_child_count'];points=world_positions(positions[si,:,:n].detach(),record['candidate_geometry'])
   wanted=torch.as_tensor(record['canonical_original_world_endpoints_mm'],dtype=points.dtype,device=device)
   world_cost[ti]=torch.linalg.vector_norm(points-wanted[None],dim=-1).mean((-1,-2)).cpu().numpy()
  cost=np.zeros((q,len(records)));chosen={}
  for qi in range(q):
   for ti,views in enumerate(options):
    bundles=[]
    for view in views:
     p,negative=canonical_target(view);n=len(p)
     distance=float(np.abs(candidate[qi,:n]-p).mean())
     if ti in world_cost:distance=float(world_cost[ti][qi])
     # Absolute GT traversal is not a target for an unconfirmed/no-direction path.
     direction_cost=0.
     if view.get('direction_semantics_confirmed') is True:
      direction_cost=float(F.binary_cross_entropy_with_logits(output['direction_logits'][si,qi].detach(),torch.tensor(0. if negative else 1.,device=device)).cpu())
     bundles.append(distance-float(count_log[qi,n])+direction_cost)
    best=int(np.argmin(bundles));cost[qi,ti]=bundles[best];chosen[(qi,ti)]=views[best]
  assert np.isfinite(cost).all()
  qq,tt=linear_sum_assignment(cost)
  for qi,ti in zip(qq.tolist(),tt.tolist()):
   record=chosen[(qi,ti)];p,negative=canonical_target(record);n=len(p)
   yy[si,qi]=1.;known[si,qi]=True
   wanted=torch.as_tensor(p,dtype=positions.dtype,device=device)
   got=positions[si,qi,:n]
   if 'candidate_geometry' in record:
    got_world=world_positions(got,record['candidate_geometry'])
    wanted_world=torch.as_tensor(record['canonical_original_world_endpoints_mm'],dtype=got.dtype,device=device)
    error=torch.linalg.vector_norm(got_world-wanted_world,dim=-1)
    geometry_terms.append(F.smooth_l1_loss(error,torch.zeros_like(error),beta=1.,reduction='mean'))
   else:geometry_terms.append(F.smooth_l1_loss(got,wanted,beta=1.,reduction='mean'))
   count_terms.append(F.cross_entropy(counts[si,qi:qi+1],torch.tensor([n],device=device)))
   if record.get('direction_semantics_confirmed') is True:
    relation_terms.append(F.binary_cross_entropy_with_logits(output['direction_logits'][si,qi],
     torch.tensor(0. if negative else 1.,device=device)))
   relation_terms.append(F.binary_cross_entropy_with_logits(output['closed_logits'][si,qi],
    torch.tensor(float(record['parent_is_closed']),device=device)))
   gaps=record['original_gap_status'][::-1] if negative else record['original_gap_status']
   if n>1:
    flags=torch.tensor([v=='POSITIVE_GAP' for v in gaps],dtype=positions.dtype,device=device)
    relation_terms.append(F.binary_cross_entropy_with_logits(output['gap_logits'][si,qi,:n-1],flags))
   domain=output['full_candidate_domain_mm'][si]
   physics=F.relu(domain[0]-got).mean()+F.relu(got-domain[1]).mean()+F.relu(1e-6+got[:,0]-got[:,1]).mean()
   if n>1:physics=physics+F.relu(got[:-1,1]-got[1:,0]).mean()
   physical_terms.append(physics)
   keep=record.get('teacher_correct_child_mask')
   if keep is not None:
    keep=list(keep)[::-1] if records[ti]['parent_chart_direction']==-1 else list(keep)
    mask=torch.tensor(keep,dtype=torch.bool,device=device)
    if bool(mask.any()):
     old=base['positions_mm'][si,record.get('teacher_correct_query_index',qi),:n]
     keep_terms.append(F.smooth_l1_loss(got[mask],old[mask],beta=1.,reduction='mean'))
   assignments.append(dict(candidate=si,predicted_parent_query=qi,original_parent_index=record['original_parent_index'],
    original_child_count=n,original_chart_direction=record['parent_chart_direction']))
 zero=presence.sum()*0.
 mean=lambda values:torch.stack(values).mean() if values else zero
 binary=F.binary_cross_entropy_with_logits(presence,yy,reduction='none')
 if group_ids is None:event=positive_negative_mean(binary,yy,known)
 else:
  group_terms=[]
  for group in torch.unique(group_ids).tolist():
   selected=(group_ids==group)[:,None].expand(b,q)&known
   group_terms.append(positive_negative_mean(binary,yy,selected))
  event=torch.stack(group_terms).mean() if group_terms else zero
 event=event+mean(count_terms)
 geometry=mean(geometry_terms);relation=mean(relation_terms);physical=mean(physical_terms);keep=mean(keep_terms)
 residual=(positions-base['positions_mm']).square().mean()
 total=event+2.*geometry+relation+keep+.1*physical+.001*residual
 assert torch.isfinite(total),'Nonfinite incremental source loss'
 return total,dict(event=event,geometry_mm=geometry,relation=relation,known_correct_teacher_keep=keep,
  physical=physical,residual_regularization=residual,assignments=assignments,
  original_positive_parents=sum(len(p) for p in parents_by_candidate),known_parent_presence_cells=int(known.sum()))

def graded_attribute_loss(output,target):
 states=output['state_logits'];known=target['state_known'].to(dtype=torch.bool,device=states.device)
 yy=target['state_class'].to(dtype=torch.long,device=states.device)
 assert states.shape[:2]==known.shape==yy.shape and states.shape[1]==16
 weights=target.get('grade_weight',torch.ones_like(yy,dtype=states.dtype)).to(states.device)
 terms=[];counts={}
 for field,name in enumerate(FIELDS):
  valid=known[:,field];counts[name]=int(valid.sum())
  if bool(valid.any()):
   loss=F.cross_entropy(states[valid,field],yy[valid,field],reduction='none')
   terms.append((loss*weights[valid,field]).sum()/weights[valid,field].sum().clamp_min(1e-8))
 numeric=target['scalar_values'].to(states.device);numeric_known=target['scalar_known'].to(dtype=torch.bool,device=states.device)
 for field,name in enumerate(FIELDS):
  valid=numeric_known[:,field]
  if not bool(valid.any()):continue
  wanted=numeric[valid,field];assert torch.isfinite(wanted).all()
  got=output['scalar_values'][valid,field]
  if name in ('has_eave','side_plate_split'):loss=F.binary_cross_entropy_with_logits(got,wanted,reduction='none')
  else:
   assert name not in ('normal_1','normal_2','hole_edge','segment_reason'),'Unfrozen or vector field cannot use a numeric scalar surrogate'
   loss=F.smooth_l1_loss(got,wanted,beta=1.,reduction='none')
  terms.append((loss*weights[valid,field]).sum()/weights[valid,field].sum().clamp_min(1e-8))
 wanted_normals=target['signed_normals'].to(states.device)
 normal_known=target['normal_known'].to(dtype=torch.bool,device=states.device)
 for side in range(2):
  valid=normal_known[:,side]
  if not bool(valid.any()):continue
  wanted=wanted_normals[valid,side];assert torch.isfinite(wanted).all()
  assert torch.all((torch.linalg.vector_norm(wanted,dim=-1)-1.).abs()<=.001)
  raw=output['raw_signed_normals'][valid,side]
  unit=output['signed_unit_normals'][valid,side]
  # Signed angle surrogate. No abs(dot), no board swap, no best permutation.
  # At zero initialization the normalization derivative is enormous. The
  # signed raw-vector anchor teaches the direction until its norm is usable;
  # validity thresholds for exported normals remain unchanged.
  usable=torch.linalg.vector_norm(raw,dim=-1)>=1e-3
  direction=torch.where(usable,1.-(unit*wanted).sum(-1).clamp(-1.,1.),torch.zeros(len(raw),device=raw.device,dtype=raw.dtype))
  scale=(torch.linalg.vector_norm(raw,dim=-1)-1.).square()
  anchor=F.smooth_l1_loss(raw,wanted,reduction='none',beta=1.).mean(-1)
  terms.append((direction+.1*scale+anchor).mean())
 total=torch.stack(terms).mean() if terms else states.sum()*0.
 assert torch.isfinite(total),'Unknown attribute payload entered arithmetic'
 return total,dict(known_state_records_by_field=counts,
  retained_original_E16_records=int(known.numel()),known_scalar_records=int(numeric_known.sum()),
  known_signed_normal_records=int(normal_known.sum()),whole_customer_approved_records=0,
  unknown_numeric_payloads_not_zero_targets=True)
