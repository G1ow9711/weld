"""Incremental event/relation/attribute heads around immutable old components.

This is a new candidate, not a promoted checkpoint. The global publication
switches must be registered once for every input, never selected by its ID.
"""
import hashlib,pickle
import torch
from torch import nn

VERSION='FROZEN_LEGACY_COMPONENTS_INCREMENTAL_EVENTS_E16_v1'
ATTRIBUTE_STATES=('unknown','value','no_object','out_of_range','mixed')

def tensor_state_digest(module):
 h=hashlib.sha256()
 for name,value in sorted(module.state_dict().items()):
  a=value.detach().cpu().contiguous()
  h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode())
  h.update(a.numpy().tobytes())
 return h.hexdigest()

class CandidateEventResidual(nn.Module):
 """Separate parents, arbitrary child count up to a registered global capacity.

Base geometry comes from admitted old predictions or the complete candidate
domain. Residual positions are physical millimetres, not added probabilities.
No learned gate multiplies a zero residual. New parents start at logit -6.
"""
 def __init__(self,feature_count,parent_capacity,child_capacity,width=128):
  super().__init__()
  assert feature_count>0 and parent_capacity>0 and child_capacity>0
  self.parent_capacity=parent_capacity;self.child_capacity=child_capacity
  self.shared=nn.Sequential(nn.Linear(feature_count,width),nn.LayerNorm(width),nn.GELU(),
   nn.Linear(width,width),nn.LayerNorm(width),nn.GELU())
  self.presence=nn.Linear(width,parent_capacity)
  self.count=nn.Linear(width,parent_capacity*(child_capacity+1))
  self.positions=nn.Linear(width,parent_capacity*child_capacity*2)
  self.gap=nn.Linear(width,parent_capacity*max(0,child_capacity-1)) if child_capacity>1 else None
  self.direction=nn.Linear(width,parent_capacity)
  self.closed=nn.Linear(width,parent_capacity)
  for layer in [self.presence,self.count,self.positions,self.gap,self.direction,self.closed]:
   if layer is not None:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

 def forward(self,features,base):
  h=self.shared(features);b=len(features);q=self.parent_capacity;c=self.child_capacity
  assert base['positions_mm'].shape==(b,q,c,2)
  positions=base['positions_mm']+self.positions(h).reshape(b,q,c,2)
  # Direct mm offsets preserve the whole legal domain; invalid free outputs
  # remain counted false hypotheses in the decoder rather than being clipped.
  output=dict(presence_logits=base['presence_logits']+self.presence(h),
   count_logits=base['count_logits']+self.count(h).reshape(b,q,c+1),
   positions_mm=positions,direction_logits=base['direction_logits']+self.direction(h),
   closed_logits=base['closed_logits']+self.closed(h),
   full_candidate_domain_mm=base['full_candidate_domain_mm'])
  output['gap_logits']=base['gap_logits']+(self.gap(h).reshape(b,q,c-1) if self.gap is not None else 0.)
  return output

class PointwiseAttributeHead(nn.Module):
 """Original board order and signed normals; explicit state per E16 field."""
 def __init__(self,features=84,width=96,fields=16):
  super().__init__();assert fields==16
  self.shared=nn.Sequential(nn.Linear(features,width),nn.LayerNorm(width),nn.GELU(),
   nn.Linear(width,width),nn.LayerNorm(width),nn.GELU())
  self.states=nn.Linear(width,fields*len(ATTRIBUTE_STATES))
  # Sixteen numeric slots plus two explicit three-vector normal outputs.
  self.scalars=nn.Linear(width,fields);self.normals=nn.Linear(width,6)
  nn.init.zeros_(self.states.weight);nn.init.zeros_(self.states.bias)
  with torch.no_grad():
   bias=self.states.bias.view(fields,len(ATTRIBUTE_STATES));bias[:,1:]=-6.
  nn.init.zeros_(self.scalars.weight);nn.init.zeros_(self.scalars.bias)
  nn.init.zeros_(self.normals.weight);nn.init.zeros_(self.normals.bias)
 def forward(self,features):
  h=self.shared(features)
  raw=self.normals(h).reshape(-1,2,3);length=torch.linalg.vector_norm(raw,dim=-1,keepdim=True)
  return dict(state_logits=self.states(h).reshape(-1,16,len(ATTRIBUTE_STATES)),
   scalar_values=self.scalars(h),raw_signed_normals=raw,
   signed_unit_normals=raw/length.clamp_min(1e-8),normal_lengths=length[...,0],
   output_scope='POINTWISE_ESTIMATE_REQUIRES_WHOLE_CURVE_AND_SIDE_REVIEW')

class HorizontalLogitResidual(nn.Module):
 """Zero initialized correction of logits; published bypass remains exact."""
 def __init__(self,features=84,width=32):
  super().__init__()
  self.net=nn.Sequential(nn.Conv1d(features,width,1),nn.GELU(),
   nn.Conv1d(width,width,3,padding=1),nn.GELU(),nn.Conv1d(width,7,1))
  nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,x):return self.net(x)

class FrozenLegacyIncrementalContainer(nn.Module):
 def __init__(self,original115_state,original_hybrid_state,config):
  super().__init__()
  from weld_pipeline.model_registry_branch import model_from_checkpoint
  from weld_pipeline.contract_horizontal_model import from_checkpoint
  self.original115=model_from_checkpoint(original115_state).eval().requires_grad_(False)
  self.original_hybrid_horizontal=from_checkpoint(original_hybrid_state['horizontal'],initialize_extent=False).eval().requires_grad_(False)
  self.original_plate_heads=original_hybrid_state['plate_heads']
  self.config=dict(config)
  assert config['global_H_parent'] in ('original115','original_hybrid_horizontal')
  assert config['original_mixed_batch_and_padding_preserved'] is True
  self.events=CandidateEventResidual(config['candidate_feature_count'],config['parent_capacity'],config['child_capacity'])
  self.attributes=PointwiseAttributeHead()
  # Whole-path records and local material points have separate estimators.
  # A local reference never supplies a whole-path normal or thickness label.
  self.path_attributes=PointwiseAttributeHead(features=84*17)
  self.horizontal_residual=HorizontalLogitResidual()
  self._frozen_digests=self.frozen_digests()
 def train(self,mode=True):
  super().train(mode);self.original115.eval();self.original_hybrid_horizontal.eval();return self
 def frozen_digests(self):
  return dict(original115=tensor_state_digest(self.original115),
   original_hybrid_horizontal=tensor_state_digest(self.original_hybrid_horizontal),
   original_plate_heads=hashlib.sha256(pickle.dumps(self.original_plate_heads,protocol=4)).hexdigest())
 def verify_frozen(self):
  assert self.frozen_digests()==self._frozen_digests,'Frozen parameter, buffer, or plate component changed'
  assert all(not p.requires_grad for m in [self.original115,self.original_hybrid_horizontal] for p in m.parameters())
 def incremental_parameters(self):
  return [p for m in [self.events,self.attributes,self.path_attributes,self.horizontal_residual] for p in m.parameters()]
 def original_H_logits(self,x):
  # Send the full original mixed batch. Do not crop H rows before its frozen
  # forward: the original horizontal branch preserves batch/padding effects.
  parent=self.original115 if self.config['global_H_parent']=='original115' else self.original_hybrid_horizontal
  with torch.no_grad():return parent(x)
 def H_logits_with_residual(self,x):
  return self.original_H_logits(x)+self.horizontal_residual(x)
 def exact_original_prediction_bypass(self,raw_arguments,original_entrypoint):
  assert not self.config['publish_vertical_extensions'] and not self.config['publish_H_residual']
  # Calling the unmodified entrypoint avoids redecoding old results at zero.
  return original_entrypoint(**raw_arguments)
