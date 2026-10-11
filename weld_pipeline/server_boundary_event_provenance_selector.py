"""Add input geometric event provenance to the unchanged endpoint selector."""
import torch
from torch import nn
from .server_boundary_event_selector import BoundaryEventSelector
class BoundaryEventProvenanceSelector(BoundaryEventSelector):
 def __init__(self):
  super().__init__()
  # Preserve base initialization RNG consumption for matched sampling experiments.
  state=torch.get_rng_state();base=self.score[0];extended=nn.Linear(271,128);torch.set_rng_state(state)
  with torch.no_grad():
   extended.weight[:,:259].copy_(base.weight);extended.weight[:,259:].zero_();extended.bias.copy_(base.bias)
  self.score[0]=extended
 def load_base_state_dict(self,state):
  copied=dict(state);w=state['score.0.weight'];assert w.shape==(128,259)
  copied['score.0.weight']=torch.cat((w,torch.zeros((128,12),dtype=w.dtype,device=w.device)),1)
  self.load_state_dict(copied)
 def forward(self,query_features,event_features,query_ratio,event_ratio,length_mm,event_provenance):
  q=self.query(query_features)[:,None,:].expand(-1,event_features.shape[1],-1);e=self.event(event_features);delta=event_ratio-query_ratio[:,None];relative=torch.stack((delta,delta*length_mm[:,None]/1000,length_mm[:,None].expand_as(delta)/1000),-1)
  return self.score(torch.cat((q,e,relative,event_provenance),-1))
