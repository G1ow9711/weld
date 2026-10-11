"""Endpoint selector with an explicit added-observed-projection input flag."""
import torch
from torch import nn
from .server_boundary_event_provenance_selector import BoundaryEventProvenanceSelector
class RawEventEndpointSelector(BoundaryEventProvenanceSelector):
 def __init__(self):
  super().__init__();rng=torch.get_rng_state();old=self.score[0];new=nn.Linear(272,128);torch.set_rng_state(rng)
  with torch.no_grad():
   new.weight[:,:271].copy_(old.weight);new.weight[:,271:].zero_();new.bias.copy_(old.bias)
  self.score[0]=new
 def load_predecessor_state_dict(self,state):
  state=dict(state);w=state['score.0.weight'];assert w.shape==(128,271);state['score.0.weight']=torch.cat((w,torch.zeros((128,1),dtype=w.dtype,device=w.device)),1);self.load_state_dict(state)
