"""Warm-start attribute head with explicit observed transverse face context."""
import torch
from torch import nn
from .server_endpoint_attribute_head import EndpointAttributeHead
class TransverseAttributeHead(EndpointAttributeHead):
 def __init__(self):
  super().__init__();rng=torch.get_rng_state();old=self.trunk[0];new=nn.Linear(497,256);torch.set_rng_state(rng)
  with torch.no_grad():
   new.weight[:,:411].copy_(old.weight);new.weight[:,411:].zero_();new.bias.copy_(old.bias)
  self.trunk[0]=new
 def load_predecessor_state_dict(self,state):
  s=dict(state);w=s['trunk.0.weight'];assert w.shape==(256,411);s['trunk.0.weight']=torch.cat((w,torch.zeros((256,86),device=w.device,dtype=w.dtype)),1);self.load_state_dict(s)
