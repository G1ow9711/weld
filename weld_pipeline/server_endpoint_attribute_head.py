"""Typed endpoint attributes on frozen input descriptors; no pose/frame output."""
import torch
from torch import nn
from torch.nn import functional as F
class EndpointAttributeHead(nn.Module):
 def __init__(self):
  super().__init__();self.trunk=nn.Sequential(nn.Linear(411,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU());self.state=nn.Linear(128,32);self.numeric=nn.Linear(128,8);self.register_buffer('numeric_scale',torch.tensor([50.,50.,500.,500.,50.,50.,50.,50.]))
 def forward(self,x):
  h=self.trunk(x);return self.state(h).reshape(-1,8,4),F.softplus(self.numeric(h))*self.numeric_scale
