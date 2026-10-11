"""Pure-input interval validity and oriented parent-continuation heads."""
import torch
from torch import nn


class IntervalParentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('node_mean',torch.zeros(132));self.register_buffer('node_scale',torch.ones(132))
        self.register_buffer('edge_mean',torch.zeros(20));self.register_buffer('edge_scale',torch.ones(20))
        self.node=nn.Sequential(nn.Linear(132,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU())
        self.valid=nn.Linear(64,1)
        self.link=nn.Sequential(nn.Linear(148,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,2))

    def encode(self,nodes):return self.node((nodes-self.node_mean)/self.node_scale)

    def node_logits(self,h):return self.valid(h).squeeze(-1)

    def edge_logits(self,h,pairs,relation):
        geometry=(relation-self.edge_mean)/self.edge_scale
        return self.link(torch.cat([h[pairs[:,0]],h[pairs[:,2]],geometry],-1))
