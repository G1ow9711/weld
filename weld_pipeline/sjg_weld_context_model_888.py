"""Zero-initialized continuous material context for the weld sequence head."""
import torch
from torch import nn
from weld_pipeline.sjg_adjacency_model_840 import AdjacencySequenceModel


class MaterialSequenceModel(AdjacencySequenceModel):
    def __init__(self):
        super().__init__();self.material=nn.Linear(8,64,bias=False);nn.init.zeros_(self.material.weight)

    def forward(self,x,condition,valid=None):
        assert x.shape[-1]==73
        scale,bias=self.condition(condition).chunk(2,-1)
        z=(self.input(x[...,:53])+self.adjacency(x[...,53:65])+self.material(x[...,65:]))*(1+.1*scale[:,None])+bias[:,None]
        if valid is not None:z=z*valid[...,None]
        for block in self.blocks:
            z=block(z)
            if valid is not None:z=z*valid[...,None]
        result=self.output(z);return result[...,:3],result[...,3:5],result[...,5:7]

    def load_baseline(self,state):
        incompatible=self.load_state_dict(state,strict=False)
        assert incompatible.missing_keys==['material.weight'] and not incompatible.unexpected_keys
        assert torch.count_nonzero(self.material.weight)==0
