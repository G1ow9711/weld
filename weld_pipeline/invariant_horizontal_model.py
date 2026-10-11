"""Horizontal predictor independent of absolute XY translation/yaw."""
import torch
from torch import nn

from weld_pipeline.typed_contact_model import TypedContactNet

MODEL_NAME='invariant_point_horizontal_v1'


class InvariantHorizontal(nn.Module):
    def __init__(self,width=64):
        super().__init__()
        self.net=TypedContactNet(141,width).net

    def forward(self,x):
        local=x.clone()
        axes=torch.stack([x[:,60:63],x[:,63:66],x[:,57:60]],dim=1)
        position=x[:,49:52]
        local[:,49:52]=(axes*position[:,None]).sum(2)
        a,b,c,d,e,f=[x[:,i] for i in range(134,140)]
        covariance=torch.stack([torch.stack([a,b,c],dim=1),torch.stack([b,d,e],dim=1),
                                torch.stack([c,e,f],dim=1)],dim=1)
        variance=torch.einsum('baip,bijp,bajp->bap',axes,covariance,axes).clamp_min(0.)
        local[:,52:55]=torch.sqrt(variance+1e-12)
        # Retain gravity components and intrinsic curvature. All XY world
        # axes/absolute XYZ and the world covariance entries are removed.
        local[:,57:59]=0.;local[:,60:62]=0.;local[:,63:65]=0.;local[:,67:70]=0.
        local[:,134:140]=0.
        return self.net(local)


def initialize(state):
    model=InvariantHorizontal()
    current=model.state_dict()
    for name,value in current.items():
        source=state['model']['horizontal.'+name]
        if name=='net.0.weight':
            value.zero_();value[:,:84]=source[:,:84]
            value[:,49:55]=0.
        else:
            value.copy_(source)
    model.load_state_dict(current)
    return model


def from_checkpoint(state):
    model=InvariantHorizontal(state.get('width',64))
    model.load_state_dict(state['model'],strict=True)
    return model
