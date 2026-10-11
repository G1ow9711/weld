"""Shared contact model with point-only multiscale surface context."""
import torch
from torch import nn
from weld_pipeline.typed_contact_model import TypedContactNet


class ContextSurfaceContactNet(nn.Module):
    def __init__(self,features,planes,width=64):
        super().__init__()
        if features!=102:raise ValueError('Expected 102 geometry/context channels')
        self.input_features=features
        self.register_buffer('plane_axes',torch.tensor([p['axis'] for p in planes],dtype=torch.long))
        self.cut_positions_mm=nn.Parameter(torch.tensor([p['position_mm'] for p in planes],dtype=torch.float32))
        self.register_buffer('scales_mm',torch.tensor([2.,8.,32.,128.]))
        self.net=TypedContactNet(features+len(planes)*5,width).net

    def forward(self,x):
        world=x[:,67:70].float()*10000.
        delta=world[:,self.plane_axes]-self.cut_positions_mm[None,:,None]
        radial=torch.exp(-.5*(delta[:,:,None]/self.scales_mm[None,None,:,None]).square()).flatten(1,2)
        signed=(delta/128.).clamp(-1,1)
        geometry=x.clone();geometry[:,49:57]=0.;geometry[:,67:70]=0.
        return self.net(torch.cat([geometry,radial,signed],dim=1))

    def load_base_state(self,state):
        current=self.state_dict()
        for name,value in state.items():
            if name=='net.0.weight':
                current[name].zero_();current[name][:,:84].copy_(value[:,:84]);current[name][:,102:].copy_(value[:,84:])
            else:
                if name not in current or current[name].shape!=value.shape:raise ValueError('Incompatible base: '+name)
                current[name].copy_(value)
        self.load_state_dict(current)
