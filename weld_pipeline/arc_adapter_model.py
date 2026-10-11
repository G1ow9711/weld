"""One shared frozen trunk with a trainable curved-sequence residual head."""
import torch
from torch import nn
from weld_pipeline.invariant_surface_model import InvariantSurfaceContactNet
from weld_pipeline.typed_contact_model import Residual

MODEL_NAME = 'arc_adapter_surface_process'


def curvature_gate(x):
    tangent = x[:,57:60]
    occupied = tangent.square().sum(dim=1) > .5
    adjacent = occupied[:,1:] & occupied[:,:-1]
    changes = (tangent[:,:,1:]-tangent[:,:,:-1]).square().sum(dim=1)
    return ((changes>1e-10)&adjacent).any(dim=1).to(x.dtype)[:,None,None]


class ArcAdapterContactNet(InvariantSurfaceContactNet):
    def __init__(self,features,planes,width=64):
        super().__init__(features,planes,width)
        for p in self.parameters():p.requires_grad_(False)
        self.arc_adapter = nn.Sequential(*[Residual(width,d) for d in [1,2,4,8]],nn.Conv1d(width,7,1))
        nn.init.zeros_(self.arc_adapter[-1].weight);nn.init.zeros_(self.arc_adapter[-1].bias)

    def load_base(self,state):
        missing,unexpected = self.load_state_dict(state,strict=False)
        if unexpected or any(not n.startswith('arc_adapter.') for n in missing):
            raise ValueError('Incompatible frozen backbone')

    def forward(self,x):
        gate = curvature_gate(x)
        if not bool(gate.any()):return super().forward(x)
        local=x.clone();local[:,55:57]=0.
        world=local[:,67:70].float()*10000.
        delta=world[:,self.plane_axes]-self.cut_positions_mm[None,:,None]
        radial=torch.exp(-.5*(delta[:,:,None]/self.scales_mm[None,None,:,None]).square()).flatten(1,2)
        signed=(delta/128.).clamp(-1,1)
        geometry=torch.cat([local[:,:49],torch.zeros_like(local[:,49:55]),local[:,55:67],
                            torch.zeros_like(local[:,67:70]),local[:,70:]],dim=1)
        shared=self.net[:-1](torch.cat([geometry,radial,signed],dim=1))
        return self.net[-1](shared)+gate*self.arc_adapter(shared)
