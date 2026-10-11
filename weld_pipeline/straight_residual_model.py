"""A shared trainable straight residual preserves the learned curved branch."""
import torch
from torch import nn
from weld_pipeline.arc_adapter_model import ArcAdapterContactNet,curvature_gate
from weld_pipeline.typed_contact_model import Residual

MODEL_NAME='straight_residual_surface_process'


class StraightResidualContactNet(ArcAdapterContactNet):
    inference_halo=512
    def __init__(self,features,planes,width=64):
        super().__init__(features,planes,width)
        for p in self.parameters():p.requires_grad_(False)
        self.straight_adapter=nn.Sequential(*[Residual(width,d) for d in [1,2,4,8,16,32]],nn.Conv1d(width,7,1))
        nn.init.zeros_(self.straight_adapter[-1].weight)
        nn.init.zeros_(self.straight_adapter[-1].bias)

    def load_prior(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        if unexpected or any(not k.startswith('straight_adapter.') for k in missing):
            raise ValueError('Prior shared model is incompatible')

    def forward_with_prior(self,x):
        gate=curvature_gate(x)
        local=x.clone();local[:,55:57]=0.
        world=local[:,67:70].float()*10000.
        delta=world[:,self.plane_axes]-self.cut_positions_mm[None,:,None]
        radial=torch.exp(-.5*(delta[:,:,None]/self.scales_mm[None,None,:,None]).square()).flatten(1,2)
        signed=(delta/128.).clamp(-1,1)
        geometry=torch.cat([local[:,:49],torch.zeros_like(local[:,49:55]),local[:,55:67],
                            torch.zeros_like(local[:,67:70]),local[:,70:]],dim=1)
        shared=self.net[:-1](torch.cat([geometry,radial,signed],dim=1))
        prior=self.net[-1](shared)+gate*self.arc_adapter(shared)
        return prior+(1-gate)*self.straight_adapter(shared),prior

    def forward(self,x):return self.forward_with_prior(x)[0]
