"""Learn candidate boundary context in horizontals; freeze the vertical branch."""
import torch
from torch import nn
from weld_pipeline.candidate_extent_model import CandidateExtentBranch, CandidateExtentVertical
from weld_pipeline.crossing_plane_model import CrossingPlaneContactNet

MODEL_NAME='contract_horizontal_extent_v1'


class ExtentHorizontal(CrossingPlaneContactNet):
    def forward(self,x):
        world=x[:,67:70].float()*10000.
        delta=world[:,self.plane_axes]-self.cut_positions_mm[None,:,None]
        crossing=(x[:,57+self.plane_axes].abs()>1e-4).to(delta.dtype)
        radial=torch.exp(-.5*(delta[:,:,None]/self.scales_mm[None,None,:,None]).square())
        radial=(radial*crossing[:,:,None]).flatten(1,2)
        signed=(delta/128.).clamp(-1,1)*crossing
        local=x.clone();local[:,49:55]=0.;local[:,67:70]=0.
        return self.net(torch.cat([local,radial,signed],dim=1))


class ContractHorizontalBranch(nn.Module):
    def __init__(self,features,vertical_planes,horizontal_planes,width=64):
        super().__init__()
        self.horizontal=ExtentHorizontal(features,horizontal_planes,width=width)
        self.vertical=CandidateExtentVertical(features,vertical_planes,width=width)
        self.vertical.requires_grad_(False);self.vertical.eval()

    def train(self,mode=True):
        super().train(mode);self.vertical.eval();return self

    def forward(self,x):
        vertical=(x[:,66]>.5).any(dim=-1)
        if bool(vertical.all()):
            with torch.no_grad():return self.vertical(x)
        output=self.horizontal(x)
        if bool(vertical.any()):
            with torch.no_grad():fixed=self.vertical(x[vertical]).to(output.dtype)
            output=output.clone();output[vertical]=fixed
        return output


def from_checkpoint(state,initialize_extent=False):
    def planes(prefix):
        return [{'axis':int(a),'position_mm':float(p)} for a,p in
            zip(state['model'][prefix+'.plane_axes'],state['model'][prefix+'.cut_positions_mm'])]
    model=ContractHorizontalBranch(state['features'],planes('vertical'),planes('horizontal'))
    model.load_state_dict(state['model'],strict=True)
    if initialize_extent:
        with torch.no_grad():model.horizontal.net[0].weight[:,55:57].zero_()
    return model
