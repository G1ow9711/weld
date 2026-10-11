"""Restore point-derived candidate boundary distances for vertical learning."""
import torch
from torch import nn
from weld_pipeline.crossing_plane_model import CrossingPlaneContactNet
from weld_pipeline.vertical_plane_model import VerticalPlaneContactNet

MODEL_NAME = 'candidate_extent_branch_v1'


class CandidateExtentVertical(VerticalPlaneContactNet):
    def forward(self, x):
        world=x[:,67:70].float()*10000.
        delta=world[:,self.plane_axes]-self.cut_positions_mm[None,:,None]
        vertical=(x[:,66]>.5).any(dim=-1)
        keep=(~(vertical[:,None] & (self.plane_axes!=2)[None,:])).to(delta.dtype)
        radial=torch.exp(-.5*(delta[:,:,None]/self.scales_mm[None,None,:,None]).square())
        radial=(radial*keep[:,:,None,None]).flatten(1,2)
        signed=(delta/128.).clamp(-1,1)*keep[:,:,None]
        local=x.clone()
        local[:,49:55]=0.
        local[:,67:70]=0.
        # Channels 55/56 are distances along the point-derived candidate;
        # no annotation, workpiece identity or endpoint snapping is involved.
        return self.net(torch.cat([local,radial,signed],dim=1))


class CandidateExtentBranch(nn.Module):
    def __init__(self,features,vertical_planes,horizontal_planes,width=64):
        super().__init__()
        self.horizontal=CrossingPlaneContactNet(features,horizontal_planes,width=width)
        self.vertical=CandidateExtentVertical(features,vertical_planes,width=width)
        self.horizontal.requires_grad_(False)
        self.horizontal.eval()

    def train(self,mode=True):
        super().train(mode);self.horizontal.eval();return self

    def forward(self,x):
        vertical=(x[:,66]>.5).any(dim=-1)
        if bool(vertical.all()):return self.vertical(x)
        with torch.no_grad():output=self.horizontal(x)
        if bool(vertical.any()):
            output=output.clone();output[vertical]=self.vertical(x[vertical]).to(output.dtype)
        return output


def from_checkpoint(state,initialize_extent=False):
    def planes(prefix):
        return [{'axis':int(a),'position_mm':float(p)} for a,p in
            zip(state['model'][prefix+'.plane_axes'],state['model'][prefix+'.cut_positions_mm'])]
    model=CandidateExtentBranch(state['features'],planes('vertical'),planes('horizontal'))
    model.load_state_dict(state['model'],strict=True)
    if initialize_extent:
        with torch.no_grad():model.vertical.net[0].weight[:,55:57].zero_()
    return model
