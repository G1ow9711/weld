"""Assembly-relative position and extent context from the pure point cloud."""
import torch
from weld_pipeline.candidate_extent_model import CandidateExtentBranch
from weld_pipeline.vertical_plane_model import VerticalPlaneContactNet

MODEL_NAME='candidate_context_branch_v1'


class CandidateContextVertical(VerticalPlaneContactNet):
    def forward(self,x):
        world=x[:,67:70].float()*10000.
        delta=world[:,self.plane_axes]-self.cut_positions_mm[None,:,None]
        vertical=(x[:,66]>.5).any(dim=-1)
        keep=(~(vertical[:,None] & (self.plane_axes!=2)[None,:])).to(delta.dtype)
        radial=torch.exp(-.5*(delta[:,:,None]/self.scales_mm[None,None,:,None]).square())
        radial=(radial*keep[:,:,None,None]).flatten(1,2)
        signed=(delta/128.).clamp(-1,1)*keep[:,:,None]
        local=x.clone();local[:,67:70]=0.
        # 49:52 = position relative to the measured workpiece center;
        # 52:55 = point-cloud bounding extents; 55:57 = candidate distances.
        # No name, label, STEP or absolute XY plane lookup is added.
        return self.net(torch.cat([local,radial,signed],dim=1))


def from_checkpoint(state,initialize_extent=False):
    def planes(prefix):
        return [{'axis':int(a),'position_mm':float(p)} for a,p in
            zip(state['model'][prefix+'.plane_axes'],state['model'][prefix+'.cut_positions_mm'])]
    model=CandidateExtentBranch(state['features'],planes('vertical'),planes('horizontal'))
    model.vertical=CandidateContextVertical(state['features'],planes('vertical'))
    model.load_state_dict(state['model'],strict=True)
    if initialize_extent:
        with torch.no_grad():model.vertical.net[0].weight[:,49:57].zero_()
    return model
