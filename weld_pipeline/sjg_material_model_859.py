"""Distinct surface and material embeddings sharing the XYZ-only PTv3 backbone."""
import torch
from torch import nn
from torch.nn import functional as F
from weld_pipeline.sjg_ptv3_runtime_807 import SurfaceFeatureModel
from weld_pipeline.sjg_ptv3_801 import cuda_input


class SurfaceMaterialModel(SurfaceFeatureModel):
    def __init__(self):
        super().__init__();self.material=nn.Linear(32,16)

    def forward(self,block):
        feature=self.backbone(cuda_input(block)).feat;inverse=torch.from_numpy(block['inverse']).cuda()
        return (self.embedding(feature)[inverse],F.normalize(self.normal(feature),dim=-1)[inverse],
            self.boundary(feature).squeeze(-1)[inverse],self.material(feature)[inverse])

    def load_surface(self,state):
        result=self.load_state_dict(state,strict=False)
        assert set(result.missing_keys)=={'material.weight','material.bias'} and not result.unexpected_keys


def material_loss(embedding,labels,core):
    """Instance labels define relations only; unknown rows never become negatives."""
    known=(labels>=0)&core;zero=embedding.sum()*0
    if not known.any():return zero,dict(pull=0.,push=0.,owners=0,sampled_points=0)
    groups=torch.unique(labels[known]);per_group=min(64,max(1,2048//len(groups)));centers=[];pull=[];sampled=0
    for group in groups:
        ids=torch.nonzero(known&(labels==group),as_tuple=True)[0]
        if len(ids)>per_group:ids=ids[torch.linspace(0,len(ids)-1,per_group,device=ids.device).long()]
        value=embedding[ids];center=value.mean(0);centers.append(center);sampled+=len(ids)
        pull.append(F.relu(torch.linalg.vector_norm(value-center,dim=1)-.2).square().mean())
    centers=torch.stack(centers);attraction=torch.stack(pull).mean()
    separation=zero
    if len(centers)>1:
        distances=torch.cdist(centers,centers);mask=~torch.eye(len(centers),device=embedding.device,dtype=torch.bool)
        separation=F.relu(1.-distances[mask]).square().mean()
    total=attraction+separation+.001*centers.norm(dim=1).mean()
    return total,dict(pull=float(attraction.detach()),push=float(separation.detach()),owners=len(groups),sampled_points=sampled)
