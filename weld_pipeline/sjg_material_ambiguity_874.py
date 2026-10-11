"""Separate material-contact ambiguity from geometric face boundaries."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial import cKDTree
from weld_pipeline.sjg_ptv3_801 import voxelize,cuda_input
from weld_pipeline.sjg_ptv3_runtime_807 import core_partitions


class MaterialAmbiguityHead(nn.Module):
    def __init__(self,mean,std):
        super().__init__();self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32))
        self.layers=nn.Sequential(nn.Linear(32,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))

    def forward(self,features):return self.layers((features-self.mean)/self.std).squeeze(-1)


def ambiguity_loss(logit,target):
    known=target>=0
    if not known.any():return logit.sum()*0
    return F.binary_cross_entropy_with_logits(logit[known],target[known].float())


def infer_backbone_features(model,xyz,progress=None,resource_check=None):
    xyz=np.asarray(xyz,np.float64);tree=cKDTree(xyz);parts=list(core_partitions(xyz));n=len(xyz)
    features=np.empty((n,32),np.float32);normal=np.empty((n,3),np.float32);boundary=np.empty(n,np.float32);seen=np.zeros(n,np.uint8)
    model.eval()
    with torch.inference_mode():
        for bi,core in enumerate(parts):
            if resource_check:resource_check()
            near=np.asarray(tree.query(xyz[core].mean(0),k=min(8192,n))[1],np.int64).reshape(-1)
            ids=np.r_[core,near[~np.isin(near,core)][:8192-len(core)]].astype(np.int64)
            block=voxelize(xyz[ids]);value=model.backbone(cuda_input(block)).feat
            value=value[torch.from_numpy(block['inverse'][:len(core)]).cuda()]
            features[core]=value.cpu().numpy();normal[core]=F.normalize(model.normal(value),dim=-1).cpu().numpy()
            boundary[core]=model.boundary(value).squeeze(-1).sigmoid().cpu().numpy();seen[core]+=1
            if progress:progress(bi+1,len(parts))
    assert np.all(seen==1) and np.isfinite(features).all()
    return dict(features=features,normal=normal,boundary=boundary,core_coverage=seen)


def targets_from_reference(owner,multiple_material):
    owner=np.asarray(owner);multiple=np.asarray(multiple_material,bool)
    assert np.all(owner[multiple]<0)
    target=np.full(len(owner),-1,np.int8);target[owner>=0]=0;target[multiple]=1
    return target
