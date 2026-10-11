"""XYZ-only PTv3 surface features. No catalogue, STEP or label-reader imports."""
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F
from weld_pipeline.sjg_ptv3_801 import build_backbone,voxelize,cuda_input


class SurfaceFeatureModel(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=build_backbone()
        self.embedding=nn.Linear(32,16);self.normal=nn.Linear(32,3);self.boundary=nn.Linear(32,1)
        for module in self.modules():
            if isinstance(module,nn.BatchNorm1d):
                module.running_mean=None;module.running_var=None
                module.num_batches_tracked=None;module.track_running_stats=False

    def forward(self,block):
        feature=self.backbone(cuda_input(block)).feat
        inverse=torch.from_numpy(block['inverse']).cuda()
        return self.embedding(feature)[inverse],F.normalize(self.normal(feature),dim=-1)[inverse],self.boundary(feature).squeeze(-1)[inverse]


def core_partitions(xyz,maximum=4096):
    pending=[np.arange(len(xyz),dtype=np.int64)]
    while pending:
        ids=pending.pop()
        if len(ids)<=maximum:yield ids;continue
        axis=int(np.argmax(np.ptp(xyz[ids],axis=0)))
        order=np.argsort(xyz[ids,axis],kind='stable');half=len(ids)//2
        pending.extend([ids[order[:half]],ids[order[half:]]])


def infer_surface_features(model,xyz_mm,progress=None,resource_check=None):
    """Stable raw-point mapping; all features use only this input point cloud."""
    xyz=np.asarray(xyz_mm,np.float64)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<16 or not np.isfinite(xyz).all():
        raise ValueError('Expected at least 16 finite XYZ points in millimetres')
    tree=cKDTree(xyz);parts=list(core_partitions(xyz));seen=np.zeros(len(xyz),np.uint8)
    embedding=np.empty((len(xyz),16),np.float32);normal=np.empty((len(xyz),3),np.float32)
    boundary=np.empty(len(xyz),np.float32);model.eval()
    with torch.inference_mode():
        for index,core in enumerate(parts):
            if resource_check is not None:resource_check()
            near=np.asarray(tree.query(xyz[core].mean(0),k=min(8192,len(xyz)))[1],np.int64).reshape(-1)
            context=near[~np.isin(near,core)][:8192-len(core)]
            ids=np.r_[core,context].astype(np.int64);block=voxelize(xyz[ids])
            features,normals,edges=model(block);count=len(core)
            embedding[core]=features[:count].cpu().numpy()
            normal[core]=normals[:count].cpu().numpy()
            boundary[core]=edges[:count].sigmoid().cpu().numpy();seen[core]+=1
            if progress is not None:progress(index+1,len(parts),len(core))
    assert np.all(seen==1) and np.isfinite(embedding).all() and np.isfinite(normal).all()
    return dict(embedding=embedding,normal=normal,boundary=boundary,core_coverage=seen)
