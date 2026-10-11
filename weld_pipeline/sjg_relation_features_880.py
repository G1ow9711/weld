"""XYZ-derived pair features and a material-relation score head."""
import numpy as np
import torch
from torch import nn

FEATURE_NAMES=['log_material_distance','log_surface_distance','face_edge_min','face_edge_max',
    'material_ambiguity_min','material_ambiguity_max','normal_abs_dot','log_distance_mm',
    'log_distance_over_spacing_min','log_distance_over_spacing_max','normal_displacement_min','normal_displacement_max',
    'log_common_finite_support','log_support_count_min','log_support_count_max','same_input_core_block',
    'log_neighbor_rank','finite_support_jaccard','log_spacing_min','log_spacing_max']


def pair_features(xyz,neighbors,material,surface,normal,face_edge,ambiguity,support,core_block):
    xyz=np.asarray(xyz,np.float64);n,k=neighbors.shape;other=neighbors.clip(0)
    distance=np.linalg.norm(xyz[other]-xyz[:,None],axis=-1)
    spacing=np.where((neighbors>=0)&(distance>1e-6),distance,np.inf).min(1);spacing[~np.isfinite(spacing)]=1.
    support_count=np.asarray(support.sum(axis=1)).ravel();features=[];flat=[]
    ranks=(1,2,4,8,16,32,64,128);assert k==len(ranks)
    for j,rank in enumerate(ranks):
        b=other[:,j];valid=(neighbors[:,j]>=0)&np.isfinite(material[:,j])&np.isfinite(surface[:,j]);a=np.flatnonzero(valid)
        shared=np.asarray(support.multiply(support[b]).sum(axis=1)).ravel()
        d=distance[:,j];unit=(xyz[b]-xyz)/np.maximum(d[:,None],1e-12)
        p=np.abs((unit*normal).sum(1));q=np.abs((unit*normal[b]).sum(1))
        lo=np.minimum(spacing,spacing[b]);hi=np.maximum(spacing,spacing[b])
        countlo=np.minimum(support_count,support_count[b]);counthi=np.maximum(support_count,support_count[b])
        x=np.stack([np.log1p(material[:,j]),np.log1p(surface[:,j]),np.minimum(face_edge,face_edge[b]),np.maximum(face_edge,face_edge[b]),
            np.minimum(ambiguity,ambiguity[b]),np.maximum(ambiguity,ambiguity[b]),np.abs((normal*normal[b]).sum(1)),np.log1p(d),
            np.log1p(d/lo),np.log1p(d/hi),np.minimum(p,q),np.maximum(p,q),np.log1p(shared),np.log1p(countlo),np.log1p(counthi),
            (core_block==core_block[b]).astype(float),np.full(n,np.log2(rank)/7),shared/np.maximum(1,support_count+support_count[b]-shared),
            np.log1p(lo),np.log1p(hi)],1)[a].astype(np.float32)
        assert x.shape==(len(a),len(FEATURE_NAMES)) and np.isfinite(x).all()
        features.append(x);flat.append(a*k+j)
    x=np.concatenate(features);idx=np.concatenate(flat).astype(np.int64)
    order=np.argsort(idx,kind='stable');return x[order],idx[order]


class RelationHead(nn.Module):
    def __init__(self,mean,std,geometry):
        super().__init__();self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32));self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32))
        mask=torch.ones(len(FEATURE_NAMES));mask[2:]=float(geometry);self.register_buffer('mask',mask)
        self.layers=nn.Sequential(nn.Linear(len(FEATURE_NAMES),32),nn.SiLU(),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,1))

    def forward(self,x):return self.layers(((x-self.mean)/self.std)*self.mask).squeeze(-1)
