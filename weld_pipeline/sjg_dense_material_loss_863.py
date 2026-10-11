"""All known core points enter material prototype attraction; no point-pair matrix."""
import torch
from torch.nn import functional as F


def dense_material_loss(embedding,labels,core):
    known=(labels>=0)&core;zero=embedding.sum()*0
    if not known.any():return zero,dict(pull=0.,push=0.,owners=0,sampled_points=0)
    features=embedding[known];_,inverse,counts=torch.unique(labels[known],return_inverse=True,return_counts=True)
    centers=torch.zeros((len(counts),embedding.shape[-1]),device=embedding.device,dtype=embedding.dtype)
    centers=centers.index_add(0,inverse,features)/counts[:,None]
    penalty=F.relu(torch.linalg.vector_norm(features-centers[inverse],dim=1)-.2).square()
    attraction=(torch.zeros(len(counts),device=embedding.device,dtype=embedding.dtype).index_add(0,inverse,penalty)/counts).mean()
    separation=zero
    if len(counts)>1:
        distance=torch.cdist(centers,centers);mask=~torch.eye(len(counts),device=embedding.device,dtype=torch.bool)
        separation=F.relu(1.-distance[mask]).square().mean()
    loss=attraction+separation+.001*centers.norm(dim=1).mean()
    return loss,dict(pull=float(attraction.detach()),push=float(separation.detach()),owners=len(counts),sampled_points=len(features))
