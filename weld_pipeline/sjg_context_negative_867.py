"""Training-only core-to-context local negative pairs and their balanced loss."""
import numpy as np
import torch
from scipy.spatial import cKDTree


def local_negative_pairs(coord,labels,core):
    """Each known core anchor sees the nearest point of every other known owner.

    Coordinates are model-visible voxel representative coordinates. Owner IDs
    only select supervised pairs; they are never input features or predictions.
    Unknown labels are excluded, not treated as different material.
    """
    labels=np.asarray(labels);core=np.asarray(core,bool);coord=np.asarray(coord)
    known_owners=np.unique(labels[labels>=0]);parts=[];groups=[];g=0
    for target in known_owners:
        targets=np.flatnonzero(labels==target)
        tree=cKDTree(coord[targets])
        for anchor_owner in np.unique(labels[core&(labels>=0)]):
            if anchor_owner==target:continue
            anchors=np.flatnonzero(core&(labels==anchor_owner))
            near=targets[np.asarray(tree.query(coord[anchors],k=1)[1],np.int64)]
            parts.append(np.stack([anchors,near],1));groups.append(np.full(len(anchors),g,np.int64));g+=1
    pairs=np.concatenate(parts).astype(np.int64) if parts else np.empty((0,2),np.int64)
    group=np.concatenate(groups) if groups else np.empty(0,np.int64)
    assert np.all(core[pairs[:,0]]) and np.all(labels[pairs]>=0)
    assert np.all(labels[pairs[:,0]]!=labels[pairs[:,1]])
    return dict(pairs=pairs,group=group)


def context_negative_loss(embedding,pairs,group):
    if not len(pairs):return embedding.sum()*0
    distances=torch.linalg.vector_norm(embedding[pairs[:,0]]-embedding[pairs[:,1]],dim=1)
    penalty=torch.relu(1.-distances).square()
    counts=torch.bincount(group);assert torch.all(counts>0)
    sums=torch.zeros(len(counts),dtype=embedding.dtype,device=embedding.device).index_add(0,group,penalty)
    return (sums/counts).mean()
