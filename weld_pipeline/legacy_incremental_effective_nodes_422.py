"""Shared-node semantics for the differentiable trainer and free decoder.

Gap choices are discrete. Only retained coordinates carry geometry gradients.
No truth, clipping, smoothing, or positive-gap closure is performed here.
"""
import numpy as np


def effective_nodes_numpy(positions, positive_gaps):
    result=np.asarray(positions).copy()
    gaps=np.asarray(positive_gaps,dtype=bool)
    assert result.ndim==2 and result.shape[1]==2 and gaps.shape==(len(result)-1,)
    for child in range(1,len(result)):
        if not gaps[child-1]:result[child,0]=result[child-1,1]
    return result


def effective_nodes_torch(positions,gap_logits):
    import torch
    assert positions.ndim>=2 and positions.shape[-1]==2
    assert gap_logits.shape==positions.shape[:-2]+(positions.shape[-2]-1,)
    starts=positions[...,0:1,0]
    if positions.shape[-2]>1:
        rest=torch.where(gap_logits>=0.,positions[...,1:,0],positions[...,:-1,1])
        starts=torch.cat((starts,rest),dim=-1)
    return torch.stack((starts,positions[...,1]),dim=-1)
