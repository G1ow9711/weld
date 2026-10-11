"""Separate Gaussian event mass and background mass in endpoint heat loss."""
import torch
from torch.nn import functional as F
from weld_pipeline.sjg_train_sequence_822 import objective as baseline_objective


def event_loss(logits, target):
    dims=tuple(range(logits.ndim-1))
    positive=target.sum(dims)
    negative=(1-target).sum(dims)
    pos=-(target*F.logsigmoid(logits)).sum(dims)/positive.clamp_min(1)
    neg=-((1-target)*F.logsigmoid(-logits)).sum(dims)/negative.clamp_min(1)
    # All-background cores contribute background only, without artificial events.
    active_pos=(positive>0).to(logits.dtype)
    active_neg=(negative>0).to(logits.dtype)
    return ((pos*active_pos+neg*active_neg)/(active_pos+active_neg).clamp_min(1)).mean()


def objective(result,truth):
    logits,offset,types=result;y=truth['targets'];mask=truth['offset_mask']
    fg=F.binary_cross_entropy_with_logits(logits[...,:1],y[...,:1],pos_weight=logits.new_tensor([2.]))
    event=event_loss(logits[...,1:],y[...,1:])
    foreground=logits[...,0].sigmoid();target=y[...,0]
    dice=1-(2*(foreground*target).sum()+1)/(foreground.sum()+target.sum()+1)
    regression=(F.smooth_l1_loss(offset,truth['offsets'],reduction='none')*mask).sum()/mask.sum().clamp_min(1)
    known=truth['types']>=0
    kind=F.cross_entropy(types[known],truth['types'][known]) if known.any() else types.sum()*0
    continuity=F.l1_loss(foreground[:,1:]-foreground[:,:-1],target[:,1:]-target[:,:-1]) if target.shape[1]>1 else foreground.sum()*0
    loss=fg+event+.25*dice+.4*regression+.25*kind+.5*continuity
    return loss,torch.stack([fg,event,dice,regression,kind,continuity]).detach().cpu().numpy()
