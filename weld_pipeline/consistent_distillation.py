"""Retain teacher behavior only where it agrees with training supervision."""
import torch
import torch.nn.functional as F


def consistent_distillation(student,teacher,target,valid,features):
    # Arc behavior has no support in the straight teacher. Padding is excluded
    # before testing curvature so a straight endpoint is not treated as a bend.
    tangent=features[:,57:60];occupied=tangent.square().sum(1)>.5
    curved=(((tangent[:,:,1:]-tangent[:,:,:-1]).square().sum(1)>1e-10)&occupied[:,1:]&occupied[:,:-1]).any(1)
    p=teacher[:,:5].float().detach().sigmoid();truth=target[:,:5].float()
    aligned=(p-truth).abs()<=.2
    weights=(1+25*truth)*aligned*valid*(~curved)[:,None,None]
    cross=F.binary_cross_entropy_with_logits(student[:,:5].float(),p,reduction='none')
    entropy=-(p*torch.log(p.clamp_min(1e-7))+(1-p)*torch.log((1-p).clamp_min(1e-7)))
    return ((cross-entropy)*weights).sum()/weights.sum().clamp_min(1.)
