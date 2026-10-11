"""Argmax-aligned auxiliary loss for known endpoint candidate sets only."""
import torch
from torch.nn import functional as F
from weld_pipeline.server_boundary_event_selector import masked_endpoint_event_loss

def endpoint_rank_loss(logits,legal,positive):
    base=masked_endpoint_event_loss(logits,legal,positive)
    terms=[]
    for side in (0,1):
        good=positive[:,:,side];bad=legal[:,:,side]&~good
        known=good.any(1)&bad.any(1)
        if not known.any():continue
        values=logits[known,:,side]
        best_good=values.masked_fill(~good[known],-torch.inf).amax(1)
        best_bad=values.masked_fill(~bad[known],-torch.inf).amax(1)
        terms.append(F.softplus(best_bad-best_good))
    return base+(torch.cat(terms).mean() if terms else logits.sum()*0)
