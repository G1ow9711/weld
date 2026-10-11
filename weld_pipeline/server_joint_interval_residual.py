"""Bounded whole-interval residual scoring over input-generated endpoint pairs.

Caller supplies frozen endpoint embeddings, provenance and base pair logits.
No references, source IDs, CAD, or event generation occur in this module.
"""
import torch
from torch import nn

class JointIntervalResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.score=nn.Sequential(nn.Linear(415,128),nn.ReLU(),nn.Linear(128,1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self,query_embedding,left_embedding,right_embedding,left_provenance,right_provenance,left_mm,right_mm,query_mm,carrier_length_mm,base_logits,pair_valid=None):
        assert left_embedding.shape==right_embedding.shape and left_embedding.shape[-1]==128
        assert left_provenance.shape==right_provenance.shape and left_provenance.shape[-1]==13
        valid=torch.ones_like(left_mm,dtype=torch.bool) if pair_valid is None else pair_valid
        assert torch.all((right_mm>left_mm)|~valid) and torch.all(carrier_length_mm>0)
        width=right_mm-left_mm
        safe_width=torch.where(valid,width,torch.ones_like(width))
        relative=torch.stack(((left_mm-query_mm)/1000,(right_mm-query_mm)/1000,width/1000,width/carrier_length_mm,(query_mm-left_mm)/safe_width),-1)
        query=query_embedding[:,None,:].expand_as(left_embedding)
        x=torch.cat((query,left_embedding,right_embedding,left_provenance,right_provenance,relative.to(left_embedding.dtype)),-1)
        assert x.shape[-1]==415
        return (base_logits+self.score(x).squeeze(-1)).masked_fill(~valid,-torch.inf)


def score_endpoint_pairs(adapter, selector, query_features, event_features, provenance,
                         positions_mm, query_mm, lengths_mm, logits, legal):
    """Shared training/runtime scoring; candidate construction never uses targets."""
    from .server_endpoint_pairs import endpoint_pairs
    pairs=endpoint_pairs(logits.float(),legal,positions_mm)
    li,ri=pairs['left_ids'],pairs['right_ids']
    with torch.no_grad():
        q=selector.query(query_features).float()
        events=selector.event(event_features).float()
    gather=lambda a,ids:a.gather(1,ids[:,:,None].expand(-1,-1,a.shape[-1]))
    scores=adapter(q,gather(events,li),gather(events,ri),gather(provenance,li),gather(provenance,ri),
                   positions_mm.gather(1,li),positions_mm.gather(1,ri),query_mm[:,None],
                   lengths_mm[:,None],pairs['base_logits'],pairs['valid'])
    return scores,pairs
