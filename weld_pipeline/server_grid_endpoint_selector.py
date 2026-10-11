"""Endpoint head with explicit synthetic-grid provenance and bounded selection.

A grid flag is not a contact label. Old provenance columns must come from their
actual input producer; callers must not invent observed-event provenance for
synthetic points. Runtime selection returns no candidates for an empty side.
"""
import torch
from .server_raw_event_endpoint_selector import RawEventEndpointSelector

class GridEndpointSelector(RawEventEndpointSelector):
    def __init__(self):
        super().__init__()
        # Add separately to retain the exact legacy GEMM for original events.
        self.grid_offset = torch.nn.Parameter(torch.zeros(128))

    def load_raw_state_dict(self, state):
        copied = dict(state)
        copied['grid_offset'] = torch.zeros_like(self.grid_offset)
        self.load_state_dict(copied)

    def forward(self, query_features, event_features, query_ratio, event_ratio,
                length_mm, event_provenance, is_grid):
        if is_grid.shape != event_ratio.shape or is_grid.dtype != torch.bool:
            raise ValueError('Explicit boolean grid provenance required')
        q = self.query(query_features)[:,None,:].expand(-1,event_features.shape[1],-1)
        e = self.event(event_features)
        delta = event_ratio-query_ratio[:,None]
        relative = torch.stack((delta,delta*length_mm[:,None]/1000,
                                length_mm[:,None].expand_as(delta)/1000),-1)
        hidden = self.score[0](torch.cat((q,e,relative,event_provenance),-1))
        hidden = hidden + is_grid[:,:,None].to(hidden.dtype)*self.grid_offset.to(hidden.dtype)
        return self.score[2](self.score[1](hidden))


class StreamingEndpointTopK:
    """Stable global top-k per side, with explicit input IDs and bounded memory.

    Chunks must have strictly increasing global IDs. This matches dense stable
    score sorting; no early thresholding and no compulsory candidate output.
    The top-k proposal pass is inference-only. Re-score selected pairs with
    gradients when training the downstream set model.
    """
    def __init__(self, k=8):
        if k < 1: raise ValueError('Positive candidate budget required')
        self.k = k
        self.scores = self.ids = self.positions = None
        self.last_id = -1

    @torch.no_grad()
    def add(self, logits, legal, positions_mm, event_ids):
        if logits.ndim != 3 or logits.shape[-1] != 2 or legal.shape != logits.shape or legal.dtype != torch.bool:
            raise ValueError('Expected B x events x 2 finite scores and legal mask')
        if positions_mm.shape != logits.shape[:2] or event_ids.shape != (logits.shape[1],) or event_ids.dtype != torch.long:
            raise ValueError('Invalid position or event ID shape')
        if not torch.isfinite(logits).all() or not torch.isfinite(positions_mm).all():
            raise ValueError('Nonfinite input')
        if not len(event_ids): return
        if int(event_ids[0]) <= self.last_id or not torch.all(event_ids[1:] > event_ids[:-1]):
            raise ValueError('Chunks must follow strictly increasing event IDs')
        self.last_id = int(event_ids[-1])
        values = logits.masked_fill(~legal, -torch.inf)
        ids = event_ids[None,:,None].expand_as(values)
        positions = positions_mm[:,:,None].expand_as(values)
        if self.scores is not None:
            values = torch.cat((self.scores, values),1)
            ids = torch.cat((self.ids, ids),1)
            positions = torch.cat((self.positions, positions),1)
        order = values.argsort(dim=1, descending=True, stable=True)[:,:self.k]
        self.scores = values.gather(1,order)
        self.ids = ids.gather(1,order)
        self.positions = positions.gather(1,order)

    def pairs(self):
        if self.scores is None: return None
        b,n,_ = self.scores.shape
        li = self.ids[:,:,0,None].expand(-1,-1,n).reshape(b,-1)
        ri = self.ids[:,None,:,1].expand(-1,n,-1).reshape(b,-1)
        lo = self.positions[:,:,0,None].expand(-1,-1,n).reshape(b,-1)
        hi = self.positions[:,None,:,1].expand(-1,n,-1).reshape(b,-1)
        scores = (self.scores[:,:,0,None]+self.scores[:,None,:,1]).reshape(b,-1)
        valid = torch.isfinite(scores)&(lo<hi)
        return dict(left_ids=li,right_ids=ri,left_mm=lo,right_mm=hi,
                    valid=valid,base_logits=scores.masked_fill(~valid,-torch.inf))
