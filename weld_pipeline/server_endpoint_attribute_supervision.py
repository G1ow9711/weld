"""Loss for independently encoded Excel endpoint evidence; no geometry inference.

Codes: exact_mm=0, greater_than_50=1, no_hole=2,
no_blocker_or_greater_than_50=3, unknown=-1. Only explicitly required, known
fields contribute. NaN placeholders are indexed out before arithmetic.
"""
import torch
from torch.nn import functional as F

def masked_endpoint_loss(state_logits, predicted_mm, target_state, target_mm,
                         state_mask, numeric_mask, row_weight=None):
    shape=target_state.shape
    assert len(shape)==2 and shape[1]==8
    assert state_logits.shape==(*shape,4) and predicted_mm.shape==shape
    assert target_mm.shape==shape and state_mask.shape==numeric_mask.shape==shape
    assert state_mask.dtype==numeric_mask.dtype==torch.bool
    assert not torch.any(numeric_mask & ~state_mask)
    assert torch.all(target_state[state_mask]>=0) and torch.all(target_state[state_mask]<=3)
    assert torch.all(target_state[numeric_mask]==0)
    assert torch.isfinite(target_mm[numeric_mask]).all()
    weight=torch.ones(shape[0],device=state_logits.device,dtype=state_logits.dtype) if row_weight is None else row_weight
    assert weight.shape==(shape[0],) and torch.isfinite(weight).all() and torch.all(weight>=0)
    states=[];numbers=[]
    for field in range(8):
        known=state_mask[:,field] & (weight>0)
        if known.any():
            logits=state_logits[known,field];assert torch.isfinite(logits).all()
            loss=F.cross_entropy(logits,target_state[known,field],reduction='none')
            states.append((loss*weight[known]).sum()/weight[known].sum())
        exact=numeric_mask[:,field] & (weight>0)
        if exact.any():
            values=predicted_mm[exact,field];assert torch.isfinite(values).all()
            loss=F.smooth_l1_loss(values,target_mm[exact,field],beta=1.,reduction='none')
            numbers.append((loss*weight[exact]).sum()/weight[exact].sum())
    # Empty slices retain autograd connections without reading unknown NaNs.
    state_loss=torch.stack(states).mean() if states else state_logits.reshape(-1)[:0].sum()
    numeric_loss=torch.stack(numbers).mean() if numbers else predicted_mm.reshape(-1)[:0].sum()
    return state_loss,numeric_loss,dict(known_states=int(state_mask.sum()),exact_numbers=int(numeric_mask.sum()),state_fields=len(states),numeric_fields=len(numbers))
