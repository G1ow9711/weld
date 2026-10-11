"""Input-only bounded Cartesian endpoint pairs shared by training and inference."""
import torch


def endpoint_pairs(logits,legal,positions_mm,k=8):
    assert logits.ndim==3 and logits.shape[-1]==2 and logits.shape==legal.shape
    assert legal.dtype==torch.bool and k==8
    assert positions_mm.shape==logits.shape[:2]
    assert torch.isfinite(logits).all() and torch.isfinite(positions_mm).all()
    b,e,_=logits.shape
    if e==0:
        empty=torch.empty((b,0),device=logits.device,dtype=torch.long)
        return dict(left_ids=empty,right_ids=empty,valid=empty.bool(),base_logits=logits.new_empty((b,0)))
    masked=logits.masked_fill(~legal,-torch.inf)
    top=masked.topk(min(k,e),dim=1).indices
    left=top[:,:,0];right=top[:,:,1];n=left.shape[1]
    li=left[:,:,None].expand(-1,-1,n).reshape(b,-1)
    ri=right[:,None,:].expand(-1,n,-1).reshape(b,-1)
    lp=positions_mm.gather(1,li);rp=positions_mm.gather(1,ri)
    valid=legal[:,:,0].gather(1,li)&legal[:,:,1].gather(1,ri)&(lp<rp)
    base=logits[:,:,0].gather(1,li)+logits[:,:,1].gather(1,ri)
    base=base.masked_fill(~valid,-torch.inf)
    return dict(left_ids=li,right_ids=ri,valid=valid,base_logits=base)


def pair_set_loss(scores,valid,positive):
    assert scores.shape==valid.shape==positive.shape
    assert not torch.any(positive&~valid)
    known=positive.any(1)
    if not known.any():return scores[torch.isfinite(scores)].sum()*0
    logp=scores[known].masked_fill(~valid[known],-torch.inf).log_softmax(1)
    return -logp.masked_fill(~positive[known],-torch.inf).logsumexp(1).mean()
