"""Exact mass of same-owner endpoint-pair union, without quadratic pair arrays."""
import torch

def owner_pair_loss(logits, legal, interpretation_ranges):
    """One sorted, strictly increasing event grid; ranges are [owner,side,lo/hi).

    Unrepresented owners provide no term. Multiple interpretations are unioned,
    never crossed, and repeated aliases cannot multiply positive probability.
    """
    if logits.ndim!=2 or logits.shape[1]!=2 or legal.shape!=logits.shape:
        raise ValueError('Single-query E x 2 endpoint scores required')
    if not len(interpretation_ranges):return logits.sum()*0
    scores=logits.double().masked_fill(~legal,-torch.inf)
    # Sum over all ordered pairs left < right, including query-side masks.
    suffix=torch.logcumsumexp(scores[1:,1].flip(0),0).flip(0)
    denominator=torch.logsumexp(scores[:-1,0]+suffix,0)
    allowed={}
    for ranges in interpretation_ranges:
        (lo,hi),(ro,rhi)=ranges
        if not (0<=lo<=hi<=len(scores) and 0<=ro<=rhi<=len(scores)):
            raise ValueError('Interpretation outside input pool')
        for left in range(int(lo),int(hi)):
            allowed.setdefault(left,set()).update(range(max(int(ro),left+1),int(rhi)))
    terms=[]
    for left,rights in allowed.items():
        if rights and legal[left,0]:
            ids=torch.tensor(sorted(rights),device=logits.device)
            if legal[ids,1].any():terms.append(scores[left,0]+torch.logsumexp(scores[ids,1],0))
    if not terms:return logits.sum()*0
    numerator=torch.logsumexp(torch.stack(terms),0)
    if not torch.isfinite(denominator):raise ValueError('No legal endpoint pair')
    return denominator-numerator
