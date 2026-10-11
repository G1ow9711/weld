"""Mine missed/false endpoints and internal cuts from training labels only."""
import numpy as np
import torch
import torch.nn.functional as F
from weld_pipeline.hard_negative_sampling import sequence_hard_positions
from weld_pipeline.typed_contact_model import infer_probabilities


def structured_positions(probability,target,max_positions=8,separation=16):
    errors=np.abs(probability[:,1:4]-target[:,1:4])
    scores=errors.max(1)
    candidates=[{'index':int(i),'score':float(scores[i]),'channel':int(errors[i].argmax()+1)}
                for i in np.flatnonzero(scores>=.45)]
    candidates.extend({**r,'channel':0} for r in sequence_hard_positions(probability,target))
    selected=[]
    for row in sorted(candidates,key=lambda r:(-r['score'],r['index'])):
        if any(abs(row['index']-s['index'])<separation for s in selected):continue
        selected.append(row)
        if len(selected)>=max_positions:break
    return selected


def mine_training_structure(model,pieces,max_per_piece=128):
    summary={}
    for piece in pieces:
        probabilities=infer_probabilities(model,piece['sequences'])
        candidates=[]
        for i,(prob,target) in enumerate(zip(probabilities,piece['targets'])):
            candidates.extend({'sequence':i,**r} for r in structured_positions(prob,target))
        candidates.sort(key=lambda r:(-r['score'],r['sequence'],r['index']))
        piece['hard_samples']=candidates[:max_per_piece]
        summary[piece['record']['name']]={'selected':len(piece['hard_samples']),
            'candidate_count':len(candidates),'channels':{str(c):sum(r['channel']==c for r in piece['hard_samples']) for c in range(4)}}
    return summary


def hard_heat_loss(logits,y,valid,k=64):
    """Add focus on high-loss positions; padded positions never contribute."""
    heat=F.binary_cross_entropy_with_logits(logits[:,1:5].float(),y[:,1:5].float(),reduction='none')
    weighted=heat*(1+25*y[:,1:5])*valid
    selected=weighted.topk(min(k,weighted.shape[-1]),dim=-1).values.sum(-1)
    denom=valid.sum(-1).clamp(min=1,max=k)
    return (selected/denom).mean()
