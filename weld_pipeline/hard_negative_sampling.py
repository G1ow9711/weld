"""Mine mistaken foreground regions in TRAINING pieces only."""
import numpy as np
from weld_pipeline.typed_contact_model import infer_probabilities


def sequence_hard_positions(probability,target,threshold=.55,max_positions=4,separation=32):
    negative=target[:,0]<.5
    score=np.convolve(np.where(negative,probability[:,0],0.),np.ones(5)/5.,mode='same')
    candidates=np.flatnonzero(negative & (score>=threshold))
    selected=[]
    for i in candidates[np.argsort(score[candidates],kind='stable')[::-1]]:
        if any(abs(int(i)-j['index'])<separation for j in selected):continue
        selected.append({'index':int(i),'score':float(score[i])})
        if len(selected)>=max_positions:break
    return selected


def mine_training_negatives(model,pieces,max_per_piece=128):
    summary={}
    for piece in pieces:
        probabilities=infer_probabilities(model,piece['sequences'])
        candidates=[]
        for i,(prob,target) in enumerate(zip(probabilities,piece['targets'])):
            candidates.extend({'sequence':i,**r} for r in sequence_hard_positions(prob,target))
        candidates.sort(key=lambda r:(-r['score'],r['sequence'],r['index']))
        piece['hard_samples']=candidates[:max_per_piece]
        summary[piece['record']['name']]={'selected':len(piece['hard_samples']),
            'candidate_count':len(candidates),'max_score':max((r['score'] for r in candidates),default=None)}
    return summary
