"""Consistent context windows for the 318-sample receptive-radius residual model."""
import numpy as np
import torch
from weld_pipeline.sequence_augmentation import augment_batch
from weld_pipeline.typed_contact_model import infer_probabilities as infer_base
from weld_pipeline.structured_sampling import structured_positions

HALO=512


def infer_probabilities(model,sequences):
    return infer_base(model,sequences,halo=HALO)


def batch_sample(data,rng,batch_size,points,augment_frames=False,hard_fraction=0.):
    channels=data[0]['sequences'][0]['features'].shape[1];total=points+2*HALO
    x=np.zeros((batch_size,channels,total),np.float32)
    y=np.zeros((batch_size,7,total),np.float32);valid=np.zeros((batch_size,1,total),np.float32)
    for bi in range(batch_size):
        piece=data[int(rng.integers(len(data)))];positive=piece['positive_sequences'];hard=None
        if piece.get('hard_samples') and rng.random()<hard_fraction:
            hard=piece['hard_samples'][int(rng.integers(len(piece['hard_samples'])))];si=hard['sequence']
        else:
            pool=positive if positive and rng.random()<.75 else range(len(piece['sequences']))
            si=int(rng.choice(pool))
        seq=piece['sequences'][si];target=piece['targets'][si];size=len(target)
        if size>points:
            if hard is not None:
                start=int(np.clip(hard['index']-rng.integers(points//4,points*3//4),0,size-points))
            else:
                focus=np.flatnonzero(target[:,1:4].max(1)>.25) if rng.random()<.5 else np.flatnonzero(target[:,0]>.5)
                if len(focus) and rng.random()<.7:
                    start=int(np.clip(int(rng.choice(focus))-rng.integers(points//4,points*3//4),0,size-points))
                else:start=int(rng.integers(size-points+1))
        else:start=0
        stop=min(size,start+points);n=stop-start;lo=start-HALO;a=max(0,lo);b=min(size,stop+HALO)
        x[bi,:,a-lo:b-lo]=seq['features'][a:b].T
        y[bi,:,HALO:HALO+n]=target[start:stop].T;valid[bi,:,HALO:HALO+n]=1
    if augment_frames:augment_batch(x,y,valid,rng)
    return [torch.from_numpy(a).cuda() for a in [x,y,valid]]


def mine_training_structure(model,pieces,max_per_piece=128):
    summary={}
    for piece in pieces:
        probabilities=infer_probabilities(model,piece['sequences']);candidates=[]
        for i,(prob,target) in enumerate(zip(probabilities,piece['targets'])):
            candidates.extend({'sequence':i,**r} for r in structured_positions(prob,target))
        candidates.sort(key=lambda r:(-r['score'],r['sequence'],r['index']))
        piece['hard_samples']=candidates[:max_per_piece]
        summary[piece['record']['name']]={'selected':len(piece['hard_samples']),'candidate_count':len(candidates)}
    return summary
