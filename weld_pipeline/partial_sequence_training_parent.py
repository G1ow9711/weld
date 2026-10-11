"""Sampling and mining that never turn unknown TRAIN context into negatives."""
import numpy as np
from weld_pipeline.parent_balanced_sampling import select_piece
import torch
from weld_pipeline.partial_supervision import clean_targets
from weld_pipeline.vertical_frame_augmentation import augment_complete_frames as augment_batch
from weld_pipeline.context_augmentation import augment_context_batch
from weld_pipeline.structured_sampling import structured_positions
from weld_pipeline.typed_contact_model import infer_probabilities


def sample_arrays(data,rng,batch_size,points,augment_frames=False,hard_fraction=0.,curve_fraction=0.):
    features=data[0]['sequences'][0]['features'].shape[1];halo=256;total=points+2*halo
    x=np.zeros((batch_size,features,total),np.float32)
    y=np.zeros((batch_size,7,total),np.float32);valid=np.zeros((batch_size,1,total),np.float32)
    for bi in range(batch_size):
        piece=select_piece(data,rng);positive=piece['positive_sequences'];hard=None
        if hard_fraction>0 and piece.get('hard_samples') and rng.random()<hard_fraction:
            hard=piece['hard_samples'][int(rng.integers(len(piece['hard_samples'])))];si=hard['sequence']
        else:
            arcs=piece.get('arc_sequences',[])
            if arcs and rng.random()<curve_fraction:pool=arcs
            else:pool=positive if positive and rng.random()<.75 else piece.get('known_sequences',range(len(piece['sequences'])))
            si=int(rng.choice(pool))
        seq=piece['sequences'][si];raw=piece['targets'][si];target,known=clean_targets(raw);size=len(raw)
        known_indices=np.flatnonzero(known)
        if not len(known_indices):raise ValueError('A training sequence has no known supervision')
        if hard is not None and not known[hard['index']]:raise ValueError('Hard sample lies in unknown supervision')
        if size>points:
            if hard is not None:center=hard['index']
            elif not known.all():
                focus=np.flatnonzero(target[:,1:4].max(1)>.25)
                center=int(rng.choice(focus if len(focus) else known_indices))
            else:
                focus=np.flatnonzero(target[:,1:4].max(1)>.25) if rng.random()<.5 else np.flatnonzero(target[:,0]>.5)
                center=int(rng.choice(focus)) if len(focus) and rng.random()<.7 else None
            start=int(np.clip(center-rng.integers(points//4,points*3//4),0,size-points)) if center is not None else int(rng.integers(size-points+1))
        else:start=0
        stop=min(size,start+points);n=stop-start;lo=start-halo;a=max(0,lo);b=min(size,stop+halo)
        x[bi,:,a-lo:b-lo]=seq['features'][a:b].T
        y[bi,:,halo:halo+n]=target[start:stop].T;valid[bi,:,halo:halo+n]=known[start:stop]
        if not valid[bi].any():raise ValueError('Sampling produced an empty supervision window')
    if augment_frames:(augment_context_batch if features==102 else augment_batch)(x,y,valid,rng)
    return x,y,valid


def batch_sample(*args,**kwargs):
    return [torch.from_numpy(a).cuda() for a in sample_arrays(*args,**kwargs)]


def masked_hard_positions(probability,raw_target):
    target,known=clean_targets(raw_target)
    probability=np.where(known[:,None]>0,probability,0.)
    return [r for r in structured_positions(probability,target) if known[r['index']]>0]


def mine_training_structure(model,pieces,max_per_piece=128):
    summary={}
    for piece in pieces:
        probabilities=infer_probabilities(model,piece['sequences']);candidates=[];unknown=0
        for si,(prob,target) in enumerate(zip(probabilities,piece['targets'])):
            candidates.extend({'sequence':si,**r} for r in masked_hard_positions(prob,target))
            unknown+=int(np.all(target==-1,axis=1).sum())
        candidates.sort(key=lambda r:(-r['score'],r['sequence'],r['index']))
        piece['hard_samples']=candidates[:max_per_piece]
        summary[piece['record']['name']]={'selected':len(piece['hard_samples']),'candidate_count':len(candidates),
                                       'unknown_positions_excluded':unknown}
    return summary
