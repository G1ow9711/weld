"""Raw-cloud context outside a vertical candidate, without expanding its output."""
import numpy as np
import torch
from scipy.spatial import cKDTree
from weld_pipeline.typed_sequences import OFFSETS
from weld_pipeline.indexed_surface_features import append_indexed_surface_features


def append_axial_context(xyz, sequences, halo=256):
    xyz=np.asarray(xyz,dtype=float)
    if halo<1 or xyz.ndim!=2 or xyz.shape[1]!=3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Finite XYZ and positive context size required')
    center=(xyz.min(0)+xyz.max(0))/2;extent=np.ptp(xyz,axis=0)
    tree=cKDTree(xyz);queries=[];indices=[]
    for i,s in enumerate(sequences):
        if s['type']!='vertical':continue
        f=s['features']
        if f.shape[1]!=84 or not (f[:,66]>.5).all():raise ValueError('Expected original vertical features')
        step=s['sample_step_mm'];direction=np.asarray(s['direction'],float)
        q=np.concatenate([np.asarray(s['xyz'][0],float)+np.arange(-halo,0)[:,None]*step*direction,
                          np.asarray(s['xyz'][-1],float)+np.arange(1,halo+1)[:,None]*step*direction])
        n1,n2=np.asarray(f[len(f)//2,60:63],float),np.asarray(f[len(f)//2,63:66],float)
        offsets=np.array([n1*x+n2*y for y in OFFSETS for x in OFFSETS])
        distance=tree.query((q[:,None]+offsets).reshape(-1,3),workers=4,distance_upper_bound=50.)[0]
        t=np.r_[np.arange(-halo,0),np.arange(len(f),len(f)+halo)]*step
        extra=np.column_stack([(q-center)/1000,np.tile(extent/1000,(len(q),1)),
            t/1000,((len(f)-1)*step-t)/1000,np.tile(np.r_[direction,n1,n2],(len(q),1)),
            np.ones(len(q)),q/10000])
        base=np.column_stack([np.minimum(distance.reshape(len(q),49),50.)/50.,extra]).astype(np.float32)
        queries.append({'xyz':q.astype(np.float32),'features':base,'type':'vertical'})
        indices.append(i)
    enriched,stats=append_indexed_surface_features(xyz,queries)
    output=[dict(s) for s in sequences]
    for i,s in zip(indices,enriched):
        output[i]['left_context']=s['features'][:halo]
        output[i]['right_context']=s['features'][halo:]
    return output,{'vertical_sequences':len(indices),'context_points':len(indices)*2*halo,'halo':halo,'surface':stats}


def feature_window(seq,lo,hi):
    size=len(seq['features']);out=np.zeros((hi-lo,84),np.float32)
    a,b=max(0,lo),min(size,hi)
    if b>a:out[a-lo:b-lo]=seq['features'][a:b]
    if lo<0 and 'left_context' in seq:
        context=seq['left_context'];start=max(lo,-len(context));stop=min(hi,0)
        if stop>start:out[start-lo:stop-lo]=context[len(context)+start:len(context)+stop]
    if hi>size and 'right_context' in seq:
        context=seq['right_context'];start=max(lo,size);stop=min(hi,size+len(context))
        if stop>start:out[start-lo:stop-lo]=context[start-size:stop-size]
    return out


@torch.no_grad()
def infer_context_probabilities(model,sequences,chunk_size=2048,halo=256,batch_size=8):
    model.eval();device=next(model.parameters()).device
    result=[np.zeros((len(s['xyz']),7),np.float32) for s in sequences];blocks=[]
    for si,s in enumerate(sequences):
        if 'left_context' in s and min(len(s['left_context']),len(s['right_context']))<halo:
            raise ValueError('Stored context is shorter than inference halo')
        for start in range(0,len(s['xyz']),chunk_size):
            stop=min(len(s['xyz']),start+chunk_size);blocks.append((si,start,stop,start-halo,stop+halo))
    for offset in range(0,len(blocks),batch_size):
        items=blocks[offset:offset+batch_size];length=max(hi-lo for _,_,_,lo,hi in items)
        batch=np.zeros((len(items),84,length),np.float32)
        for bi,(si,start,stop,lo,hi) in enumerate(items):batch[bi,:,:hi-lo]=feature_window(sequences[si],lo,hi).T
        probability=model(torch.from_numpy(batch).to(device)).sigmoid().cpu().numpy()
        for bi,(si,start,stop,lo,hi) in enumerate(items):result[si][start:stop]=probability[bi,:,start-lo:stop-lo].T
    return result
