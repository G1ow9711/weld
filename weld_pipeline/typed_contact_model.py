"""Shared 1D model for point-derived horizontal/vertical contact proposals."""
import numpy as np
import torch
from torch import nn
from scipy.signal import find_peaks


class ChannelNorm(nn.Module):
    def __init__(self,width):
        super().__init__();self.norm=nn.LayerNorm(width)
    def forward(self,x):return self.norm(x.transpose(1,2)).transpose(1,2)


class Residual(nn.Module):
    def __init__(self,width,dilation):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(width,width,3,padding=dilation,dilation=dilation),
            ChannelNorm(width),nn.GELU(),nn.Conv1d(width,width,1))
    def forward(self,x):return x+self.net(x)


class TypedContactNet(nn.Module):
    def __init__(self,features=70,width=64):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(features,width,1),ChannelNorm(width),nn.GELU(),
            *[Residual(width,d) for d in [1,2,4,8,16,32,64,128]],nn.Conv1d(width,7,1))
    def forward(self,x):return self.net(x)


def decode_typed(sequences,probabilities,threshold=.5):
    result=[]
    for seq,prob in zip(sequences,probabilities):
        step=seq['sample_step_mm']
        def peaks(heat):
            indices,_=find_peaks(heat,height=threshold,distance=max(1,int(np.ceil(4/step))),prominence=.05)
            refined=[]
            for i in indices:
                a,b,c=np.log(np.clip(heat[i-1:i+2],1e-7,1.))
                denom=a-2*b+c
                offset=np.clip(.5*(a-c)/denom,-.5,.5) if abs(denom)>1e-8 else 0.
                refined.append(float(i+offset))
            return refined
        starts=peaks(prob[:,1]);ends=peaks(prob[:,2]);internal=peaks(prob[:,3]);used=set()
        for a in starts:
            possible=[(i,b) for i,b in enumerate(ends) if i not in used and b-a>3/step]
            if not possible:continue
            i,b=min(possible,key=lambda v:v[1]);used.add(i)
            mask=prob[int(np.ceil(a)):int(np.floor(b))+1,0]
            if not len(mask) or mask.mean()<.5:continue
            divisions=[a]+[c for c in internal if a+1.5/step<c<b-1.5/step]+[b]
            origin=seq['xyz'][0].astype(float);direction=seq['direction'].astype(float)*step
            segments=[{'kind':'line','points':[(origin+direction*x).tolist(),(origin+direction*y).tolist()]} for x,y in zip(divisions[:-1],divisions[1:])]
            result.append({'id':f'weld_{len(result)+1:04d}','type':seq['type'],'segments':segments,
                'confidence':float(min(prob[round(a),1],prob[round(b),2]))})
    return result


@torch.no_grad()
def infer_probabilities(model,sequences,chunk_size=2048,halo=256,batch_size=8):
    model.eval();device=next(model.parameters()).device
    result=[np.zeros((len(s['xyz']),7),np.float32) for s in sequences]
    blocks=[]
    for si,s in enumerate(sequences):
        size=len(s['xyz'])
        for start in range(0,size,chunk_size):
            stop=min(size,start+chunk_size)
            lo=start-halo;hi=stop+halo
            blocks.append((si,start,stop,lo,hi))
    for offset in range(0,len(blocks),batch_size):
        items=blocks[offset:offset+batch_size];length=max(hi-lo for _,_,_,lo,hi in items)
        batch=np.zeros((len(items),sequences[0]['features'].shape[1],length),np.float32)
        for bi,(si,start,stop,lo,hi) in enumerate(items):
            a=max(0,lo);b=min(len(sequences[si]['features']),hi)
            batch[bi,:,a-lo:b-lo]=sequences[si]['features'][a:b].T
        prob=model(torch.from_numpy(batch).to(device)).sigmoid().cpu().numpy()
        for bi,(si,start,stop,lo,hi) in enumerate(items):result[si][start:stop]=prob[bi,:,start-lo:stop-lo].T
    return result
