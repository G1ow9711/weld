"""Small 1D neural decoder over point-derived contact features; experimental."""
import numpy as np
import torch
from torch import nn
from scipy.signal import find_peaks


class Residual(nn.Module):
    def __init__(self,width,dilation):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(width,width,3,padding=dilation,dilation=dilation),
            nn.GroupNorm(8,width),nn.GELU(),nn.Conv1d(width,width,1))

    def forward(self,x):return x+self.net(x)


class ContactNet(nn.Module):
    def __init__(self,features=59,width=64):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(features,width,1),nn.GELU(),
            *[Residual(width,d) for d in [1,2,4,8,16,32,64,128]],nn.Conv1d(width,5,1))

    def forward(self,x):return self.net(x)


def heat_peaks(heat,threshold=.5):
    indices,_=find_peaks(heat,height=threshold,distance=3,prominence=.05)
    refined=[]
    for i in indices:
        left,right=max(0,i-2),min(len(heat),i+3)
        weight=np.maximum(heat[left:right]-.05,0)
        refined.append(float(np.sum(np.arange(left,right)*weight)/max(weight.sum(),1e-10)))
    return refined


def decode(sequences,probabilities,threshold=.5):
    welds=[]
    for seq,prob in zip(sequences,probabilities):
        starts=heat_peaks(prob[:,1],threshold);ends=heat_peaks(prob[:,2],threshold)
        internal=heat_peaks(prob[:,3],threshold)
        used_end=set()
        for a in starts:
            possible=[(i,b) for i,b in enumerate(ends) if i not in used_end and b>a+2]
            if not possible:continue
            i,b=min(possible,key=lambda v:v[1]);used_end.add(i)
            if prob[int(np.ceil(a)):int(np.floor(b))+1,0].mean()<.5:continue
            div=[a]+[c for c in internal if a+2<c<b-2]+[b]
            origin=seq['xyz'][0].astype(float)
            # Subtracting two float32 global coordinates and extrapolating that
            # rounded step over meters amplifies sub-mm rounding unnecessarily.
            step=seq['direction'].astype(float)*seq.get('sample_step_mm',2.)
            segments=[{'kind':'line','points':[(origin+step*x).astype(float).tolist(),(origin+step*y).astype(float).tolist()]} for x,y in zip(div[:-1],div[1:])]
            welds.append({'id':f'weld_{len(welds)+1:04d}','type':'horizontal','segments':segments,
                'confidence':float(min(prob[round(a),1],prob[round(b),2]))})
    return welds
