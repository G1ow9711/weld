"""Learn interval validity; match peaks globally with unmatched choices."""
import numpy as np
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment


class IntervalPairModel(nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('mean',torch.zeros(156));self.register_buffer('scale',torch.ones(156))
        self.network=nn.Sequential(nn.Linear(156,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))

    def forward(self,x):return self.network((x-self.mean)/self.scale).squeeze(-1)


def match_intervals(children,logits):
    assert len(children)==len(logits) and np.isfinite(logits).all()
    groups={}
    for i,c in enumerate(children):groups.setdefault(c['profile'],[]).append(i)
    selected=[]
    for profile,ids in sorted(groups.items()):
        starts=sorted({children[i]['start_peak_index'] for i in ids});ends=sorted({children[i]['end_peak_index'] for i in ids})
        s_index={v:i for i,v in enumerate(starts)};e_index={v:i for i,v in enumerate(ends)}
        # Each start can choose any dummy column: unmatched score zero.
        score=np.full((len(starts),len(ends)+len(starts)),-1e6,float);score[:,len(ends):]=0.
        lookup={}
        for i in ids:
            a=s_index[children[i]['start_peak_index']];b=e_index[children[i]['end_peak_index']]
            assert (a,b) not in lookup;lookup[(a,b)]=i;score[a,b]=float(logits[i])
        rows,columns=linear_sum_assignment(-score)
        for a,b in zip(rows,columns):
            if b<len(ends) and (a,b) in lookup and score[a,b]>0:selected.append(lookup[(a,b)])
    return sorted(selected,key=lambda i:(children[i]['profile'],children[i]['lo'],children[i]['hi']))
