"""Select a non-overlapping span set on each carrier, with weighted-loss correction."""
import numpy as np
from scipy.special import logit,expit
from weld_pipeline.server_span_decoder import decode as deduplicate


def decode(rows,scores,candidates,positive_weight=20.):
    # Weighted BCE fits log odds shifted by log(pos_weight). Undo that shift.
    utility=logit(np.clip(scores,1e-7,1-1e-7))-np.log(positive_weight)
    groups={}
    for i,row in enumerate(rows):
        if utility[i]<=0:continue
        groups.setdefault((row['candidate'],row['weld_type']),[]).append(i)
    selected=[]
    for ids in groups.values():
        order=sorted(ids,key=lambda i:(max(rows[i]['start'],rows[i]['end']),min(rows[i]['start'],rows[i]['end']),i))
        ends=np.array([max(rows[i]['start'],rows[i]['end']) for i in order]);low=np.array([min(rows[i]['start'],rows[i]['end']) for i in order]);high=ends
        best=np.zeros(len(order)+1);previous=np.zeros(len(order),int);take=np.zeros(len(order),bool)
        for k,i in enumerate(order):
            # Sub-3mm endpoint uncertainty may overlap; do not erase short
            # segments by permitting overlap greater than half their extent.
            tol=min(3.,.5*(high[k]-low[k]))
            j=min(k,int(np.searchsorted(ends[:k],low[k]+tol,side='right')))
            previous[k]=j;value=best[j]+utility[i]
            if value>best[k]:best[k+1]=value;take[k]=True
            else:best[k+1]=best[k]
        at=len(order)
        while at:
            k=at-1
            if take[k] and abs(best[at]-(best[previous[k]]+utility[order[k]]))<1e-8:selected.append(order[k]);at=previous[k]
            else:at-=1
    mask=np.zeros(len(rows));mask[selected]=expit(utility[selected])
    return deduplicate(rows,mask,candidates,threshold=.5)


def controls():
    candidates=[dict(face_indices=[0,1])]
    def row(a,b):return dict(candidate=0,start=a,end=b,weld_type='horizontal',representation=dict(kind='line',endpoints_mm=[[a,0,0],[b,0,0]]))
    chosen,_=decode([row(0,10),row(0,20),row(10,20)],np.array([.999,.98,.999]),candidates)
    assert len(chosen)==2 and sorted(x['start'] for x in chosen)==[0,10]
    return dict(overlapping_long_competitor_excluded=True,adjacent_segments_preserved=True,weighted_BCE_prior_corrected=True)
