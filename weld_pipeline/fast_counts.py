"""Vectorized acceptance counts for calibration; full audit uses evaluation.py."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from weld_pipeline.evaluation import endpoints,topology,welding_type


def score_counts(predictions,truths,limit_mm=3.):
    n,m=len(predictions),len(truths)
    if not m:raise ValueError('Empty truth requires separate scoring')
    if not n:return {'truth_count':m,'prediction_count':0,'valid_welds':0,'false_or_duplicate_predictions':0}
    p=np.array([endpoints(w) for w in predictions]);t=np.array([endpoints(w) for w in truths])
    direct=np.linalg.norm(p[:,None]-t[None],axis=3)
    flipped=np.linalg.norm(p[:,None,::-1]-t[None],axis=3)
    dmax=direct.max(2);rmax=flipped.max(2);dsum=direct.sum(2);rsum=flipped.sum(2)
    reverse=(rmax<dmax)|((rmax==dmax)&(rsum<dsum))
    errors=np.where(reverse[:,:,None],flipped,direct)
    valid=errors.max(2)<=limit_mm
    for i,j in zip(*np.nonzero(valid)):
        a,b=predictions[i],truths[j]
        if welding_type(a['type'])!=welding_type(b['type']):valid[i,j]=False;continue
        kinds=topology(a)[::-1] if reverse[i,j] else topology(a)
        if kinds!=topology(b):valid[i,j]=False;continue
        pe=np.array([v for s in a['segments'] for v in [s['points'][0],s['points'][-1]]],float)
        te=np.array([v for s in b['segments'] for v in [s['points'][0],s['points'][-1]]],float)
        if reverse[i,j]:pe=pe[::-1]
        if np.linalg.norm(pe-te,axis=1).max()>limit_mm:valid[i,j]=False
    cost=np.where(valid,0.,1e9)+np.minimum(errors.sum(2),1e5)
    ii,jj=linear_sum_assignment(cost);count=int(valid[ii,jj].sum())
    return {'truth_count':m,'prediction_count':n,'valid_welds':count,'false_or_duplicate_predictions':n-count}
