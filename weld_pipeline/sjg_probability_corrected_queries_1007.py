"""Candidate sampler for a subsequent locally reviewed training round.

Draw from the actual 2 mm inference grid: a uniform population component keeps
every background point eligible, and a TRAIN endpoint-focus component supplies
rare examples. Exact mixture probabilities support unbiased full-grid losses.
This sampler does not modify the running run187/run188 experiment.
"""
import numpy as np
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_local_curve_supervision_915 import event_targets


def grid_layout(profiles,supervision):
    curves=[CurveCarrier(p['representation']) for p in profiles]
    counts=np.array([max(1,int(np.ceil(c.length/2.)))+1 for c in curves],np.int64)
    ptr=np.r_[0,np.cumsum(counts)];focus=[];soft=np.zeros(4,np.float64)
    for i,curve in enumerate(curves):
        events=supervision['by_profile'].get(str(i),[])
        if not events:continue
        q=curve.grid(2.);labels,offset,mask=event_targets(profiles[i],events,q)
        # Also include closed-curve occupancy where no endpoint can be defined.
        selected=mask.any(1)|(labels[:,3]>.5)
        focus.append(ptr[i]+np.flatnonzero(selected));soft+=labels.astype(np.float64).sum(0)
    focused=np.unique(np.concatenate(focus)) if focus else np.empty(0,np.int64)
    return dict(counts=counts,ptr=ptr,focus_indices=focused,lengths=np.array([c.length for c in curves]),
        full_grid_soft_prior=soft/max(1,int(ptr[-1])),grid_step_mm=2.)


def mixture_probability(indices,total,focus,alpha):
    indices=np.asarray(indices,np.int64);focus=np.asarray(focus,np.int64)
    if total<=0 or not 0<=alpha<1:raise ValueError('Nonempty population and a nonzero uniform component required')
    if len(focus)==0:
        if alpha!=0:raise ValueError('Focus mixture cannot have mass without points')
        return np.full(len(indices),1./total)
    return np.full(len(indices),(1-alpha)/total)+np.isin(indices,focus)*(alpha/len(focus))


def grid_positions(layout,nodes,local):
    """Match numpy.linspace's step multiplication and explicit last endpoint."""
    nodes=np.asarray(nodes,np.int64);local=np.asarray(local,np.int64)
    last=layout['counts'][nodes]-1;length=layout['lengths'][nodes]
    q=local*(length/last)
    return np.where(local==last,length,q)


def draw(layout,seed,count=2048,focus_fraction=.5):
    if count<1 or not 0<=focus_fraction<1:raise ValueError('Positive count and uniform support required')
    total=int(layout['ptr'][-1]);focus=layout['focus_indices']
    if total<1:raise ValueError('Empty inference population')
    rng=np.random.default_rng(seed);nf=int(count*focus_fraction) if len(focus) else 0;alpha=nf/count
    indices=np.r_[rng.integers(total,size=count-nf),rng.choice(focus,size=nf,replace=True) if nf else np.empty(0,np.int64)]
    rng.shuffle(indices);probability=mixture_probability(indices,total,focus,alpha)
    nodes=np.searchsorted(layout['ptr'],indices,side='right')-1
    local=indices-layout['ptr'][nodes]
    q=grid_positions(layout,nodes,local)
    return dict(grid_indices=indices,nodes=nodes,q_mm=q,sampling_probability=probability,
        importance_weight=(1./total)/probability,focus_fraction=alpha)


def labels_for(profiles,supervision,nodes,q):
    nodes=np.asarray(nodes,np.int64);q=np.asarray(q,np.float64)
    y=np.zeros((len(q),5),np.float32);known=np.ones((len(q),5),bool);known[:,4]=False
    offsets=np.zeros((len(q),2),np.float32);offset_known=np.zeros((len(q),2),bool)
    for node in np.unique(nodes):
        ids=np.flatnonzero(nodes==node);events=supervision['by_profile'].get(str(int(node)),[])
        yy,oo,mm=event_targets(profiles[int(node)],events,q[ids]);y[ids,:4]=yy;offsets[ids]=oo;offset_known[ids]=mm
        directions=np.zeros((len(ids),2),bool)
        for event in events:
            inside=np.zeros(len(ids),bool)
            for lo,hi in event['intervals']:inside|=(q[ids]>=lo)&(q[ids]<=hi)
            directions[inside,int(event['target_direction_forward'])]=True
        valid=directions.sum(1)==1;known[ids[valid],4]=True;y[ids[valid],4]=directions[valid,1]
    return dict(target=y,target_mask=known,offset=offsets,offset_mask=offset_known)


def controls():
    # Enumerating every possible draw proves the importance identity exactly;
    # the same check covers both rare soft targets and background risk.
    population=np.arange(101,dtype=np.int64);focus=np.array([3,4,5,6,7]);alpha=.5
    p=mixture_probability(population,len(population),focus,alpha);w=1./len(population)/p
    assert abs(p.sum()-1)<1e-12
    values=np.sin(population*.13)**2+np.exp(-((population-5)/2)**2)
    assert abs(np.sum(p*w*values)-values.mean())<1e-12
    from weld_pipeline.sjg_local_curve_supervision_915 import compatible_interval
    profiles=[dict(representation=dict(kind='line',endpoints_mm=[[0,0,0],[100,0,0]])),
        dict(representation=dict(kind='line',endpoints_mm=[[0,20,0],[2.+1e-12,20,0]]))]
    event=compatible_interval(profiles[0],dict(kind='line',endpoints_mm=[[20,0,0],[40,0,0]]))
    supervision=dict(by_profile={'0':[event]});layout=grid_layout(profiles,supervision)
    assert layout['counts'].tolist()==[51,3]
    sample=draw(layout,1007,count=2048)
    for node in range(2):
        ids=sample['nodes']==node;reference=CurveCarrier(profiles[node]['representation']).grid(2.)
        local=sample['grid_indices'][ids]-layout['ptr'][node]
        assert np.array_equal(sample['q_mm'][ids],reference[local])
    assert np.isfinite(sample['importance_weight']).all() and (sample['sampling_probability']>0).all()
    labels=labels_for(profiles,supervision,sample['nodes'],sample['q_mm'])
    assert not labels['target_mask'][sample['nodes']==1,4].any()
    empty=grid_layout(profiles,dict(by_profile={}))
    assert np.all(draw(empty,7)['importance_weight']==1.)
    return dict(exact_importance_expectation_matches_full_grid=True,all_background_points_have_nonzero_probability=True,
        query_positions_match_actual_inference_grid=True,unknown_direction_keeps_mask=True,
        empty_focus_reverts_to_uniform=True,optimizer_updates=0,current_model_and_thresholds_unchanged=True)
