"""Finite-pool interval-set loss; native identities and whole-parent parity.

The score inside every hinge is the deployment SUM, never a length/count mean.
Only the outer supervision responsibility is normalized by original children.
This is a carrier training constraint, not the final whole-parent D3 metric.
"""
from functools import lru_cache
import numpy as np


def sparse_dp(pairs, scores):
    """Exact sparse version of 1248 DP: lowest start tie, skip wins equality."""
    if not len(pairs):return ()
    cats=np.argmax(scores,axis=1);values=scores[np.arange(len(pairs)),cats]
    ends=np.unique(pairs[:,2]);best={0:(0.,())};last=(0.,())
    # A sparse boundary has the same prefix optimum as the dense skip chain.
    boundaries=np.unique(pairs[:,1:3]);history_values=[];history_paths=[]
    for boundary in boundaries:
        ids=np.flatnonzero(pairs[:,2]==boundary)
        chosen=last
        for i in ids[np.argsort(pairs[ids,1],kind='stable')]:
            at=int(np.searchsorted(boundaries[:len(history_values)],pairs[i,1],side='right'))-1
            prior=(history_values[at],history_paths[at]) if at>=0 else (0.,())
            value=prior[0]+float(values[i])
            if value>chosen[0]:chosen=(value,prior[1]+((int(i),int(cats[i])),))
        history_values.append(chosen[0]);history_paths.append(chosen[1]);last=chosen
    return last[1]


def compatible_options(record, pair_index, category):
    return record['compat'].get((int(pair_index),int(category)),())


def legal_best(record,pairs,scores):
    """Ordered exactly-one-per-owner DP, with a consistent parity per parent.

    Drop a parent's parity only after its final child in this carrier scope.
    Adjacent intervals are allowed; known physical gaps are not filled.
    """
    groups=record['groups']
    if not groups:return (),0.
    last={g['parent']:j for j,g in enumerate(groups)}
    states={(0,()):(0.,())}
    for position,g in enumerate(groups):
        new={}
        for i,c,parity in g['options']:
            for (end,bindings),(value,path) in states.items():
                if end>pairs[i,1]:continue
                bound=dict(bindings)
                if g['parent'] in bound and bound[g['parent']]!=parity:continue
                bound[g['parent']]=parity
                bound={p:v for p,v in bound.items() if last[p]>position}
                key=(int(pairs[i,2]),tuple(sorted(bound.items())))
                candidate=(value+float(scores[i,c]),path+((int(i),int(c)),))
                if key not in new or candidate[0]>new[key][0]:new[key]=candidate
        if not new:return None,-np.inf
        states=new
    value,path=max(states.values(),key=lambda v:v[0])
    return path,value


def delta(record, selection):
    """One-to-one ordered native matching, allowing only whole-parent flips."""
    original_count=len(selection)
    selection=tuple(v for v in selection if compatible_options(record,*v))
    groups=record['groups'];last={g['parent']:j for j,g in enumerate(groups)}
    @lru_cache(None)
    def match(k,j,bindings):
        if k==len(selection) or j==len(groups):return 0
        g=groups[j];bound=dict(bindings)
        nextbound=tuple((p,v) for p,v in bindings if last[p]>j)
        result=max(match(k+1,j,bindings),match(k,j+1,nextbound))
        for owner,parity in compatible_options(record,*selection[k]):
            if owner!=g['owner']:continue
            if g['parent'] in bound and bound[g['parent']]!=parity:continue
            b=dict(bound);b[g['parent']]=parity
            b=tuple(sorted((p,v) for p,v in b.items() if last[p]>j))
            result=max(result,1+match(k+1,j+1,b))
        return result
    matched=match(0,0,())
    return original_count+len(groups)-2*matched


def set_value(scores,selection):
    return sum(float(scores[i,c]) for i,c in selection)


def structured_score_gradient(scores,pairs,records,memory,update_memory=True):
    """Return loss and score gradient; UNKNOWN records create no gradient."""
    gradient=np.zeros_like(scores);loss=0.;trace=[]
    for r in records:
        if not r['eligible']:continue
        ids=r['indices'];local_pairs=pairs[ids];local_scores=scores[ids]
        gold,gold_value=legal_best(r,local_pairs,local_scores)
        assert gold is not None and delta(r,gold)==0
        selection=sparse_dp(local_pairs,local_scores)
        pool=memory.setdefault(r['row'],{})
        candidates=dict(pool)
        candidates[selection]=delta(r,selection);candidates[gold]=0
        if update_memory:pool.update(candidates)
        # A newly observed complete legal alias is admitted to the legal side.
        legal=[y for y,d in candidates.items() if d==0]
        gold=max(legal,key=lambda y:set_value(local_scores,y));gold_value=set_value(local_scores,gold)
        worst=max(candidates,key=lambda y:set_value(local_scores,y)+candidates[y])
        gap=set_value(local_scores,worst)+candidates[worst]-gold_value
        weight=r['weight'];value=max(0.,gap)
        if gap>0:
            for i,c in worst:gradient[ids[i],c]+=weight
            for i,c in gold:gradient[ids[i],c]-=weight
        loss+=weight*value
        trace.append(dict(row=r['row'],loss=value,weight=weight,current_delta=candidates[selection],
                          current_sum=set_value(local_scores,selection),legal_sum=gold_value,
                          worst_delta=candidates[worst],competitors=len(candidates)))
    return loss,gradient,trace


def controls():
    """Only new loss/DP tests. No repeat of old integral/model controls."""
    pairs=np.array([[0,0,4],[0,0,2],[0,2,4],[0,0,4],[0,0,1],[0,3,4]])
    def record(groups):
        compat={}
        for g in groups:
            for i,c,p in g['options']:compat.setdefault((i,c),[]).append((g['owner'],p))
        return dict(row=0,indices=np.arange(len(pairs)),groups=groups,compat=compat,eligible=True,weight=1.)
    full=record([dict(owner=0,parent=0,options=[(0,0,False),(0,2,True),(3,0,False),(3,2,True)])])
    scores=np.full((6,4),-20.);scores[0,0]=1.;scores[1,0]=2.3;scores[2,0]=2.1;scores[3,0]=.7
    mem={};value,grad,trace=structured_score_gradient(scores,pairs,[full],mem)
    assert value>0 and grad[1,0]>0 and grad[2,0]>0 and grad[0,0]<0
    eps=1e-6;maximum=0.
    for i in range(6):
        for c in range(4):
            a=scores.copy();b=scores.copy();a[i,c]+=eps;b[i,c]-=eps
            finite=(structured_score_gradient(a,pairs,[full],mem,False)[0]-structured_score_gradient(b,pairs,[full],mem,False)[0])/(2*eps)
            maximum=max(maximum,abs(finite-grad[i,c]))
    assert maximum<1e-7
    adjacent=record([dict(owner=0,parent=0,options=[(1,0,False),(1,2,True)]),
                     dict(owner=1,parent=0,options=[(2,0,False),(2,2,True)])])
    assert delta(adjacent,((1,0),(2,0)))==0 and delta(adjacent,((1,2),(2,2)))==0
    assert delta(adjacent,((1,0),(2,2)))>0
    gap=record([dict(owner=0,parent=0,options=[(4,0,False),(4,2,True)]),
                dict(owner=1,parent=0,options=[(5,0,False),(5,2,True)])])
    assert delta(gap,((4,0),(5,0)))==0
    assert delta(full,((3,0),))==0 and delta(full,((0,2),))==0
    unknown=dict(full,eligible=False)
    u,g,_=structured_score_gradient(scores,pairs,[unknown],{})
    assert u==0 and not g.any()
    # Closed whole loop uses the same ordered arc-length endpoints, with a
    # permitted whole-parent reversal; no length rule exists in this loss.
    closed=dict(full,is_closed=True)
    assert delta(closed,((0,2),))==0
    # Dense reference copied literally from deployment tie logic, on UNIQUE pairs.
    rng=np.random.default_rng(1298);tested=0
    for n in (3,7,12):
        ii,jj=np.triu_indices(n,1);mask=rng.random(len(ii))>.4
        pp=np.c_[np.zeros(mask.sum(),int),ii[mask],jj[mask]]
        for mode in range(5):
            ss=rng.normal(size=(len(pp),4)) if mode else np.ones((len(pp),4))
            dense=np.full((n,n),-np.inf);cats=np.zeros((n,n),int)
            dense[pp[:,1],pp[:,2]]=ss.max(1);cats[pp[:,1],pp[:,2]]=ss.argmax(1)
            best=np.zeros(n);previous=np.full(n,-1,int);taken=np.zeros(n,bool)
            for j in range(1,n):
                values=best[:j]+dense[:j,j];i=int(values.argmax());best[j]=best[j-1];previous[j]=j-1
                if values[i]>best[j]:best[j]=values[i];previous[j]=i;taken[j]=True
            lookup={(int(i),int(j)):k for k,(_,i,j) in enumerate(pp)};chosen=[];j=n-1
            while j>0:
                if taken[j]:chosen.append((lookup[(int(previous[j]),j)],int(cats[previous[j],j])))
                j=int(previous[j])
            assert sparse_dp(pp,ss)==tuple(chosen[::-1]);tested+=1
    return dict(status='PASS',new_loss_finite_difference_max_error=maximum,sparse_dense_DP_cases=tested,
                fragmentation_penalized=True,adjacent_and_gap_preserved=True,whole_parent_reversal_preserved=True,
                independent_child_parity_rejected=True,correct_alias_legal=True,UNKNOWN_zero_gradient=True,
                closed_arc_length_representation_preserved=True,not_a_geometry_decoder_test=True,fit_calls=0)
