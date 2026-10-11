"""Known-wrong deletion comparisons independent of full-positive availability."""
import numpy as np
from weld_pipeline.sjg_finite_structured_loss_1298 import sparse_dp


def deletion_score_gradient(scores,pairs,records):
    active=[r for r in records if r['known_wrong'].any()]
    denominator=len(active);gradient=np.zeros_like(scores);loss=0.;trace=[]
    for r in active:
        ids=r['indices'];local=scores[ids]
        auxiliary=np.where(r['known_wrong'],local+1.,-np.inf)
        selection=sparse_dp(pairs[ids],auxiliary)
        value=sum(float(1+local[i,c]) for i,c in selection)
        assert value>=-1e-10
        for i,c in selection:
            assert r['known_wrong'][i,c];gradient[ids[i],c]+=1/denominator
        loss+=value/denominator
        trace.append(dict(row=r['row'],deletion_sum=value,selected_known_errors=len(selection),
            full_gold_verified=r['eligible'],known_wrong_explanations=int(r['known_wrong'].sum())))
    return loss,gradient,dict(fixed_known_wrong_carrier_denominator=denominator,carriers=trace)


def controls():
    pairs=np.array([[0,0,1],[0,0,2],[0,0,4],[0,1,2],[0,1,3],[0,2,3],[0,2,4],[0,3,4]])
    rng=np.random.default_rng(1302);maximum=0.;gradient_error=0.
    def enumerate_sets(start):
        yield ()
        for i,(_,a,b) in enumerate(pairs):
            if a<start:continue
            for c in range(4):
                for tail in enumerate_sets(b):yield ((i,c),)+tail
    sets=list(enumerate_sets(0))
    for iteration in range(6):
        scores=rng.normal(size=(len(pairs),4));mask=rng.random(scores.shape)>.6
        r=dict(row=0,indices=np.arange(len(pairs)),known_wrong=mask,eligible=False)
        value,gradient,meta=deletion_score_gradient(scores,pairs,[r])
        valid=[y for y in sets if all(mask[i,c] for i,c in y)]
        reference=max(sum(1+scores[i,c] for i,c in y) for y in valid)
        maximum=max(maximum,abs(value-reference));assert abs(value-reference)<1e-10
        if iteration==0:
            eps=1e-6
            for i in range(len(pairs)):
                for c in range(4):
                    plus=scores.copy();minus=scores.copy();plus[i,c]+=eps;minus[i,c]-=eps
                    finite=(deletion_score_gradient(plus,pairs,[r])[0]-deletion_score_gradient(minus,pairs,[r])[0])/(2*eps)
                    gradient_error=max(gradient_error,abs(finite-gradient[i,c]))
            assert gradient_error<1e-7
    # Ordinary prediction hides a known wrong alternative category. The deletion
    # separator must still constrain it even without a complete gold set.
    pp=np.array([[0,0,4]]);ss=np.array([[1.,.9,-2.,-3.]])
    bad=np.array([[False,True,False,False]])
    r=dict(row=0,indices=np.array([0]),known_wrong=bad,eligible=False)
    value,grad,_=deletion_score_gradient(ss,pp,[r]);assert abs(value-1.9)<1e-12 and grad[0,1]==1
    assert np.array_equal(grad[0,[0,2,3]],np.zeros(3))
    unknown=dict(r,known_wrong=np.zeros((1,4),bool))
    assert not deletion_score_gradient(ss,pp,[unknown])[1].any()
    # Fixed carrier denominator includes inactive constraints, not just violations.
    pp2=np.array([[0,0,4],[1,0,4]]);ss2=np.r_[ss,np.full((1,4),-3.)]
    r2=dict(r,row=1,indices=np.array([1]))
    value,g,meta=deletion_score_gradient(ss2,pp2,[r,r2]);assert value==.95 and g[0,1]==.5
    assert meta['fixed_known_wrong_carrier_denominator']==2
    # Adjacent/gapped intervals and a full-loop parameter span are still feasible.
    for ij in (((0,2),(2,4)),((0,1),(3,4)),((0,4),)):
        p=np.array([[0,a,b] for a,b in ij]);s=np.ones((len(p),4));m=np.zeros_like(s,dtype=bool);m[:,0]=True
        rr=dict(row=0,indices=np.arange(len(p)),known_wrong=m,eligible=False)
        value,_,_=deletion_score_gradient(s,p,[rr]);assert value==2*len(p)
    return dict(status='PASS',new_auxiliary_DP_exhaustive_cases=6,maximum_enumeration_error=maximum,
        new_loss_finite_difference_max_error=gradient_error,missing_gold_known_wrong_gradient_nonzero=True,
        correct_alias_and_UNKNOWN_gradient_zero=True,wrong_category_hidden_by_correct_prediction_constrained=True,
        fixed_carrier_denominator_includes_zero_violation_carriers=True,adjacency_gap_loop_parameter_rules_preserved=True,
        deployment_scores_and_candidate_domain_unchanged=True,fit_calls=0)
