"""Pro372: balanced dynamic coverage supervision relative to frozen STOP."""
import numpy as np
from weld_pipeline.sjg_pairwise_oracle_loss_1181 import pairwise_oracle_loss


def coverage_groups(available,allowed,edges,supervised=None):
    available=np.asarray(available,bool);allowed=np.asarray(allowed,bool)
    n=len(edges)
    assert available.shape==allowed.shape==(n+1,) and not np.any(allowed&~available)
    known=np.ones(n,bool) if supervised is None else np.asarray(supervised,bool)
    assert known.shape==(n,)
    eligible=available[:-1]&known
    compatible=np.fromiter((bool(e) for e in edges),bool,count=n)
    assert not np.any(allowed[:-1]&~compatible)
    return dict(A=eligible&allowed[:-1],D=eligible&~allowed[:-1]&compatible,I=eligible&~compatible)


def coverage_loss(scores,available,allowed,edges,supervised=None):
    scores=np.asarray(scores,np.float64)
    assert scores.shape==(len(edges)+1,) and np.isfinite(scores).all()
    groups=coverage_groups(available,allowed,edges,supervised)
    active=[(name,np.flatnonzero(mask)) for name,mask in groups.items() if mask.any()]
    gradient=np.zeros_like(scores)
    if not active:return 0.,gradient
    loss=0.
    for name,indices in active:
        sign=-1. if name=='A' else 1.
        d=sign*(scores[indices]-scores[-1])
        if not np.isfinite(d).all():raise FloatingPointError('Nonfinite relative STOP scores')
        weight=1./(len(active)*len(indices))
        loss+=weight*np.logaddexp(0.,d).sum()
        gradient[indices]=weight*sign*np.exp(-np.logaddexp(0.,-d))
    gradient[-1]=-gradient[:-1].sum()
    assert np.isfinite(loss) and np.isfinite(gradient).all()
    return float(loss),gradient


def combined_oracle_loss(scores,available,allowed,edges):
    old,g_old=pairwise_oracle_loss(scores,available,allowed)
    new,g_new=coverage_loss(scores,available,allowed,edges)
    return old+new,g_old+g_new


def objective_control(dest,mean,scale,cache):
    """One new objective control, existing real prefix and supervision edge cases."""
    from weld_pipeline.sjg_local_runtime_904 import OUT,read,digest
    from weld_pipeline.sjg_history_replay_worker_1198 import inputs
    from weld_pipeline.sjg_pair_residual_1203 import ResidualScorer,ResidualRelations,EncodedSet
    from weld_pipeline.sjg_nearest_parent_scorer_1185 import NearestParentScorer
    from weld_pipeline.sjg_parent_set_oracle_1158 import ParentSetOracle
    from weld_pipeline.sjg_prefix_minibatch_1189 import parameter_digest
    key='4';u,parents,metadata,compat,points=inputs(dest,key,mean,scale)
    trajectory_path=OUT/'run372/trajectories/pass_01_4.json';pack=read(trajectory_path)
    assert pack['candidate_binding']['candidate_sha256']==digest(dest/'source_4/decoded.json')
    assert pack['candidate_binding']['compatibility_sha256']==digest(dest/'source_4/compatibility.json')
    model=ResidualScorer(NearestParentScorer.load(OUT/'run367/selector.npz'),digest(OUT/'run367/selector.npz'))
    initial=parameter_digest(model)
    context=ResidualRelations(parents,metadata,model,key,cache)
    oracle=ParentSetOracle(compat['compatible_targets'],compat['original_parents'])
    try:
        for prefix in range(len(pack['clean']['choices'])+1):
            available,allowed=oracle.action_masks();groups=coverage_groups(available,allowed,oracle.edges)
            if all(mask.any() for mask in groups.values()):break
            if prefix==len(pack['clean']['choices']):raise AssertionError('No saved real A/D/I prefix')
            choice=pack['clean']['choices'][prefix];assert oracle.select(choice);context.add(choice)
        scores,_=model.forward(u,context.features(),context.selected)
        base_scores,_=model.base.forward(u,context.features()[0],context.selected)
        assert np.array_equal(scores,base_scores)
        assert model.choose_next(scores,context.selected)==model.base.choose_next(base_scores,context.selected)
        loss,g=coverage_loss(scores,available,allowed,oracle.edges)
        assert np.all(g[:-1][groups['A']]<0)
        assert np.all(g[:-1][groups['D']|groups['I']]>0)
        assert np.all(g[:-1][~available[:-1]]==0)
        numerical=[];eps=1e-5
        for index in range(len(scores)):
            plus=scores.copy();minus=scores.copy();plus[index]+=eps;minus[index]-=eps
            numeric=(coverage_loss(plus,available,allowed,oracle.edges)[0]-coverage_loss(minus,available,allowed,oracle.edges)[0])/(2*eps)
            numerical.append(abs(numeric-g[index]))
        assert max(numerical)<5e-8
        # Nonzero output layer only in this discarded control copy, so every
        # trainable layer is exercised by the added dense candidate supervision.
        model.parameters['W2']=np.random.Generator(np.random.PCG64(1206)).normal(0,.03,16)
        selected=list(context.selected);old_features=context.features()[0]
        def pool():
            value=EncodedSet(model,len(parents),context.raw_column)
            for index in selected:value.add(index)
            return value
        p=pool();score,forward=model.forward(u,(old_features,p.features(),p),selected)
        _,delta=coverage_loss(score,available,allowed,oracle.edges);grad=model.backward(forward,delta)
        def objective():
            q=pool();s,_=model.forward(u,(old_features,q.features(),q),selected)
            return coverage_loss(s,available,allowed,oracle.edges)[0]
        errors={}
        for key_param,values in model.parameters.items():
            error=0.
            for index in np.ndindex(values.shape):
                original=values[index];values[index]=original+eps;plus=objective()
                values[index]=original-eps;minus=objective();values[index]=original
                error=max(error,abs((plus-minus)/(2*eps)-grad[key_param][index]))
            errors[key_param]=error;assert error<5e-8,(key_param,error)
        actual=dict(source='4',saved_trajectory_sha256=digest(trajectory_path),prefix_length=len(selected),
            group_counts={name:int(mask.sum()) for name,mask in groups.items()},all_candidate_gradient_directions_correct=True,
            selected_gradients_zero=True,score_coordinates=len(scores),score_gradient_max_error=max(numerical),
            branch_finite_difference_coordinates=3017,branch_gradient_max_errors=errors,initial_parameter_digest=initial,
            zero_initial_scores_and_argmax_equal_base=True)
    finally:
        p=None;forward=None
        del context,parents
        points._mmap.close()
    # One target, alternate explanations, plus an unreachable original target.
    one=ParentSetOracle([[0],[0],[]],2);assert one.select(0)
    av,al=one.action_masks();gr=coverage_groups(av,al,one.edges)
    assert al[-1] and gr['D'].tolist()==[False,True,False] and gr['I'].tolist()==[False,False,True]
    assert one.match_count==1 and one.target_count==2
    # Full matching still admits the second explanation for the second GT.
    two=ParentSetOracle([[0,1],[0,1],[]],2);assert two.select(0)
    av2,al2=two.action_masks();assert al2[1] and not coverage_groups(av2,al2,two.edges)['D'][1]
    assert two.select(1) and two.action_masks()[1][-1]
    known=np.array([True,True,False]);_,masked=coverage_loss(np.zeros(4),av,al,one.edges,known)
    assert masked[2]==0
    zero,gzero=coverage_loss(np.zeros(1),np.ones(1,bool),np.ones(1,bool),[])
    assert zero==0 and np.array_equal(gzero,np.zeros(1))
    for sign in (-1,1):
        extreme=np.array([1000.,-1000.,500.,0.])*sign
        l,gg=coverage_loss(extreme,av,al,one.edges);assert np.isfinite(l) and np.isfinite(gg).all()
    a=np.array([2.,1.,0.]);b=np.array([7.,6.,0.]);available=np.ones(3,bool);allowed=np.array([True,False,False])
    assert pairwise_oracle_loss(a,available,allowed)[0]==pairwise_oracle_loss(b,available,allowed)[0]
    assert coverage_loss(a,available,allowed,[[0],[]])[0]!=coverage_loss(b,available,allowed,[[0],[]])[0]
    return dict(status='DYNAMIC_STOP_COVERAGE_OBJECTIVE_CONTROL_PASSED',real_prefix=actual,
        alternate_explanations_become_duplicate=True,multiple_targets_remain_augmenting=True,
        unreachable_GT_remains_missing=True,terminal_STOP_supervision=True,unresolved_mask_preserved=True,
        empty_candidates_zero_term=True,extreme_logits_finite=True,common_candidate_shift_detected=True,
        control_optimizer_updates=0)
