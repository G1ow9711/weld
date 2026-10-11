"""Pro366 TRAIN-only clean/recovery mixture; deployment is unchanged."""
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import check_resources
from weld_pipeline.sjg_parent_set_scorer_1159 import ParentSetScorer
from weld_pipeline.sjg_nearest_parent_features_1184 import NearestParentRelations
from weld_pipeline.sjg_parent_set_oracle_1158 import ParentSetOracle
from weld_pipeline.sjg_prefix_minibatch_1189 import parameter_digest,partition,mask_digest,evaluate_prefixes,batch_weights


def recovery_action(scores,selected,allowed):
    predicted=ParentSetScorer.choose_next(scores,selected)
    if allowed[-1]:return predicted,None,'TERMINAL_SUPERVISED_STOP'
    if predicted is not None:return predicted,predicted,'MODEL_CANDIDATE'
    legal=np.flatnonzero(allowed[:-1]);best=int(legal[np.argmax(scores[legal])])
    return None,best,'TRAIN_RECOVERY_AFTER_ILLEGAL_STOP'


def freeze_recovery(model,u,parents,metadata,compat,key,cache,traversal):
    initial_hash=parameter_digest(model)
    context=NearestParentRelations(parents,metadata,key,cache)
    oracle=ParentSetOracle(compat['compatible_targets'],compat['original_parents'])
    choices=[];masks=[];actions=[];n=len(parents)
    for prefix in range(n+1):
        check_resources();scores,_=model.forward(u,context.features(),context.selected)
        available,allowed=oracle.action_masks();masks.append(mask_digest(available,allowed))
        predicted,applied,reason=recovery_action(scores,context.selected,allowed)
        index=n if predicted is None else predicted
        row=dict(prefix=prefix,model_action=predicted,applied_action=applied,reason=reason,
            model_action_legal=bool(allowed[index]),selected_count=len(context.selected),matched_parents=oracle.match_count)
        actions.append(row)
        if applied is None:
            assert allowed[-1];break
        assert available[applied]
        row['applied_augments']=bool(oracle.select(applied));context.add(applied);choices.append(applied)
        assert context.selected==oracle.selection_order and len(choices)==len(set(choices))
    else:raise AssertionError('Recovery exceeded N+1 states')
    assert parameter_digest(model)==initial_hash
    T=len(masks);batches,counts=partition(T,traversal)
    return dict(status='FROZEN_COMPLETE_RECOVERY_TRAJECTORY',kind='recovery',source=key,
        traversal_zero_based=traversal,model_state_sha256=initial_hash,choices=choices,prefix_count=T,
        mask_sha256=masks,batches=batches,prefix_occurrences=counts.tolist(),actions=actions,
        final_STOP_included=True,worst_case_state_bound=n+1,GT_scope='TRAIN_recovery_only',actual_model_updates=0)


def mixture(model,u,parents,metadata,compat,key,cache,pair,batch=None):
    total_loss=0.;total_gradient={k:np.zeros_like(v) for k,v in model.parameters.items()}
    for name in ('clean','recovery'):
        trajectory=pair[name];T=trajectory['prefix_count']
        weights={t:.5/T for t in range(T)} if batch is None else {t:.5*w for t,w in batch_weights(trajectory,batch).items()}
        loss,gradient=evaluate_prefixes(model,u,parents,metadata,compat,key,cache,trajectory,weights)
        total_loss+=loss
        for k in total_gradient:total_gradient[k]+=gradient[k]
    return total_loss,total_gradient


def trajectory_control(pair,compat,frozen_choices):
    recovery=pair['recovery'];actions=recovery['actions'];first_intervention=None
    repeated=incompatible=0;oracle=ParentSetOracle(compat['compatible_targets'],compat['original_parents'])
    for row in actions:
        prefix=row['prefix']
        if first_intervention is None:
            expected=frozen_choices[prefix] if prefix<len(frozen_choices) else None
            assert row['model_action']==expected
        if row['reason']!='MODEL_CANDIDATE' and first_intervention is None:first_intervention=prefix
        applied=row['applied_action']
        if applied is None:
            assert oracle.action_masks()[1][-1];continue
        if row['reason']=='TRAIN_RECOVERY_AFTER_ILLEGAL_STOP':
            assert row['model_action'] is None and not oracle.action_masks()[1][-1]
            assert oracle.action_masks()[1][applied]
        augmented=oracle.select(applied);assert augmented==row['applied_augments']
        if not augmented:
            if compat['compatible_targets'][applied]:repeated+=1
            else:incompatible+=1
    assert recovery['prefix_count']<=len(compat['compatible_targets'])+1
    # The real run366 error prefix contains both kinds; they must be retained.
    assert repeated>=29 and incompatible>=3
    # Small graph: an incompatible selection remains, then an augmenting path reassigns owners.
    small=ParentSetOracle([[0,1],[0],[]],2)
    assert small.select(0) and not small.select(2) and small.select(1)
    assert small.left_match==[1,0,-1] and small.selected=={0,1,2}
    assert small.action_masks()[1][-1] and len(small.selection_order)+1==4
    try:small.select(0)
    except ValueError:pass
    else:raise AssertionError('Duplicate index accepted')
    stop=ParentSetOracle([[0],[]],1)
    pred,applied,reason=recovery_action(np.array([0.,-1.,2.]),[],stop.action_masks()[1])
    assert pred is None and applied==0 and reason=='TRAIN_RECOVERY_AFTER_ILLEGAL_STOP'
    duplicate=ParentSetOracle([[0],[0],[1]],2)
    assert duplicate.select(0) and not duplicate.select(1) and duplicate.select(2)
    return dict(actual_prefix_before_first_intervention_exact=True,first_intervention=first_intervention,
        repeated_candidates_retained=repeated,incompatible_candidates_retained=incompatible,
        maximum_matching_reassignment_and_unique_index_control=True,complete_terminal_STOP_recorded=True,
        recovery_states=recovery['prefix_count'])


def weight_control(model,u,parents,metadata,compat,key,cache,pair):
    before=parameter_digest(model)
    full_loss,full_gradient=mixture(model,u,parents,metadata,compat,key,cache,pair)
    batches=[mixture(model,u,parents,metadata,compat,key,cache,pair,batch) for batch in range(4)]
    loss_error=float(abs(sum(v[0] for v in batches)/4-full_loss))
    assert np.isclose(sum(v[0] for v in batches)/4,full_loss,rtol=1e-11,atol=1e-12)
    errors={}
    for k,g in full_gradient.items():
        average=sum(v[1][k] for v in batches)/4
        errors[k]=float(np.abs(average-g).max());assert np.allclose(average,g,rtol=1e-11,atol=1e-12)
    # Separate half-normalization stays correct for every pair of small lengths.
    for tc in (1,2,3):
        for tr in (1,2,3):
            for traversal in range(4):
                for length in (tc,tr):
                    divided,counts=partition(length,traversal);coefficient=np.zeros(length)
                    for batch in divided:
                        for t in batch:coefficient[t]+=.5/(length*counts[t])
                    assert np.allclose(coefficient,np.full(length,.5/length),rtol=0,atol=1e-15)
    clean=pair['clean'];T=clean['prefix_count']
    clean_loss,clean_gradient=evaluate_prefixes(model,u,parents,metadata,compat,key,cache,clean,{t:1./T for t in range(T)})
    same_loss,same_gradient=mixture(model,u,parents,metadata,compat,key,cache,{'clean':clean,'recovery':clean})
    assert np.isclose(same_loss,clean_loss,rtol=1e-11,atol=1e-12)
    assert all(np.allclose(same_gradient[k],g,rtol=1e-11,atol=1e-12) for k,g in clean_gradient.items())
    assert parameter_digest(model)==before
    return dict(loss_error=loss_error,gradient_max_errors=errors,small_T_independent_half_weights=True,
        identical_trajectories_reduce_to_clean=True,gradients_before_clipping=True,actual_model_updates=0)
