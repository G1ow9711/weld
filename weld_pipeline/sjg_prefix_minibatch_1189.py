"""Pro365: fixed clean trajectories, four balanced prefix minibatches per visit."""
import hashlib
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,read,write,digest,check_resources
from weld_pipeline.sjg_nearest_parent_scorer_1185 import NearestParentScorer,NearestAdamState,SHAPES
from weld_pipeline.sjg_nearest_parent_features_1184 import NearestParentRelations
from weld_pipeline.sjg_parent_set_oracle_1158 import ParentSetOracle
from weld_pipeline.sjg_pairwise_oracle_loss_1181 import pairwise_oracle_loss


def restore_final(source,expected_step=336):
    training=read(source/'selector_training.json');receipt=read(source/'selector_resume_checkpoint.json')
    assert training['updates']==receipt['updates']==expected_step
    assert receipt['registered_updates']==expected_step
    if 'next_position' in receipt:assert receipt['next_position'] is None
    else:assert receipt['next_epoch'] is None and receipt['next_source'] is None
    assert digest(source/'selector.npz')==training['selector_sha256']
    assert digest(source/'selector_resume_checkpoint.npz')==receipt['checkpoint_sha256']
    assert digest(source/'selector_normalization.npz')==training['normalization_sha256']
    assert digest(source/'source_snapshot/sjg_nearest_parent_scorer_1185.py')==digest(ROOT/'weld_pipeline/sjg_nearest_parent_scorer_1185.py')
    assert receipt['bindings']==training['bindings'] and len(receipt['trace'])==expected_step
    assert [r['step'] for r in receipt['trace']]==list(range(1,expected_step+1))
    model=NearestParentScorer.load(source/'selector.npz');optimizer=NearestAdamState(model)
    with np.load(source/'selector_normalization.npz') as z:mean=z['mean'].copy();scale=z['scale'].copy()
    assert mean.shape==scale.shape==(160,) and np.isfinite(mean).all() and np.isfinite(scale).all() and np.all(scale>0)
    with np.load(source/'selector_resume_checkpoint.npz') as data:
        expected={f'{prefix}_{key}' for prefix in ('parameter','first','second') for key in SHAPES}|{'updates','mean','scale'}
        assert set(data.files)==expected and int(data['updates'])==expected_step
        assert np.array_equal(mean,data['mean']) and np.array_equal(scale,data['scale'])
        for key,shape in SHAPES.items():
            assert np.array_equal(model.parameters[key],data[f'parameter_{key}'])
            for prefix,target in (('first',optimizer.first),('second',optimizer.second)):
                value=data[f'{prefix}_{key}']
                assert value.shape==shape and value.dtype==np.float64 and np.isfinite(value).all()
                if prefix=='second':assert np.all(value>=0)
                target[key]=value.copy()
    assert receipt['random_draws_after_initialization']==0
    assert receipt['initialization_rng_state']==model.initialization_rng_state
    model.resume_rng=np.random.Generator(np.random.PCG64(0))
    model.resume_rng.bit_generator.state=receipt['initialization_rng_state'];optimizer.step_count=expected_step
    return model,optimizer,mean,scale,receipt


def parameter_digest(model):
    h=hashlib.sha256()
    for key in sorted(model.parameters):h.update(key.encode());h.update(model.parameters[key].tobytes())
    return h.hexdigest()


def partition(T,traversal):
    if T<1:raise ValueError('Empty trajectory')
    batches=[[] for _ in range(4)];counts=np.zeros(T,dtype=np.int64)
    # For T<4, repeated slots cyclically fill empty batches; prefix identities stay fixed.
    for slot in range(max(T,4)):
        prefix=slot%T;batches[(slot+traversal)%4].append(prefix);counts[prefix]+=1
    assert all(batches) and np.all(counts>0)
    return batches,counts


def mask_digest(available,allowed):
    return hashlib.sha256(available.tobytes()+allowed.tobytes()).hexdigest()


def freeze_trajectory(model,u,parents,metadata,compat,key,cache,traversal):
    initial_hash=parameter_digest(model)
    context=NearestParentRelations(parents,metadata,key,cache)
    oracle=ParentSetOracle(compat['compatible_targets'],compat['original_parents'])
    choices=[];masks=[];diagnostics=[]
    while True:
        check_resources();scores,_=model.forward(u,context.features(),context.selected)
        available,allowed=oracle.action_masks();masks.append(mask_digest(available,allowed))
        chosen=model.choose_next(scores,context.selected);actual=len(scores)-1 if chosen is None else chosen
        diagnostics.append(dict(illegal_top1=not bool(allowed[actual]),premature_STOP=chosen is None and not bool(allowed[-1])))
        if allowed[-1]:break
        legal=np.flatnonzero(allowed[:-1]);best=int(legal[np.argmax(scores[legal])])
        assert oracle.select(best);context.add(best);choices.append(best)
    T=len(masks);batches,counts=partition(T,traversal)
    assert parameter_digest(model)==initial_hash
    return dict(status='FROZEN_COMPLETE_CLEAN_TRAJECTORY',source=key,traversal_zero_based=traversal,
        model_state_sha256=initial_hash,choices=choices,prefix_count=T,mask_sha256=masks,
        batches=batches,prefix_occurrences=counts.tolist(),diagnostics=diagnostics,
        final_STOP_included=True,GT_scope='TRAIN_supervision_only',actual_model_updates=0)


def evaluate_prefixes(model,u,parents,metadata,compat,key,cache,trajectory,weights):
    context=NearestParentRelations(parents,metadata,key,cache)
    oracle=ParentSetOracle(compat['compatible_targets'],compat['original_parents'])
    gradients={k:np.zeros_like(v) for k,v in model.parameters.items()};loss_sum=0.
    choices=trajectory['choices'];visited=[]
    for prefix in range(len(choices)+1):
        check_resources();available,allowed=oracle.action_masks()
        assert mask_digest(available,allowed)==trajectory['mask_sha256'][prefix]
        if prefix in weights:
            scores,forward=model.forward(u,context.features(),context.selected)
            loss,delta=pairwise_oracle_loss(scores,available,allowed)
            value=model.backward(forward,delta);weight=weights[prefix]
            loss_sum+=weight*loss
            for key2 in gradients:gradients[key2]+=weight*value[key2]
            visited.append(prefix)
        if prefix==len(choices):assert allowed[-1];break
        choice=choices[prefix]
        if trajectory.get('kind')=='recovery':
            assert available[choice]
            augmented=oracle.select(choice)
            assert augmented==trajectory['actions'][prefix]['applied_augments']
        else:assert allowed[choice] and oracle.select(choice)
        context.add(choice)
    assert set(visited)==set(weights)
    return loss_sum,gradients


def batch_weights(trajectory,batch):
    T=trajectory['prefix_count'];counts=trajectory['prefix_occurrences'];result={}
    for prefix in trajectory['batches'][batch]:result[prefix]=result.get(prefix,0.)+4./(T*counts[prefix])
    return result


def control(model,u,parents,metadata,compat,key,cache,trajectory):
    before=parameter_digest(model);T=trajectory['prefix_count']
    full_loss,full_gradient=evaluate_prefixes(model,u,parents,metadata,compat,key,cache,trajectory,{t:1./T for t in range(T)})
    losses=[];gradients=[]
    for batch in range(4):
        loss,gradient=evaluate_prefixes(model,u,parents,metadata,compat,key,cache,trajectory,batch_weights(trajectory,batch))
        losses.append(loss);gradients.append(gradient)
    loss_error=float(abs(np.mean(losses)-full_loss));assert np.isclose(np.mean(losses),full_loss,rtol=1e-11,atol=1e-12)
    errors={}
    for name,value in full_gradient.items():
        combined=sum(g[name] for g in gradients)/4
        errors[name]=float(np.max(np.abs(combined-value)))
        assert np.allclose(combined,value,rtol=1e-11,atol=1e-12)
    small=[]
    for length in (1,2,3):
        for traversal in range(4):
            batches,counts=partition(length,traversal)
            weights=np.zeros(length)
            for batch in batches:
                for prefix in batch:weights[prefix]+=1./(length*counts[prefix])
            assert np.allclose(weights,np.full(length,1./length),rtol=0,atol=1e-15)
            small.append(dict(T=length,traversal=traversal,batches=batches,counts=counts.tolist()))
    assert parameter_digest(model)==before
    return dict(status='FOUR_BATCH_FROZEN_GRADIENT_MEAN_CONTROL_PASSED',source=key,T=T,
        loss_error=loss_error,gradient_max_errors=errors,small_T_controls=small,
        gradients_compared_before_clipping=True,actual_model_updates=0,
        four_Adam_steps_not_claimed_equal_to_one=True)
