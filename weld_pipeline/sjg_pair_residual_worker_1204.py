"""Pro371 residual-only controls, training and label-free final selection."""
import argparse,gc,hashlib,shutil,time,traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources
from weld_pipeline.sjg_pair_residual_1203 import ResidualScorer,ResidualAdam,ResidualRelations,EncodedSet,SHAPES
from weld_pipeline.sjg_nearest_parent_scorer_1185 import NearestParentScorer
from weld_pipeline.sjg_nearest_parent_features_1184 import NearestParentRelations
from weld_pipeline.sjg_prefix_minibatch_1189 import restore_final,parameter_digest,partition,mask_digest,batch_weights
from weld_pipeline.sjg_parent_set_oracle_1158 import ParentSetOracle
from weld_pipeline.sjg_pairwise_oracle_loss_1181 import pairwise_oracle_loss
from weld_pipeline.sjg_mixed_recovery_1192 import recovery_action
from weld_pipeline.sjg_history_replay_worker_1198 import inputs
from weld_pipeline.sjg_recall_set_worker_1162 import load_parents
from weld_pipeline.sjg_synchronous_pool_1200 import ReadOnlyColumns,verify_columns_unchanged
from weld_pipeline.sjg_selector_resume_state_1178 import reuse_columns
from weld_pipeline.sjg_selector_trial_worker_1174 import bindings
from weld_pipeline.sjg_seven_relative_prepare_1146 import KEYS
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write


def initialization(dest):
    old=OUT/'run367';base,old_adam,mean,scale,receipt=restore_final(old,expected_step=560)
    assert receipt['bindings']=={k:dict(candidate_sha256=digest(dest/f'source_{k}/decoded.json'),
        compatibility_sha256=digest(dest/f'source_{k}/compatibility.json')) for k in KEYS}
    branch=ResidualScorer(base,digest(old/'selector.npz'))
    return branch,ResidualAdam(branch),mean,scale,receipt


def software_control():
    rng=np.random.Generator(np.random.PCG64(1204));model=ResidualScorer(NearestParentScorer(),'software')
    model.parameters['W2']=rng.normal(0,.03,16)
    n=5;raw=rng.uniform(0,2,(n,n,16));u=rng.normal(0,.2,(n,160));old=rng.normal(0,.2,(n,48));selected=[3,1,2]
    def pool_of():
        p=EncodedSet(model,n,lambda index:raw[:,index,:])
        for index in selected:p.add(index)
        return p
    empty=EncodedSet(model,n,lambda index:raw[:,index,:]);assert np.array_equal(empty.features(),np.zeros((n,16)))
    p=pool_of();all_h=np.stack([np.tanh(raw[:,i,:]@model.parameters['Wr']+model.parameters['br']) for i in selected],axis=1)
    assert np.allclose(p.features(),np.c_[all_h.max(1),all_h.mean(1)],rtol=0,atol=1e-15)
    tied=raw.copy();tied[:,1]=tied[:,3]
    tie=EncodedSet(model,n,lambda index:tied[:,index,:]);tie.add(3);tie.add(1);assert np.all(tie.argmax==1)
    delta=rng.normal(size=n+1)
    scores,forward=model.forward(u,(old,p.features(),p),selected);g=model.backward(forward,delta)
    assert scores[-1]==model.base.forward(u,old,selected)[0][-1]
    errors={};eps=1e-5
    def objective():
        pp=pool_of();return float(model.forward(u,(old,pp.features(),pp),selected)[0]@delta)
    for key,value in model.parameters.items():
        error=0.
        for index in np.ndindex(value.shape):
            previous=value[index];value[index]=previous+eps;plus=objective()
            value[index]=previous-eps;minus=objective();value[index]=previous
            error=max(error,abs((plus-minus)/(2*eps)-g[key][index]))
        errors[key]=error;assert error<5e-8,(key,error)
    # Same original max/mean/nearest, different joint distance-topology relation.
    a=np.zeros((1,3,16));a[0,:,:3]=np.array([.1,1.,2.])[:,None];a[0,1,15]=1
    b=a.copy();b[0,1,15]=0;b[0,2,15]=1
    assert np.array_equal(a.max(1),b.max(1)) and np.array_equal(a.mean(1),b.mean(1)) and np.array_equal(a[:,0],b[:,0])
    model.parameters['Wr'][:]=0;model.parameters['Wr'][0,0]=1;model.parameters['Wr'][15,0]=1
    encoded=[]
    for value in (a,b):
        h=np.tanh(value@model.parameters['Wr']+model.parameters['br']);encoded.append(np.c_[h.max(1),h.mean(1)])
    assert not np.allclose(*encoded)
    return dict(status='RESIDUAL_BRANCH_SOFTWARE_CONTROL_PASSED',empty_set_zero=True,max_tie_lowest_candidate_index=True,
        incremental_equals_full=True,finite_difference_coordinates=3017,finite_difference_max_errors=errors,
        original_summaries_equal_but_joint_encoded_summaries_different=True,extra_information_not_accuracy_proof=True,
        new_branch_training_updates=0)


def controls(dest):
    model,adam,mean,scale,receipt=initialization(dest);old=OUT/'run367'
    cache=ReadOnlyColumns(dest/'control_relation_cache',bindings(dest,KEYS))
    reuse_columns(OUT/'run371/training_relation_cache',cache)
    reference=NearestParentScorer.load(old/'selector.npz');rows=[]
    try:
        for key in ('21','BM79A_native_local','10'):
            u,parents,metadata,compat,points=inputs(dest,key,mean,scale)
            original=NearestParentRelations(parents,metadata,key,cache)
            wrapped=ResidualRelations(parents,metadata,model,key,cache)
            saved=read(old/f'test_{key}/final_prediction.json');choices=saved['selected_candidate_indices']
            h=hashlib.sha256()
            try:
                for i in range(len(choices)+1):
                    check_resources();a,_=reference.forward(u,original.features(),original.selected)
                    b,_=model.forward(u,wrapped.features(),wrapped.selected)
                    assert np.array_equal(a,b);h.update(b.tobytes())
                    expected=choices[i] if i<len(choices) else None
                    assert model.choose_next(b,wrapped.selected)==reference.choose_next(a,original.selected)==expected
                    if expected is not None:original.add(expected);wrapped.add(expected)
                rows.append(dict(source=key,prefixes=len(choices)+1,scores_sha256=h.hexdigest(),saved_actions_and_STOP_exact=True))
            finally:
                del original,wrapped,parents
                points._mmap.close()
        assert adam.step_count==0 and digest(old/'selector.npz')==model.base_sha
        for value in model.base.parameters.values():assert not value.flags.writeable
        write(dest/'zero_residual_equivalence.json',dict(status='FROZEN367_ALL_THREE_ACTUAL_PREFIXES_EXACT',cases=rows,
            score_reference='Original frozen forward recomputed; historical score arrays did not exist',
            base_Adam_step=560,new_branch_Adam_step=0,base_checkpoint_sha256=receipt['checkpoint_sha256'],
            parameters_moments_rng_normalization_restored=True))
        write(dest/'residual_branch_control.json',software_control())
        print('RESIDUAL_TWO_CONTROLS_PASSED',rows,flush=True)
    finally:write(dest/'control_relation_cache_receipt.json',cache.receipt())


def freeze(model,u,parents,metadata,compat,key,cache,traversal,recovery):
    before=parameter_digest(model);context=ResidualRelations(parents,metadata,model,key,cache)
    oracle=ParentSetOracle(compat['compatible_targets'],compat['original_parents'])
    choices=[];masks=[];actions=[]
    for prefix in range(len(parents)+1):
        check_resources();scores,_=model.forward(u,context.features(),context.selected)
        available,allowed=oracle.action_masks();masks.append(mask_digest(available,allowed))
        if recovery:predicted,applied,reason=recovery_action(scores,context.selected,allowed)
        else:
            predicted=model.choose_next(scores,context.selected);reason='CLEAN_BEST_LEGAL'
            legal=np.flatnonzero(allowed[:-1]);applied=None if allowed[-1] else int(legal[np.argmax(scores[legal])])
        row=dict(model_action=predicted,applied_action=applied,reason=reason);actions.append(row)
        if applied is None:assert allowed[-1];break
        assert available[applied];row['applied_augments']=bool(oracle.select(applied))
        if not recovery:assert row['applied_augments']
        context.add(applied);choices.append(applied)
    else:raise AssertionError('No complete terminal state')
    assert parameter_digest(model)==before
    T=len(masks);batches,counts=partition(T,traversal)
    return dict(source=key,traversal_zero_based=traversal,kind='recovery' if recovery else 'clean',
        model_state_sha256=before,base_sha256=model.base_sha,choices=choices,prefix_count=T,mask_sha256=masks,
        batches=batches,prefix_occurrences=counts.tolist(),actions=actions,actual_model_updates=0)


def mixture(model,u,parents,metadata,compat,key,cache,pair,batch):
    loss_sum=0.;grad={k:np.zeros_like(v) for k,v in model.parameters.items()}
    for name in ('clean','recovery'):
        t=pair[name];weights={p:.5*w for p,w in batch_weights(t,batch).items()}
        context=ResidualRelations(parents,metadata,model,key,cache)
        oracle=ParentSetOracle(compat['compatible_targets'],compat['original_parents'])
        for prefix in range(len(t['choices'])+1):
            check_resources();available,allowed=oracle.action_masks()
            assert mask_digest(available,allowed)==t['mask_sha256'][prefix]
            if prefix in weights:
                scores,forward=model.forward(u,context.features(),context.selected)
                loss,delta=pairwise_oracle_loss(scores,available,allowed);g=model.backward(forward,delta)
                loss_sum+=weights[prefix]*loss
                for k in grad:grad[k]+=weights[prefix]*g[k]
            if prefix==len(t['choices']):assert allowed[-1];break
            choice=t['choices'][prefix];assert available[choice]
            assert oracle.select(choice)==t['actions'][prefix]['applied_augments']
            context.add(choice)
    return loss_sum,grad


def train(dest,limit):
    start=time.perf_counter();old=OUT/'run367'
    assert read(dest/'zero_residual_equivalence.json')['status']=='FROZEN367_ALL_THREE_ACTUAL_PREFIXES_EXACT'
    assert read(dest/'residual_branch_control.json')['status']=='RESIDUAL_BRANCH_SOFTWARE_CONTROL_PASSED'
    model,adam,mean,scale,restored=initialization(dest)
    shutil.copy2(old/'selector.npz',dest/'base_selector.npz');shutil.copy2(old/'selector_normalization.npz',dest/'selector_normalization.npz')
    cache=ReadOnlyColumns(dest/'training_relation_cache',bindings(dest,KEYS));reuse_columns(dest/'control_relation_cache',cache)
    trajectories=dest/'trajectories';trajectories.mkdir();trace=[]
    def deadline():
        check_resources()
        if time.perf_counter()-start>=limit:raise TimeoutError('Shared residual trial budget exhausted')
    try:
        for e in range(4):
            for key in KEYS:
                deadline();u,parents,metadata,compat,points=inputs(dest,key,mean,scale)
                try:
                    pair={kind:freeze(model,u,parents,metadata,compat,key,cache,e,kind=='recovery') for kind in ('clean','recovery')}
                    assert pair['clean']['model_state_sha256']==pair['recovery']['model_state_sha256']
                    pair.update(candidate_binding=restored['bindings'][key],base_sha256=model.base_sha)
                    path=trajectories/f'pass_{e+1:02}_{key}.json';write(path,pair);sha=digest(path)
                    for batch in range(4):
                        deadline();loss,g=mixture(model,u,parents,metadata,compat,key,cache,pair,batch)
                        update=adam.update(model,g);count=adam.step_count
                        trace.append(dict(stage_epoch=e+1,source=key,batch=batch+1,weighted_batch_loss=loss,
                            trajectory_sha256=sha,clean_prefixes=pair['clean']['prefix_count'],recovery_prefixes=pair['recovery']['prefix_count'],**update))
                        data={f'parameter_{k}':v for k,v in model.parameters.items()}
                        data.update({f'first_{k}':v for k,v in adam.first.items()});data.update({f'second_{k}':v for k,v in adam.second.items()})
                        data.update(updates=np.array(count,dtype=np.int64),mean=mean,scale=scale)
                        atomic_binary_write(dest/'selector_resume_checkpoint.npz',lambda stream:np.savez(stream,**data))
                        visit=count//4
                        write(dest/'selector_resume_checkpoint.json',dict(status='RESIDUAL_ONLY_RESUME_CHECKPOINT',updates=count,
                            registered_updates=112,base_Adam_step=560,base_parameters_and_moments_frozen=True,
                            base_checkpoint_sha256=restored['checkpoint_sha256'],base_selector_sha256=model.base_sha,
                            checkpoint_sha256=digest(dest/'selector_resume_checkpoint.npz'),bindings=restored['bindings'],trace=trace,
                            initialization_rng_state=model.resume_rng.bit_generator.state,random_draws_after_initialization=0,
                            next_position=None if count==112 else dict(traversal_zero_based=visit//7,source=KEYS[visit%7],batch_zero_based=count%4,
                                trajectory_path=str(path) if count%4 else None,trajectory_sha256=sha if count%4 else None)))
                        write(dest/'selector_progress.json',dict(status='PARTIAL_RESIDUAL_NOT_FINAL',updates=count,registered_updates=112,
                            base_Adam_step=560,last=trace[-1],seconds=time.perf_counter()-start))
                        print('RESIDUAL_EPOCH',e+1,'/4','PIECE',key,'BATCH',batch+1,'/4','BRANCH_UPDATE',count,'/112',
                            'BASE_FROZEN_STEP',560,'LOSS',loss,flush=True)
                finally:
                    del parents
                    points._mmap.close()
                gc.collect()
        assert adam.step_count==112
        assert digest(old/'selector.npz')==model.base_sha==digest(dest/'base_selector.npz')
        assert digest(old/'selector_resume_checkpoint.npz')==restored['checkpoint_sha256']
        model.save(dest/'selector.npz');loaded=ResidualScorer.load(dest/'selector.npz',dest/'base_selector.npz')
        assert all(np.array_equal(v,loaded.parameters[k]) for k,v in model.parameters.items())
        write(dest/'selector_training.json',dict(status='FINAL_RESIDUAL_SELECTOR_TRAINED',updates=112,new_updates=112,
            base_Adam_step=560,base_selector_sha256=model.base_sha,base_parameters_and_moments_frozen=True,
            base_checkpoint_sha256=restored['checkpoint_sha256'],trace=trace,bindings=restored['bindings'],
            selector_sha256=digest(dest/'selector.npz'),normalization_sha256=digest(dest/'selector_normalization.npz'),
            seconds=time.perf_counter()-start,trainable_parameters=3017,DEV_reads=0,only_final_weights_evaluated=True))
        write(dest/'source_column_integrity_after.json',verify_columns_unchanged(OUT/'run371/training_relation_cache',KEYS))
    finally:write(dest/'training_relation_cache_receipt.json',cache.receipt())


def select(dest,key,limit):
    start=time.perf_counter();folder=dest/f'test_{key}';training=read(dest/'selector_training.json')
    assert digest(dest/'selector.npz')==training['selector_sha256']
    assert digest(dest/'selector_normalization.npz')==training['normalization_sha256']
    model=ResidualScorer.load(dest/'selector.npz',dest/'base_selector.npz')
    with np.load(dest/'selector_normalization.npz') as z:u=(np.load(folder/'unary.npy')-z['mean'])/z['scale']
    parents,points=load_parents(folder);context=ResidualRelations(parents,read(folder/'parent_metadata.json'),model)
    for step in range(len(parents)+1):
        check_resources()
        if time.perf_counter()-start>min(120.,limit):raise TimeoutError('Residual selection development time exhausted')
        scores,_=model.forward(u,context.features(),context.selected);chosen=model.choose_next(scores,context.selected)
        if chosen is None:break
        context.add(chosen)
    else:raise RuntimeError('Missing STOP after all candidates')
    indices=list(context.selected);del context,parents;points._mmap.close()
    original=read(folder/'decoded.json');result=dict(original);result['parents']=[original['parents'][i] for i in indices]
    result['total_prediction_children']=sum(p['child_count'] for p in result['parents'])+sum(v['counted_false_children'] for v in original['invalid_predictions'])
    result.update(runtime_model='FROZEN367_WITH_PAIR_SET_SIGNED_RESIDUAL',GT_reads=0)
    write(folder/'final_decoded.json',result)
    write(folder/'final_prediction.json',dict(status='LABEL_FREE_RESIDUAL_PREDICTION_FROZEN',source=key,
        selected_candidate_indices=indices,candidate_count=len(original['parents']),decoded_sha256=digest(folder/'final_decoded.json'),
        selector_sha256=training['selector_sha256'],base_selector_sha256=model.base_sha,GT_reads=0,seconds=time.perf_counter()-start))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--stage',required=True)
    ap.add_argument('--source',default='');ap.add_argument('--limit-seconds',type=float,required=True)
    a=ap.parse_args();dest=OUT/a.run;start=time.perf_counter()
    try:
        if a.stage=='residual_control':controls(dest)
        elif a.stage=='residual_train':train(dest,a.limit_seconds)
        elif a.stage=='select':select(dest,a.source,a.limit_seconds)
        else:raise ValueError(a.stage)
        write(dest/f'stage_{a.stage}_{a.source}_complete.json',dict(status='STAGE_COMPLETE',seconds=time.perf_counter()-start,worker_source_sha256=digest(__file__)))
    except BaseException:
        write(dest/f'stage_{a.stage}_{a.source}_failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-start));raise

if __name__=='__main__':main()
