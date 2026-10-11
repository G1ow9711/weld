"""Pro368 fixed fresh/history replay branch from run367 step560 to672."""
import argparse,gc,time,traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,check_resources
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_recall_set_worker_1162 import load_parents
from weld_pipeline.sjg_nearest_parent_scorer_1185 import NearestParentScorer
from weld_pipeline.sjg_prefix_minibatch_1189 import restore_final,freeze_trajectory,control,evaluate_prefixes,batch_weights
from weld_pipeline.sjg_relation_column_cache_1173 import RelationColumnCache
from weld_pipeline.sjg_selector_resume_state_1178 import reuse_columns
from weld_pipeline.sjg_selector_trial_worker_1174 import bindings
from weld_pipeline.sjg_seven_relative_prepare_1146 import KEYS
from weld_pipeline.sjg_history_replay_1197 import history_pool,bundle,replay_loss,replay_control
from weld_pipeline.sjg_mixed_recovery_1192 import freeze_recovery,mixture,trajectory_control,weight_control


def inputs(dest,key,mean,scale):
    folder=dest/f'source_{key}';receipt=read(folder/'receipt.json')
    assert digest(folder/'unary.npy')==receipt['unary_sha256']
    assert digest(folder/'decoded.json')==receipt['decoded_sha256']
    u=(np.load(folder/'unary.npy')-mean)/scale
    compat=read(folder/'compatibility.json');assert compat['prediction_sha256']==receipt['decoded_sha256']
    parents,points=load_parents(folder)
    return u,parents,read(folder/'parent_metadata.json'),compat,points


def check(dest):
    from weld_pipeline.sjg_nearest_parent_features_1184 import NearestParentRelations
    from weld_pipeline.sjg_prefix_minibatch_1189 import parameter_digest
    import hashlib,shutil
    source=OUT/'run367';model,optimizer,mean,scale,restored=restore_final(source,expected_step=560)
    actual={}
    for key in KEYS:
        folder=dest/f'source_{key}';receipt=read(folder/'receipt.json')
        assert digest(folder/'unary.npy')==receipt['unary_sha256']
        actual[key]=dict(candidate_sha256=digest(folder/'decoded.json'),compatibility_sha256=digest(folder/'compatibility.json'))
    assert actual==restored['bindings']
    cache=RelationColumnCache(dest/'control_relation_cache',bindings(dest,KEYS))
    reused=reuse_columns(OUT/'run368/training_relation_cache',cache)
    try:
        u,parents,metadata,compat,points=inputs(dest,'21',mean,scale)
        saved=read(source/'test_21/final_prediction.json');selected=saved['selected_candidate_indices']
        assert selected and saved['selector_sha256']==digest(source/'selector.npz')
        reference=NearestParentScorer.load(source/'selector.npz')
        uncached=NearestParentRelations(parents,metadata)
        cached=NearestParentRelations(parents,metadata,'21',cache)
        for index in selected:uncached.add(index);cached.add(index)
        before=parameter_digest(model)
        original_scores,_=reference.forward(u,uncached.features(),uncached.selected)
        restored_scores,_=model.forward(u,cached.features(),cached.selected)
        assert np.array_equal(original_scores,restored_scores)
        assert model.choose_next(restored_scores,cached.selected) is None
        assert reference.choose_next(original_scores,uncached.selected) is None
        assert optimizer.step_count==560 and parameter_digest(model)==before
        write(dest/'continuation_restore_control.json',dict(status='COMPLETE_560_STATE_RESTORED',
            initial_step=560,source_checkpoint_sha256=restored['checkpoint_sha256'],
            source_model_sha256=digest(source/'selector.npz'),bindings_exact=True,
            parameters_moments_normalization_rng_restored_exact=True,cache_reuse=reused,
            nonempty_prefix_source='21',selected=selected,final_saved_action='STOP',
            score_reference_provenance='Recomputed from archived final model and uncached relations; historical prediction stored actions, not score arrays',
            candidate_and_STOP_scores_exact=True,scores_sha256=hashlib.sha256(restored_scores.tobytes()).hexdigest(),
            scores=restored_scores.tolist(),actual_training_updates=0))
        shutil.copy2(source/'fixed_diagnostic_trajectories.json',dest/'fixed_diagnostic_trajectories.json')
        points._mmap.close()
        history_pool(dest,KEYS,restored['bindings'])
        u,parents,metadata,compat,points=inputs(dest,'10',mean,scale)
        clean=freeze_trajectory(model,u,parents,metadata,compat,'10',cache,0)
        recovery=freeze_recovery(model,u,parents,metadata,compat,'10',cache,0)
        pack=bundle(dest,'10',0,clean,recovery)
        control_result=replay_control(model,u,parents,metadata,compat,'10',cache,pack)
        assert optimizer.step_count==560
        write(dest/'history_replay_control.json',control_result)
        write(dest/'control_six_trajectories_10.json',pack)
        points._mmap.close();print('HISTORY_REPLAY_CONTROL_PASSED',control_result,flush=True)
    finally:write(dest/'control_relation_cache_receipt.json',cache.receipt())


def train(dest,limit):
    begin=time.perf_counter();source=OUT/'run367'
    assert read(dest/'history_replay_control.json')['status']=='SIX_TRAJECTORY_WEIGHT_AND_ORDER_CONTROL_PASSED'
    model,optimizer,mean,scale,restored=restore_final(source,expected_step=560);trace=list(restored['trace'])
    np.savez(dest/'selector_normalization.npz',mean=mean,scale=scale)
    cache=RelationColumnCache(dest/'training_relation_cache',bindings(dest,KEYS))
    reused=reuse_columns(dest/'control_relation_cache',cache)
    assert reused['disk_bytes']==read(dest/'control_relation_cache_receipt.json')['disk_bytes']
    write(dest/'relation_cache_reuse.json',reused)
    write(dest/'minibatch_initialization.json',dict(source='run367',initial_step=560,
        parameters_moments_normalization_rng_restored_exact=True,source_checkpoint_sha256=restored['checkpoint_sha256']))
    trajectories=dest/'trajectories';trajectories.mkdir()
    def deadline():
        if time.perf_counter()-begin>=limit:raise TimeoutError('Shared trial time exhausted')
        check_resources()
    try:
        for traversal in range(4):
            for key in KEYS:
                deadline();u,parents,metadata,compat,points=inputs(dest,key,mean,scale)
                clean=freeze_trajectory(model,u,parents,metadata,compat,key,cache,traversal)
                recovery=freeze_recovery(model,u,parents,metadata,compat,key,cache,traversal)
                assert clean['model_state_sha256']==recovery['model_state_sha256']
                trajectory=bundle(dest,key,traversal,clean,recovery)
                trajectory.update(starting_Adam_step=optimizer.step_count,candidate_binding=restored['bindings'][key])
                path=trajectories/f'pass_{traversal+1:02}_{key}.json';write(path,trajectory);trajectory_hash=digest(path)
                for batch in range(4):
                    deadline()
                    loss,gradient=replay_loss(model,u,parents,metadata,compat,key,cache,trajectory,batch)
                    update=optimizer.update(model,gradient);count=optimizer.step_count
                    assert 561<=count<=672
                    trace.append(dict(epoch=57+traversal,stage_epoch=traversal+1,source=key,batch=batch+1,
                        clean_prefixes=clean['prefix_count'],recovery_prefixes=recovery['prefix_count'],
                        clean_batch_prefixes=clean['batches'][batch],recovery_batch_prefixes=recovery['batches'][batch],
                        weighted_batch_loss=loss,trajectory_sha256=trajectory_hash,
                        six_trajectory_prefix_counts=[v['trajectory']['prefix_count'] for v in trajectory['entries']],**update))
                    checkpoint={f'parameter_{k}':v for k,v in model.parameters.items()}
                    checkpoint.update({f'first_{k}':v for k,v in optimizer.first.items()})
                    checkpoint.update({f'second_{k}':v for k,v in optimizer.second.items()})
                    checkpoint.update(updates=np.array(count,dtype=np.int64),mean=mean,scale=scale)
                    atomic_binary_write(dest/'selector_resume_checkpoint.npz',lambda stream:np.savez(stream,**checkpoint))
                    done=count-560;visit=done//4
                    next_position=None if done==112 else dict(traversal_zero_based=visit//7,source=KEYS[visit%7],batch_zero_based=done%4,
                        frozen_trajectory_path=str(path.relative_to(dest)) if done%4 else None,
                        frozen_trajectory_sha256=trajectory_hash if done%4 else None)
                    write(dest/'selector_resume_checkpoint.json',dict(status='RESUME_ONLY_NOT_FINAL_MODEL',updates=count,
                        registered_updates=672,stage_initial_update=560,training_stage='half_fresh_half_two_round_history_four_batches',
                        next_position=next_position,current_trajectory_path=str(path.relative_to(dest)),current_trajectory_sha256=trajectory_hash,
                        initialization_rng_state=model.resume_rng.bit_generator.state,random_draws_after_initialization=0,
                        checkpoint_sha256=digest(dest/'selector_resume_checkpoint.npz'),bindings=restored['bindings'],trace=trace,
                        training_seconds=restored['training_seconds']+time.perf_counter()-begin,current_process_seconds=time.perf_counter()-begin))
                    write(dest/'selector_progress.json',dict(status='PARTIAL_STAGE_UPDATES_NOT_FINAL_MODEL',updates=count,
                        new_updates=done,registered_new_updates=112,last=trace[-1],seconds=time.perf_counter()-begin))
                    print('HISTORY_REPLAY_EPOCH',traversal+1,'/4','PIECE',key,'BATCH',batch+1,'/4',
                        'NEW_UPDATE',done,'/112','ADAM_STEP',count,'WEIGHTED_LOSS',loss,flush=True)
                del parents;points._mmap.close();gc.collect()
        assert optimizer.step_count==672
        model.save(dest/'selector.npz');loaded=NearestParentScorer.load(dest/'selector.npz')
        assert all(np.array_equal(v,loaded.parameters[k]) for k,v in model.parameters.items())
        write(dest/'selector_training.json',dict(status='FINAL_SET_SELECTOR_TRAINED',updates=672,new_updates=112,
            completed_before_stage=560,stage_epochs=4,total_epochs=60,batches_per_piece=4,trace=trace,
            bindings=restored['bindings'],loss_kind='best_legal_vs_best_illegal_softplus',
            selector_sha256=digest(dest/'selector.npz'),normalization_sha256=digest(dest/'selector_normalization.npz'),
            seconds=time.perf_counter()-begin,DEV_reads=0,only_final_weights_evaluated=True,optimizer_moments_reset=False,history_runs=['run367','run368'],fresh_weight=.5,historical_weight=.5))
    finally:write(dest/'training_relation_cache_receipt.json',cache.receipt())


def diagnose(dest):
    from weld_pipeline.sjg_nearest_parent_features_1184 import NearestParentRelations
    from weld_pipeline.sjg_parent_set_oracle_1158 import ParentSetOracle
    from weld_pipeline.sjg_pairwise_oracle_loss_1181 import pairwise_oracle_loss
    fixed=read(dest/'fixed_diagnostic_trajectories.json');source=OUT/'run366'
    assert digest(source/'selector.npz')==fixed['model_sha256']
    assert digest(source/'test_10/final_prediction.json')==fixed['original_prediction_sha256']
    previous=OUT/'run367';assert digest(previous/'selector.npz')==read(previous/'selector_training.json')['selector_sha256']
    models={'run367':NearestParentScorer.load(previous/'selector.npz'),'current':NearestParentScorer.load(dest/'selector.npz')}
    assert digest(dest/'selector.npz')==read(dest/'selector_training.json')['selector_sha256']
    assert digest(dest/'selector_normalization.npz')==digest(source/'selector_normalization.npz')
    with np.load(dest/'selector_normalization.npz') as z:mean=z['mean'].copy();scale=z['scale'].copy()
    cache=RelationColumnCache(dest/'final_diagnostic_cache',bindings(dest,['10']))
    reuse_columns(dest/'training_relation_cache',cache);results=[]
    try:
        u,parents,metadata,compat,points=inputs(dest,'10',mean,scale)
        current_frozen=OUT/'run368/test_10/final_prediction.json'
        assert digest(current_frozen)==read(OUT/'run368/stop_and_missed_parent_diagnosis.json')['prediction_sha256']
        trajectories=[('run366_clean',fixed['clean_choices']),('run366_actual',fixed['actual_choices']),
            ('run368_actual',read(current_frozen)['selected_candidate_indices'])]
        for kind,choices in trajectories:
            context=NearestParentRelations(parents,metadata,'10',cache)
            oracle=ParentSetOracle(compat['compatible_targets'],compat['original_parents'])
            rows={name:[] for name in models}
            for prefix in range(len(choices)+1):
                available,allowed=oracle.action_masks();features=context.features()
                for name,model in models.items():
                    scores,_=model.forward(u,features,context.selected);choice=model.choose_next(scores,context.selected)
                    action=len(scores)-1 if choice is None else choice
                    loss,_=pairwise_oracle_loss(scores,available,allowed)
                    rows[name].append(dict(illegal=not bool(allowed[action]),loss=loss,
                        premature_STOP=choice is None and not bool(allowed[-1])))
                if prefix==len(choices):break
                oracle.select(choices[prefix]);context.add(choices[prefix])
            for name,values in rows.items():
                record=dict(frozen_trajectory=kind,model=name,prefixes=len(values),
                    illegal_top1=sum(v['illegal'] for v in values),mean_pairwise_loss=float(np.mean([v['loss'] for v in values])),
                    premature_STOP=sum(v['premature_STOP'] for v in values))
                results.append(record);print('FIXED_STATE_CROSS_SCORE',record,flush=True)
        assert results[0]['illegal_top1']==49 and results[2]['illegal_top1']==13 and results[4]['illegal_top1']==1
        write(dest/'final_fixed_state_diagnosis.json',dict(status='SAME_FROZEN_CLEAN_AND_ACTUAL_STATE_CROSS_SCORE',
            results=results,actual_model_updates=0,scope='TRAIN supervised diagnostic, not extraction accuracy'))
        points._mmap.close()
    finally:write(dest/'final_diagnostic_cache_receipt.json',cache.receipt())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--stage',required=True)
    ap.add_argument('--source',default='');ap.add_argument('--limit-seconds',type=float,required=True)
    args=ap.parse_args();dest=OUT/args.run;begin=time.perf_counter()
    try:
        check_resources()
        if args.stage=='replay_control':check(dest)
        elif args.stage=='replay_train':train(dest,args.limit_seconds)
        elif args.stage=='fixed_diagnostic':diagnose(dest)
        else:raise ValueError(args.stage)
        write(dest/f'stage_{args.stage}_{args.source}_complete.json',dict(status='STAGE_COMPLETE',seconds=time.perf_counter()-begin,worker_source_sha256=digest(__file__)))
    except BaseException:
        write(dest/f'stage_{args.stage}_{args.source}_failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-begin));raise


if __name__=='__main__':main()
