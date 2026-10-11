"""Pro365 prefix minibatch control, four passes, final clean-state diagnostic."""
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


def inputs(dest,key,mean,scale):
    folder=dest/f'source_{key}';receipt=read(folder/'receipt.json')
    assert digest(folder/'unary.npy')==receipt['unary_sha256']
    assert digest(folder/'decoded.json')==receipt['decoded_sha256']
    u=(np.load(folder/'unary.npy')-mean)/scale
    compat=read(folder/'compatibility.json');assert compat['prediction_sha256']==receipt['decoded_sha256']
    parents,points=load_parents(folder)
    return u,parents,read(folder/'parent_metadata.json'),compat,points


def check(dest):
    source=OUT/'run365';model,optimizer,mean,scale,restored=restore_final(source)
    cache=RelationColumnCache(dest/'control_relation_cache',bindings(dest,KEYS))
    reused=reuse_columns(source/'training_relation_cache',cache)
    try:
        u,parents,metadata,compat,points=inputs(dest,'10',mean,scale)
        trajectory=freeze_trajectory(model,u,parents,metadata,compat,'10',cache,0)
        write(dest/'control_trajectory_10.json',trajectory)
        result=control(model,u,parents,metadata,compat,'10',cache,trajectory)
        assert optimizer.step_count==336
        result.update(initial_step=336,parameters_moments_normalization_rng_restored_exact=True,
            source_checkpoint_sha256=restored['checkpoint_sha256'],cache_reuse=reused)
        write(dest/'prefix_minibatch_control.json',result)
        points._mmap.close();print('PREFIX_MINIBATCH_CONTROL_PASSED',result,flush=True)
    finally:write(dest/'control_relation_cache_receipt.json',cache.receipt())


def train(dest,limit):
    begin=time.perf_counter();source=OUT/'run365'
    assert read(dest/'prefix_minibatch_control.json')['status']=='FOUR_BATCH_FROZEN_GRADIENT_MEAN_CONTROL_PASSED'
    model,optimizer,mean,scale,restored=restore_final(source);trace=list(restored['trace'])
    np.savez(dest/'selector_normalization.npz',mean=mean,scale=scale)
    cache=RelationColumnCache(dest/'training_relation_cache',bindings(dest,KEYS))
    reused=reuse_columns(dest/'control_relation_cache',cache)
    assert reused['disk_bytes']==read(dest/'control_relation_cache_receipt.json')['disk_bytes']
    write(dest/'relation_cache_reuse.json',reused)
    write(dest/'minibatch_initialization.json',dict(source='run365',initial_step=336,
        parameters_moments_normalization_rng_restored_exact=True,source_checkpoint_sha256=restored['checkpoint_sha256']))
    trajectories=dest/'trajectories';trajectories.mkdir()
    def deadline():
        if time.perf_counter()-begin>=limit:raise TimeoutError('Shared trial time exhausted')
        check_resources()
    try:
        for traversal in range(4):
            for key in KEYS:
                deadline();u,parents,metadata,compat,points=inputs(dest,key,mean,scale)
                trajectory=freeze_trajectory(model,u,parents,metadata,compat,key,cache,traversal)
                trajectory.update(starting_Adam_step=optimizer.step_count,candidate_binding=restored['bindings'][key])
                path=trajectories/f'pass_{traversal+1:02}_{key}.json';write(path,trajectory);trajectory_hash=digest(path)
                for batch in range(4):
                    deadline()
                    loss,gradient=evaluate_prefixes(model,u,parents,metadata,compat,key,cache,trajectory,batch_weights(trajectory,batch))
                    update=optimizer.update(model,gradient);count=optimizer.step_count
                    assert 337<=count<=448
                    trace.append(dict(epoch=49+traversal,stage_epoch=traversal+1,source=key,batch=batch+1,
                        prefixes=trajectory['prefix_count'],batch_prefixes=trajectory['batches'][batch],
                        weighted_batch_loss=loss,trajectory_sha256=trajectory_hash,**update))
                    checkpoint={f'parameter_{k}':v for k,v in model.parameters.items()}
                    checkpoint.update({f'first_{k}':v for k,v in optimizer.first.items()})
                    checkpoint.update({f'second_{k}':v for k,v in optimizer.second.items()})
                    checkpoint.update(updates=np.array(count,dtype=np.int64),mean=mean,scale=scale)
                    atomic_binary_write(dest/'selector_resume_checkpoint.npz',lambda stream:np.savez(stream,**checkpoint))
                    done=count-336;visit=done//4
                    next_position=None if done==112 else dict(traversal_zero_based=visit//7,source=KEYS[visit%7],batch_zero_based=done%4,
                        frozen_trajectory_path=str(path.relative_to(dest)) if done%4 else None,
                        frozen_trajectory_sha256=trajectory_hash if done%4 else None)
                    write(dest/'selector_resume_checkpoint.json',dict(status='RESUME_ONLY_NOT_FINAL_MODEL',updates=count,
                        registered_updates=448,stage_initial_update=336,training_stage='four_prefix_minibatches_per_visit',
                        next_position=next_position,current_trajectory_path=str(path.relative_to(dest)),current_trajectory_sha256=trajectory_hash,
                        initialization_rng_state=model.resume_rng.bit_generator.state,random_draws_after_initialization=0,
                        checkpoint_sha256=digest(dest/'selector_resume_checkpoint.npz'),bindings=restored['bindings'],trace=trace,
                        training_seconds=restored['training_seconds']+time.perf_counter()-begin,current_process_seconds=time.perf_counter()-begin))
                    write(dest/'selector_progress.json',dict(status='PARTIAL_STAGE_UPDATES_NOT_FINAL_MODEL',updates=count,
                        new_updates=done,registered_new_updates=112,last=trace[-1],seconds=time.perf_counter()-begin))
                    print('PREFIX_EPOCH',traversal+1,'/4','PIECE',key,'BATCH',batch+1,'/4',
                        'NEW_UPDATE',done,'/112','ADAM_STEP',count,'WEIGHTED_LOSS',loss,flush=True)
                del parents;points._mmap.close();gc.collect()
        assert optimizer.step_count==448
        model.save(dest/'selector.npz');loaded=NearestParentScorer.load(dest/'selector.npz')
        assert all(np.array_equal(v,loaded.parameters[k]) for k,v in model.parameters.items())
        write(dest/'selector_training.json',dict(status='FINAL_SET_SELECTOR_TRAINED',updates=448,new_updates=112,
            completed_before_stage=336,stage_epochs=4,total_epochs=52,batches_per_piece=4,trace=trace,
            bindings=restored['bindings'],loss_kind='best_legal_vs_best_illegal_softplus',
            selector_sha256=digest(dest/'selector.npz'),normalization_sha256=digest(dest/'selector_normalization.npz'),
            seconds=time.perf_counter()-begin,DEV_reads=0,only_final_weights_evaluated=True,optimizer_moments_reset=False))
    finally:write(dest/'training_relation_cache_receipt.json',cache.receipt())


def diagnose(dest):
    model=NearestParentScorer.load(dest/'selector.npz')
    assert digest(dest/'selector.npz')==read(dest/'selector_training.json')['selector_sha256']
    with np.load(dest/'selector_normalization.npz') as z:mean=z['mean'].copy();scale=z['scale'].copy()
    cache=RelationColumnCache(dest/'final_diagnostic_cache',bindings(dest,['10']))
    reuse_columns(dest/'training_relation_cache',cache)
    try:
        u,parents,metadata,compat,points=inputs(dest,'10',mean,scale)
        t=freeze_trajectory(model,u,parents,metadata,compat,'10',cache,0)
        r=dict(status='ZERO_UPDATE_FINAL_CLEAN_ORACLE_DIAGNOSTIC',source='10',prefixes=t['prefix_count'],
            illegal_top1=sum(v['illegal_top1'] for v in t['diagnostics']),
            premature_STOP_states=sum(v['premature_STOP'] for v in t['diagnostics']),
            final_STOP_correct=not t['diagnostics'][-1]['illegal_top1'],baseline_illegal_top1=51,
            baseline_prefixes=124,actual_model_updates=0,scope='TRAIN label-assisted diagnostic, not extraction accuracy')
        write(dest/'final_clean_oracle_diagnosis.json',r);print('FINAL_CLEAN_ORACLE_DIAGNOSTIC',r,flush=True)
        points._mmap.close()
    finally:write(dest/'final_diagnostic_cache_receipt.json',cache.receipt())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--stage',required=True)
    ap.add_argument('--source',default='');ap.add_argument('--limit-seconds',type=float,required=True)
    args=ap.parse_args();dest=OUT/args.run;begin=time.perf_counter()
    try:
        check_resources()
        if args.stage=='minibatch_control':check(dest)
        elif args.stage=='minibatch_train':train(dest,args.limit_seconds)
        elif args.stage=='clean_diagnostic':diagnose(dest)
        else:raise ValueError(args.stage)
        write(dest/f'stage_{args.stage}_{args.source}_complete.json',dict(status='STAGE_COMPLETE',seconds=time.perf_counter()-begin,worker_source_sha256=digest(__file__)))
    except BaseException:
        write(dest/f'stage_{args.stage}_{args.source}_failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-begin));raise


if __name__=='__main__':main()
