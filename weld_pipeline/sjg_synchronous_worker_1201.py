"""Pro369/370: fixed original pools, eight Adam probes, sixteen joint updates."""
import argparse,gc,time,traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,check_resources
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_nearest_parent_scorer_1185 import NearestParentScorer
from weld_pipeline.sjg_prefix_minibatch_1189 import restore_final,parameter_digest
from weld_pipeline.sjg_relation_column_cache_1173 import RelationColumnCache
from weld_pipeline.sjg_selector_resume_state_1178 import reuse_columns
from weld_pipeline.sjg_selector_trial_worker_1174 import bindings
from weld_pipeline.sjg_seven_relative_prepare_1146 import KEYS
from weld_pipeline.sjg_history_replay_worker_1198 import inputs
from weld_pipeline.sjg_history_replay_1197 import replay_loss
from weld_pipeline.sjg_synchronous_pool_1200 import freeze_pool,pool_for,full_pool,average_gradients,isolated_copy,ReadOnlyColumns,verify_columns_unchanged


def train(dest,limit):
    started=time.perf_counter();source=OUT/'run367'
    model,optimizer,mean,scale,restored=restore_final(source,expected_step=560);trace=list(restored['trace'])
    actual={k:dict(candidate_sha256=digest(dest/f'source_{k}/decoded.json'),
        compatibility_sha256=digest(dest/f'source_{k}/compatibility.json')) for k in KEYS}
    assert actual==restored['bindings']
    entries=freeze_pool(dest,KEYS,actual)
    np.savez(dest/'selector_normalization.npz',mean=mean,scale=scale)
    assert digest(dest/'selector_normalization.npz')==digest(source/'selector_normalization.npz')
    cache=ReadOnlyColumns(dest/'training_relation_cache',bindings(dest,KEYS))
    write(dest/'relation_cache_reuse.json',reuse_columns(OUT/'run369/training_relation_cache',cache))
    write(dest/'synchronous_initialization.json',dict(source='run367',initial_step=560,
        complete_parameters_moments_rng_normalization_restored=True,checkpoint_sha256=restored['checkpoint_sha256']))
    def deadline():
        check_resources()
        if time.perf_counter()-started>=limit:raise TimeoutError('Shared synchronous trial budget exhausted')
    def evaluate_one(candidate,key,gradient=False):
        deadline();u,parents,meta,compat,points=inputs(dest,key,mean,scale)
        try:return full_pool(candidate,u,parents,meta,compat,key,cache,pool_for(dest,key,0)[0],gradient)
        finally:
            del parents
            points._mmap.close()
    try:
        initial_hash=parameter_digest(model);initial=[];gradients=[]
        for key in KEYS:
            value,g=evaluate_one(model,key,True);initial.append(value);gradients.append(g)
            print('ADAM_IMPACT_BASELINE',key,value,flush=True)
        aggregate=average_gradients(gradients);impact=[];displacements={}
        for s,g in zip([*KEYS,'seven_mean'],[*gradients,aggregate]):
            deadline();probe,adam=isolated_copy(model,optimizer);update=adam.update(probe,g)
            assert adam.step_count==561 and optimizer.step_count==560 and parameter_digest(model)==initial_hash
            delta={k:probe.parameters[k]-model.parameters[k] for k in model.parameters}
            for k,v in delta.items():displacements[s+'__'+k]=v
            after=[]
            for j,key in enumerate(KEYS):
                value,_=evaluate_one(probe,key)
                after.append(dict(source=key,loss_before=initial[j]['loss'],loss_after=value['loss'],
                    loss_change=value['loss']-initial[j]['loss'],
                    linearized_gradient_dot_displacement=float(sum(np.sum(gradients[j][k]*delta[k]) for k in delta)),
                    illegal_before=initial[j]['illegal_top1'],illegal_after=value['illegal_top1'],
                    weighted_illegal_before=initial[j]['weighted_illegal_top1'],weighted_illegal_after=value['weighted_illegal_top1']))
            impact.append(dict(update_source=s,optimizer_update=update,
                parameter_displacement_l2=float(np.sqrt(sum(np.sum(v*v) for v in delta.values()))),targets=after))
            print('ADAM_IMPACT_PROBE',s,'LOSS_CHANGES',[round(v['loss_change'],7) for v in after],flush=True)
            del probe,adam;gc.collect()
        atomic_binary_write(dest/'diagnostic_parameter_displacements.npz',lambda stream:np.savez(stream,**displacements))
        write(dest/'adam_cross_impact.json',dict(status='EIGHT_ISOLATED_ONE_STEP_ADAM_PROBES_COMPLETE',baseline=initial,
            impact=impact,diagnostic_Adam_calls=8,formal_training_commits=0,initial_step=560,
            displacement_sha256=digest(dest/'diagnostic_parameter_displacements.npz'),probe_models_retained_for_extraction=False))
        del gradients,aggregate,displacements
        assert optimizer.step_count==560 and parameter_digest(model)==initial_hash
        for traversal in range(4):
            for batch in range(4):
                deadline();before=parameter_digest(model);values=[];local_gradients=[];pool_hashes={}
                for key in KEYS:
                    deadline();u,parents,meta,compat,points=inputs(dest,key,mean,scale)
                    try:
                        pack,sha=pool_for(dest,key,traversal)
                        loss,g=replay_loss(model,u,parents,meta,compat,key,cache,pack,batch)
                        values.append(dict(source=key,loss=loss));local_gradients.append(g);pool_hashes[key]=sha
                        assert parameter_digest(model)==before
                    finally:
                        del parents
                        points._mmap.close()
                g=average_gradients(local_gradients)
                if traversal==0 and batch==0:
                    errors={k:float(np.abs(g[k]-np.stack([v[k] for v in local_gradients]).mean(0)).max()) for k in g}
                    assert all(v<=1e-12 for v in errors.values())
                    write(dest/'synchronous_gradient_control.json',dict(status='SEVEN_UNCLIPPED_GRADIENT_MEAN_PASSED',
                        gradient_max_errors=errors,parameters_held_fixed_between_seven_sources=True,
                        per_source_clipping=False,mean_Adam_displacements=False,formal_training_commits=0))
                update=optimizer.update(model,g);count=optimizer.step_count;done=count-560
                assert done==traversal*4+batch+1
                trace.append(dict(epoch=57+traversal,stage_epoch=traversal+1,source='ALL_SEVEN_SYNCHRONOUS',
                    batch=batch+1,source_losses=values,mean_batch_loss=float(np.mean([v['loss'] for v in values])),
                    fixed_pool_sha256=pool_hashes,**update))
                checkpoint={f'parameter_{k}':v for k,v in model.parameters.items()}
                checkpoint.update({f'first_{k}':v for k,v in optimizer.first.items()})
                checkpoint.update({f'second_{k}':v for k,v in optimizer.second.items()})
                checkpoint.update(updates=np.array(count,dtype=np.int64),mean=mean,scale=scale)
                atomic_binary_write(dest/'selector_resume_checkpoint.npz',lambda stream:np.savez(stream,**checkpoint))
                write(dest/'selector_resume_checkpoint.json',dict(status='RESUME_ONLY_NOT_FINAL_MODEL',updates=count,
                    registered_updates=576,stage_initial_update=560,training_stage='seven_source_synchronous_fixed_six_trajectory_pools',
                    next_position=None if done==16 else dict(traversal_zero_based=done//4,batch_zero_based=done%4),
                    fixed_pool_manifest_sha256=digest(dest/'fixed_pool_manifest.json'),
                    initialization_rng_state=model.resume_rng.bit_generator.state,random_draws_after_initialization=0,
                    checkpoint_sha256=digest(dest/'selector_resume_checkpoint.npz'),bindings=actual,trace=trace,
                    training_seconds=restored['training_seconds']+time.perf_counter()-started,current_process_seconds=time.perf_counter()-started))
                write(dest/'selector_progress.json',dict(status='PARTIAL_STAGE_UPDATES_NOT_FINAL_MODEL',updates=count,
                    new_updates=done,registered_new_updates=16,last=trace[-1],seconds=time.perf_counter()-started))
                print('SYNCHRONOUS_EPOCH',traversal+1,'/4','BATCH',batch+1,'/4','SEVEN_SOURCES',
                    'NEW_UPDATE',done,'/16','ADAM_STEP',count,'MEAN_LOSS',trace[-1]['mean_batch_loss'],flush=True)
                del local_gradients;gc.collect()
        assert optimizer.step_count==576 and len(trace)==576
        model.save(dest/'selector.npz');loaded=NearestParentScorer.load(dest/'selector.npz')
        assert all(np.array_equal(v,loaded.parameters[k]) for k,v in model.parameters.items())
        write(dest/'selector_training.json',dict(status='FINAL_SET_SELECTOR_TRAINED',updates=576,new_updates=16,
            completed_before_stage=560,stage_epochs=4,total_epochs=60,batches_per_group=4,source_batches_used=112,
            diagnostic_Adam_calls=8,trace=trace,bindings=actual,loss_kind='best_legal_vs_best_illegal_softplus',
            selector_sha256=digest(dest/'selector.npz'),normalization_sha256=digest(dest/'selector_normalization.npz'),
            seconds=time.perf_counter()-started,DEV_reads=0,only_final_weights_evaluated=True,optimizer_moments_reset=False,
            fixed_pool_manifest_sha256=digest(dest/'fixed_pool_manifest.json'),trajectory_generation_calls=0))
        final=[]
        for key in KEYS:
            value,_=evaluate_one(model,key);final.append(value)
        write(dest/'fixed_pool_before_after.json',dict(status='SAME_FIRST_TRAVERSAL_FULL_SEVEN_POOLS',before=initial,after=final,
            mean_loss_before=float(np.mean([v['loss'] for v in initial])),mean_loss_after=float(np.mean([v['loss'] for v in final])),
            scope='TRAIN_FIXED_STATE_DIAGNOSTIC_NOT_EXTRACTION_METRIC',additional_training_updates=0))
        write(dest/'source_column_integrity_after.json',verify_columns_unchanged(OUT/'run369/training_relation_cache',KEYS))
    finally:write(dest/'training_relation_cache_receipt.json',cache.receipt())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--stage',required=True)
    ap.add_argument('--source',default='');ap.add_argument('--limit-seconds',type=float,required=True)
    a=ap.parse_args();dest=OUT/a.run;start=time.perf_counter()
    try:
        assert a.stage=='synchronous_train';train(dest,a.limit_seconds)
        write(dest/f'stage_{a.stage}_{a.source}_complete.json',dict(status='STAGE_COMPLETE',seconds=time.perf_counter()-start,worker_source_sha256=digest(__file__)))
    except BaseException:
        write(dest/f'stage_{a.stage}_{a.source}_failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-start));raise

if __name__=='__main__':main()
