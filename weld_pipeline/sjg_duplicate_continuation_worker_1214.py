"""Pro374: resume full run374 head112 for exactly 56 new updates."""
import argparse,gc,hashlib,shutil,time,traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources
from weld_pipeline.sjg_pair_residual_1203 import ResidualScorer,ResidualRelations
from weld_pipeline.sjg_duplicate_head_1209 import DuplicateScorer,DuplicateAdam,DuplicateRelations,reverse_column,SHAPES
from weld_pipeline.sjg_duplicate_supervision_1210 import initialize,PairSupervision,head_control
from weld_pipeline.sjg_nearest_parent_scorer_1185 import NearestParentScorer
from weld_pipeline.sjg_nearest_parent_features_1184 import NearestParentRelations
from weld_pipeline.sjg_prefix_minibatch_1189 import restore_final,parameter_digest,partition,mask_digest,batch_weights
from weld_pipeline.sjg_parent_set_oracle_1158 import ParentSetOracle
from weld_pipeline.sjg_stop_coverage_loss_1206 import combined_oracle_loss
from weld_pipeline.sjg_mixed_recovery_1192 import recovery_action
from weld_pipeline.sjg_history_replay_worker_1198 import inputs
from weld_pipeline.sjg_recall_set_worker_1162 import load_parents
from weld_pipeline.sjg_synchronous_pool_1200 import ReadOnlyColumns,verify_columns_unchanged
from weld_pipeline.sjg_selector_resume_state_1178 import reuse_columns
from weld_pipeline.sjg_selector_trial_worker_1174 import bindings
from weld_pipeline.sjg_seven_relative_prepare_1146 import KEYS
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write


from weld_pipeline.sjg_duplicate_worker_1211 import freeze,mixture,select
from weld_pipeline.sjg_duplicate_continuation_1213 import restore_head,restore_and_clip_control,seed_prior_pair_checks,fixed_actual_diagnostic

def train(dest,limit):
    start=time.perf_counter();old=OUT/'run367'
    assert read(dest/'resume_clip_control.json')['status']=='COMPLETE_HEAD112_RESTORED_AND_CLIP_REPAIR_EXACT'
    model,adam,mean,scale,restored=restore_head(dest)
    shutil.copy2(OUT/'run373/selector.npz',dest/'frozen_residual.npz');shutil.copy2(old/'selector.npz',dest/'base_selector.npz');shutil.copy2(old/'selector_normalization.npz',dest/'selector_normalization.npz')
    cache=ReadOnlyColumns(dest/'training_relation_cache',bindings(dest,KEYS));reuse_columns(OUT/'run374/training_relation_cache',cache)
    trajectories=dest/'trajectories';trajectories.mkdir();trace=list(restored['head_resume']['trace']);supervision=PairSupervision(dest,KEYS)
    def deadline():
        check_resources()
        if time.perf_counter()-start>=limit:raise TimeoutError('Shared residual trial budget exhausted')
    try:
        seed_prior_pair_checks(dest,supervision,cache,KEYS)
        fixed_actual_diagnostic(dest,model,mean,scale,cache,'before')
        deadline()
        for local_e in range(2):
            e=4+local_e
            for key in KEYS:
                deadline();u,parents,metadata,compat,points=inputs(dest,key,mean,scale)
                try:
                    pair={kind:freeze(model,u,parents,metadata,compat,key,cache,e,kind=='recovery',supervision) for kind in ('clean','recovery')}
                    assert pair['clean']['model_state_sha256']==pair['recovery']['model_state_sha256']
                    pair.update(candidate_binding=restored['bindings'][key],base_sha256=model.base_sha)
                    path=trajectories/f'pass_{e+1:02}_{key}.json';write(path,pair);sha=digest(path)
                    for batch in range(4):
                        deadline();loss,g=mixture(model,u,parents,metadata,compat,key,cache,pair,batch,supervision)
                        update=adam.update(model,g);count=adam.step_count
                        trace.append(dict(stage_epoch=e+1,continuation_epoch=local_e+1,source=key,batch=batch+1,weighted_batch_loss=loss,
                            trajectory_sha256=sha,clean_prefixes=pair['clean']['prefix_count'],recovery_prefixes=pair['recovery']['prefix_count'],**update))
                        data={f'parameter_{k}':v for k,v in model.parameters.items()}
                        data.update({f'first_{k}':v for k,v in adam.first.items()});data.update({f'second_{k}':v for k,v in adam.second.items()})
                        data.update(updates=np.array(count,dtype=np.int64),mean=mean,scale=scale)
                        atomic_binary_write(dest/'selector_resume_checkpoint.npz',lambda stream:np.savez(stream,**data))
                        visit=count//4
                        write(dest/'selector_resume_checkpoint.json',dict(status='DUPLICATE_HEAD_ONLY_RESUME_CHECKPOINT',updates=count,
                            registered_updates=168,base_Adam_step=560,base_parameters_and_moments_frozen=True,
                            base_checkpoint_sha256=restored['checkpoint_sha256'],base_selector_sha256=model.base_sha,
                            frozen_residual_sha256=model.frozen_sha,frozen_residual_Adam_step=112,
                            frozen_residual_checkpoint_sha256=restored['frozen_residual_checkpoint_sha256'],
                            checkpoint_sha256=digest(dest/'selector_resume_checkpoint.npz'),bindings=restored['bindings'],trace=trace,
                            initialization_rng_state=model.resume_rng.bit_generator.state,random_draws_after_initialization=0,
                            next_position=None if count==168 else dict(traversal_zero_based=visit//7,source=KEYS[visit%7],batch_zero_based=count%4,
                                trajectory_path=str(path) if count%4 else None,trajectory_sha256=sha if count%4 else None)))
                        write(dest/'selector_progress.json',dict(status='PARTIAL_DUPLICATE_HEAD_NOT_FINAL',updates=count,registered_updates=168,
                            base_Adam_step=560,last=trace[-1],seconds=time.perf_counter()-start))
                        print('DUPLICATE_CONTINUE_EPOCH',local_e+1,'/2','PIECE',key,'BATCH',batch+1,'/4','NEW_UPDATE',count-112,'/56','HEAD_STEP',count,'/168',
                            'BASE_FROZEN_STEP',560,'LOSS',loss,flush=True)
                finally:
                    del parents
                    points._mmap.close()
                gc.collect()
        assert adam.step_count==168 and len(trace)==168
        assert digest(old/'selector.npz')==model.base_sha==digest(dest/'base_selector.npz')
        assert digest(old/'selector_resume_checkpoint.npz')==restored['checkpoint_sha256']
        assert digest(OUT/'run373/selector.npz')==model.frozen_sha==digest(dest/'frozen_residual.npz')
        assert digest(OUT/'run373/selector_resume_checkpoint.npz')==restored['frozen_residual_checkpoint_sha256']
        assert digest(OUT/'run374/selector.npz')==restored['head_source_sha256']
        assert digest(OUT/'run374/selector_resume_checkpoint.npz')==restored['head_source_checkpoint_sha256']
        model.save(dest/'selector.npz');loaded=DuplicateScorer.load(dest/'selector.npz',dest/'frozen_residual.npz',dest/'base_selector.npz')
        assert all(np.array_equal(v,loaded.parameters[k]) for k,v in model.parameters.items())
        fixed_actual_diagnostic(dest,loaded,mean,scale,cache,'after')
        deadline()
        write(dest/'selector_training.json',dict(status='FINAL_DUPLICATE_HEAD_CONTINUED',updates=168,new_updates=56,completed_before_stage=112,
            resumed_run='run374',resumed_selector_sha256=restored['head_source_sha256'],resumed_checkpoint_sha256=restored['head_source_checkpoint_sha256'],
            base_Adam_step=560,base_selector_sha256=model.base_sha,base_parameters_and_moments_frozen=True,
            frozen_residual_Adam_step=112,frozen_residual_sha256=model.frozen_sha,frozen_residual_parameters_and_moments_frozen=True,
            frozen_residual_checkpoint_sha256=restored['frozen_residual_checkpoint_sha256'],
            base_checkpoint_sha256=restored['checkpoint_sha256'],trace=trace,bindings=restored['bindings'],
            selector_sha256=digest(dest/'selector.npz'),normalization_sha256=digest(dest/'selector_normalization.npz'),
            seconds=time.perf_counter()-start,trainable_parameters=145,DEV_reads=0,loss_kind='pairwise_plus_coverage_plus_symmetric_pair_duplicate',coverage_coefficient=1.,duplicate_coefficient=1.,only_final_weights_evaluated=True))
        write(dest/'source_column_integrity_after.json',verify_columns_unchanged(OUT/'run374/training_relation_cache',KEYS))
    finally:
        write(dest/'training_relation_cache_receipt.json',cache.receipt())
        write(dest/'pair_supervision_receipt.json',supervision.receipt())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--stage',required=True)
    ap.add_argument('--source',default='');ap.add_argument('--limit-seconds',type=float,required=True)
    a=ap.parse_args();dest=OUT/a.run;start=time.perf_counter()
    try:
        if a.stage=='duplicate_control':restore_and_clip_control(dest)
        elif a.stage=='duplicate_train':train(dest,a.limit_seconds)
        elif a.stage=='select':select(dest,a.source,a.limit_seconds)
        else:raise ValueError(a.stage)
        write(dest/f'stage_{a.stage}_{a.source}_complete.json',dict(status='STAGE_COMPLETE',seconds=time.perf_counter()-start,worker_source_sha256=digest(__file__)))
    except BaseException:
        write(dest/f'stage_{a.stage}_{a.source}_failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-start));raise

if __name__=='__main__':main()
