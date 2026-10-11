"""Pro375: independent full-candidate supervision, frozen run375 selection."""
import argparse,copy,hashlib,shutil,time,traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources
from weld_pipeline.sjg_incompatibility_head_1216 import IncompatibilityScorer,IncompatibilityAdam,SHAPES,labels_from_edges,candidate_loss
from weld_pipeline.sjg_duplicate_head_1209 import DuplicateScorer,DuplicateRelations,SHAPES as DUP_SHAPES
from weld_pipeline.sjg_duplicate_supervision_1210 import initialize as restore_367_373
from weld_pipeline.sjg_prefix_minibatch_1189 import parameter_digest
from weld_pipeline.sjg_history_replay_worker_1198 import inputs
from weld_pipeline.sjg_recall_set_worker_1162 import load_parents
from weld_pipeline.sjg_synchronous_pool_1200 import ReadOnlyColumns,verify_columns_unchanged
from weld_pipeline.sjg_selector_resume_state_1178 import reuse_columns
from weld_pipeline.sjg_selector_trial_worker_1174 import bindings
from weld_pipeline.sjg_seven_relative_prepare_1146 import KEYS
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write


from weld_pipeline.sjg_incompatibility_worker_1217 import all_data,select
from weld_pipeline.sjg_incompatibility_continuation_1219 import restore
from weld_pipeline.sjg_stop_risk_1222 import controls_and_envelopes,load_envelopes,risk_loss

def train(dest,limit):
    start=time.perf_counter();model,adam,mean,scale,restored=restore(dest)
    assert read(dest/'stop_risk_control.json')['status']=='NEW_STOP_RISK_LOSS_CONTROL_PASSED'
    envelopes=load_envelopes(dest)
    shutil.copy2(OUT/'run376/candidate_supervision_control.json',dest/'candidate_supervision_control.json')
    assert read(dest/'candidate_supervision_control.json')['status']=='ALL_SEVEN_U160_FINITE_NO_OPPOSITE_LABEL_CONFLICT'
    for source,name in ((OUT/'run367/selector.npz','base_selector.npz'),(OUT/'run373/selector.npz','frozen_residual.npz'),
                        (OUT/'run375/selector.npz','frozen_duplicate.npz'),(OUT/'run367/selector_normalization.npz','selector_normalization.npz')):
        shutil.copy2(source,dest/name)
    data=all_data(dest,mean,scale);trace=list(restored['head_resume']['trace'])
    for local_epoch in range(8):
        epoch=16+local_epoch
        for key in KEYS:
            check_resources()
            if time.perf_counter()-start>=limit:raise TimeoutError('Registered training balance exhausted')
            u,y=data[key];z,forward=model.logits(u);loss,delta,loss_parts=risk_loss(z,y,envelopes[key]);g=model.backward(forward,delta)
            update=adam.update(model,g);count=adam.step_count
            trace.append(dict(epoch=epoch+1,continuation_epoch=local_epoch+1,source=key,loss=loss,loss_parts=loss_parts,candidate_count=len(y),known_count=int(np.sum(y>=0)),**update))
            arrays={f'parameter_{k}':v for k,v in model.parameters.items()}
            arrays.update({f'first_{k}':v for k,v in adam.first.items()});arrays.update({f'second_{k}':v for k,v in adam.second.items()})
            arrays.update(updates=np.array(count,dtype=np.int64),mean=mean,scale=scale)
            atomic_binary_write(dest/'selector_resume_checkpoint.npz',lambda stream:np.savez(stream,**arrays))
            write(dest/'selector_resume_checkpoint.json',dict(status='INCOMPATIBILITY_HEAD_ONLY_CHECKPOINT',updates=count,registered_updates=168,
                checkpoint_sha256=digest(dest/'selector_resume_checkpoint.npz'),bindings=restored['bindings'],trace=trace,
                envelope_manifest_sha256=digest(dest/'envelope_manifest.json'),
                base_checkpoint_sha256=restored['checkpoint_sha256'],frozen_residual_checkpoint_sha256=restored['frozen_residual_checkpoint_sha256'],
                frozen_duplicate_checkpoint_sha256=restored['frozen_duplicate_checkpoint_sha256'],
                initialization_rng_state=model.resume_rng.bit_generator.state,random_draws_after_initialization=0,
                next_position=None if count==168 else dict(epoch_zero_based=count//7,source=KEYS[count%7])))
            write(dest/'selector_progress.json',dict(status='PARTIAL_INCOMPATIBILITY_HEAD_NOT_FINAL',updates=count,registered_updates=168,last=trace[-1]))
            print('STOP_RISK_EPOCH',local_epoch+1,'/8','PIECE',key,'NEW_HEAD_UPDATE',count-112,'/56','HEAD_STEP',count,'/168','LOSS',loss,flush=True)
    assert adam.step_count==168 and len(trace)==168
    for source,field in (('run367','checkpoint_sha256'),('run373','frozen_residual_checkpoint_sha256'),('run375','frozen_duplicate_checkpoint_sha256')):
        assert digest(OUT/source/'selector_resume_checkpoint.npz')==restored[field]
    assert digest(dest/'base_selector.npz')==model.base_sha==digest(OUT/'run367/selector.npz')
    assert digest(dest/'frozen_residual.npz')==model.frozen.frozen_sha==digest(OUT/'run373/selector.npz')
    assert digest(dest/'frozen_duplicate.npz')==model.frozen_sha==digest(OUT/'run375/selector.npz')
    assert digest(OUT/'run376/selector.npz')==restored['head_source_sha256']
    assert digest(OUT/'run376/selector_resume_checkpoint.npz')==restored['head_source_checkpoint_sha256']
    model.save(dest/'selector.npz');loaded=IncompatibilityScorer.load(dest/'selector.npz',dest/'frozen_duplicate.npz',dest/'frozen_residual.npz',dest/'base_selector.npz')
    assert all(np.array_equal(v,loaded.parameters[k]) for k,v in model.parameters.items())
    classification=[]
    for key,(u,y) in data.items():
        z,_=loaded.logits(u);known=y>=0;final_loss,unused_gradient,parts=risk_loss(z,y,envelopes[key])
        classification.append(dict(source=key,known=int(known.sum()),compatible_false_bad=int(np.sum((y==0)&(z>0))),
            incompatible_nonpositive=int(np.sum((y==1)&(z<=0))),loss=candidate_loss(z,y)[0],total_risk_objective=final_loss,risk_loss=parts['risk_loss'],not_extraction_metric=True))
    write(dest/'candidate_classification_final.json',dict(status='FINAL_HEAD_SIGN_DIAGNOSTIC_ONLY',cases=classification))
    write(dest/'selector_training.json',dict(status='FINAL_STOP_RISK_HEAD_TRAINED_FROM376',updates=168,new_updates=56,completed_before_stage=112,trainable_parameters=2593,
        resumed_run='run376',resumed_selector_sha256=restored['head_source_sha256'],resumed_checkpoint_sha256=restored['head_source_checkpoint_sha256'],
        base_Adam_step=560,frozen_residual_Adam_step=112,frozen_duplicate_Adam_step=168,all_previous_parameters_and_moments_frozen=True,
        base_selector_sha256=model.base_sha,frozen_residual_sha256=model.frozen.frozen_sha,frozen_duplicate_sha256=model.frozen_sha,
        base_checkpoint_sha256=restored['checkpoint_sha256'],frozen_residual_checkpoint_sha256=restored['frozen_residual_checkpoint_sha256'],
        frozen_duplicate_checkpoint_sha256=restored['frozen_duplicate_checkpoint_sha256'],trace=trace,bindings=restored['bindings'],
        selector_sha256=digest(dest/'selector.npz'),normalization_sha256=digest(dest/'selector_normalization.npz'),
        seconds=time.perf_counter()-start,envelope_manifest_sha256=digest(dest/'envelope_manifest.json'),risk_coefficient=1.,loss_kind='full_candidate_BCE_plus_observed_STOP_group_max_risk',DEV_reads=0,trajectory_generation_calls=0,
        only_final_weights_evaluated=True,geometric_labels_only=True,direction_and_E16_still_unconfirmed=True))
    write(dest/'source_column_integrity_after.json',verify_columns_unchanged(OUT/'run377/diagnostic_relation_cache',KEYS))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--stage',required=True)
    ap.add_argument('--source',default='');ap.add_argument('--limit-seconds',type=float,required=True)
    a=ap.parse_args();dest=OUT/a.run;start=time.perf_counter()
    try:
        if a.stage=='incompatibility_control':controls_and_envelopes(dest)
        elif a.stage=='incompatibility_train':train(dest,a.limit_seconds)
        elif a.stage=='select':select(dest,a.source,a.limit_seconds)
        else:raise ValueError(a.stage)
        write(dest/f'stage_{a.stage}_{a.source}_complete.json',dict(status='STAGE_COMPLETE',seconds=time.perf_counter()-start,worker_source_sha256=digest(__file__)))
    except BaseException:
        write(dest/f'stage_{a.stage}_{a.source}_failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-start));raise

if __name__=='__main__':main()
