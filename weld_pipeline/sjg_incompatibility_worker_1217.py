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


def initialize(dest):
    unused,unused_adam,mean,scale,receipt=restore_367_373(dest)
    old=OUT/'run375';t=read(old/'selector_training.json');r=read(old/'selector_resume_checkpoint.json')
    assert t['updates']==r['updates']==r['registered_updates']==168 and r['next_position'] is None
    assert t['trace']==r['trace'] and [v['step'] for v in r['trace']]==list(range(1,169))
    assert t['bindings']==r['bindings']==receipt['bindings']
    assert digest(old/'selector.npz')==t['selector_sha256'] and digest(old/'selector_resume_checkpoint.npz')==r['checkpoint_sha256']
    assert digest(old/'selector_normalization.npz')==t['normalization_sha256']==digest(OUT/'run367/selector_normalization.npz')
    assert r['base_checkpoint_sha256']==receipt['checkpoint_sha256'] and r['frozen_residual_checkpoint_sha256']==receipt['frozen_residual_checkpoint_sha256']
    frozen=DuplicateScorer.load(old/'selector.npz',old/'frozen_residual.npz',old/'base_selector.npz')
    with np.load(old/'selector_resume_checkpoint.npz') as z:
        assert set(z.files)=={f'{prefix}_{key}' for prefix in ('parameter','first','second') for key in DUP_SHAPES}|{'updates','mean','scale'}
        assert int(z['updates'])==168 and np.array_equal(z['mean'],mean) and np.array_equal(z['scale'],scale)
        for key,shape in DUP_SHAPES.items():
            assert np.array_equal(z[f'parameter_{key}'],frozen.parameters[key])
            for prefix in ('first','second'):
                v=z[f'{prefix}_{key}'];assert v.shape==shape and v.dtype==np.float64 and np.isfinite(v).all()
                if prefix=='second':assert np.all(v>=0)
    assert r['initialization_rng_state']==frozen.initialization_rng_state and r['random_draws_after_initialization']==0
    receipt.update(frozen_duplicate_sha256=t['selector_sha256'],frozen_duplicate_checkpoint_sha256=r['checkpoint_sha256'])
    model=IncompatibilityScorer(frozen,t['selector_sha256'])
    return model,IncompatibilityAdam(model),mean,scale,receipt


def all_data(dest,mean,scale):
    result={}
    for key in KEYS:
        folder=dest/f'source_{key}';r=read(folder/'receipt.json');c=read(folder/'compatibility.json')
        assert digest(folder/'unary.npy')==r['unary_sha256'] and digest(folder/'decoded.json')==r['decoded_sha256']==c['prediction_sha256']
        assert c['matching_policy']=='Whole-parent D3 with unchanged 3mm gate; no independent child flips'
        u=(np.load(folder/'unary.npy')-mean)/scale;assert u.shape==(len(c['compatible_targets']),160) and np.isfinite(u).all()
        # This frozen D3 schema labels geometric compatibility for every candidate.
        # Its unconfirmed E16/directed-side fields are not promoted to known truth.
        known=c.get('known_candidate_mask')
        y=labels_from_edges(c['compatible_targets'],known);result[key]=(u,y)
    return result


def controls(dest):
    model,adam,mean,scale,receipt=initialize(dest);initial=parameter_digest(model);data=all_data(dest,mean,scale)
    seen={};counts=[]
    for key,(u,y) in data.items():
        counts.append(dict(source=key,candidates=len(y),compatible=int(np.sum(y==0)),incompatible=int(np.sum(y==1)),unknown=int(np.sum(y<0))))
        for i in np.flatnonzero(y>=0):
            row=u[i].copy();row[row==0]=0.;code=row.tobytes();current=dict(source=key,candidate=int(i),label=int(y[i]))
            prior=seen.get(code)
            if prior is not None and prior['label']!=current['label']:
                write(dest/'unary_label_conflict.json',dict(status='EXACT_U160_OPPOSITE_KNOWN_LABELS_STOP',previous=prior,current=current,input=row.tolist()))
                raise RuntimeError('Exact U160 label conflict: no fitting, no masking, no identity features')
            if prior is None:seen[code]=current
    assert sum(v['candidates'] for v in counts)==3899
    write(dest/'candidate_supervision_control.json',dict(status='ALL_SEVEN_U160_FINITE_NO_OPPOSITE_LABEL_CONFLICT',cases=counts,
        unique_inputs=len(seen),geometric_compatibility_only=True,direction_and_E16_still_unconfirmed=True,model_updates=0))
    cache=ReadOnlyColumns(dest/'control_relation_cache',bindings(dest,KEYS));reuse_columns(OUT/'run375/training_relation_cache',cache)
    reference=DuplicateScorer.load(OUT/'run375/selector.npz',OUT/'run375/frozen_residual.npz',OUT/'run375/base_selector.npz');cases=[]
    try:
        for key in ('21','BM79A_native_local','10'):
            u,parents,metadata,compat,points=inputs(dest,key,mean,scale)
            wrapped=DuplicateRelations(parents,metadata,model.frozen,key,cache);original=DuplicateRelations(parents,metadata,reference,key,cache)
            choices=read(OUT/f'run375/test_{key}/final_prediction.json')['selected_candidate_indices'];z,_=model.logits(u)
            assert np.array_equal(z,np.zeros(len(u)));h=hashlib.sha256()
            try:
                for prefix in range(len(choices)+1):
                    check_resources();a,_=reference.forward(u,original.features(),original.selected)
                    b=model.forward(u,wrapped.features(),wrapped.selected,z);assert np.array_equal(a,b);h.update(b.tobytes())
                    expected=choices[prefix] if prefix<len(choices) else None
                    assert model.choose_next(b,wrapped.selected)==reference.choose_next(a,original.selected)==expected
                    if expected is not None:wrapped.add(expected);original.add(expected)
                cases.append(dict(source=key,prefixes=len(choices)+1,scores_sha256=h.hexdigest(),saved_actions_and_STOP_exact=True))
            finally:
                del wrapped,original,parents
                points._mmap.close()
        write(dest/'zero_incompatibility_equivalence.json',dict(status='RUN375_THREE_ACTUAL_PREFIXES_EXACT',cases=cases,model_updates=0))
    finally:write(dest/'control_relation_cache_receipt.json',cache.receipt())
    # One small loss/interface control: all coordinates, nonzero discarded output
    # layer to exercise hidden derivatives; no optimizer calls or fitted controls.
    u,y=data['10'];ids=np.r_[np.flatnonzero(y==0)[:3],np.flatnonzero(y==1)[:3]];u=u[ids];y=y[ids]
    probe=np.array([-2.,0.,2.,-2.,0.,2.]);loss,d=candidate_loss(probe,y)
    assert np.all(d[y==0]>0) and np.all(d[y==1]<0)
    eps=1e-5;logit_errors=[]
    for i in range(len(probe)):
        a=probe.copy();b=probe.copy();a[i]+=eps;b[i]-=eps
        logit_errors.append(abs((candidate_loss(a,y)[0]-candidate_loss(b,y)[0])/(2*eps)-d[i]))
    assert max(logit_errors)<5e-8
    assert labels_from_edges([[],[0],[0,1]],[False,True,True]).tolist()==[-1,0,0]
    assert candidate_loss(np.array([10.,-1.]),np.array([-1,1]))[1][0]==0
    test=copy.deepcopy(model);test.parameters['W2']=np.random.Generator(np.random.PCG64(1217)).normal(0,.03,16)
    z,forward=test.logits(u);loss,d=candidate_loss(z,y);g=test.backward(forward,d);errors={}
    for key,value in test.parameters.items():
        error=0.
        for index in np.ndindex(value.shape):
            old=value[index];value[index]=old+eps;plus=candidate_loss(test.logits(u)[0],y)[0]
            value[index]=old-eps;minus=candidate_loss(test.logits(u)[0],y)[0];value[index]=old
            error=max(error,abs((plus-minus)/(2*eps)-g[key][index]))
        errors[key]=error;assert error<5e-8,(key,error)
    assert parameter_digest(model)==initial and adam.step_count==0
    write(dest/'incompatibility_loss_control.json',dict(status='RAW_LOGIT_SUPERVISION_AND_GRADIENT_PASSED',
        logit_max_error=max(logit_errors),parameter_max_errors=errors,trainable_parameters=2593,
        inactive_deployment_deduction_still_supervised=True,unknown_auxiliary_only_mask=True,model_updates=0))
    print('INCOMPATIBILITY_CONTROLS_PASSED',counts,cases,errors,flush=True)


def train(dest,limit):
    start=time.perf_counter();model,adam,mean,scale,restored=initialize(dest)
    assert read(dest/'candidate_supervision_control.json')['status']=='ALL_SEVEN_U160_FINITE_NO_OPPOSITE_LABEL_CONFLICT'
    assert read(dest/'zero_incompatibility_equivalence.json')['status']=='RUN375_THREE_ACTUAL_PREFIXES_EXACT'
    assert read(dest/'incompatibility_loss_control.json')['status']=='RAW_LOGIT_SUPERVISION_AND_GRADIENT_PASSED'
    for source,name in ((OUT/'run367/selector.npz','base_selector.npz'),(OUT/'run373/selector.npz','frozen_residual.npz'),
                        (OUT/'run375/selector.npz','frozen_duplicate.npz'),(OUT/'run367/selector_normalization.npz','selector_normalization.npz')):
        shutil.copy2(source,dest/name)
    data=all_data(dest,mean,scale);trace=[]
    for epoch in range(16):
        for key in KEYS:
            check_resources()
            if time.perf_counter()-start>=limit:raise TimeoutError('Registered training balance exhausted')
            u,y=data[key];z,forward=model.logits(u);loss,delta=candidate_loss(z,y);g=model.backward(forward,delta)
            update=adam.update(model,g);count=adam.step_count
            trace.append(dict(epoch=epoch+1,source=key,loss=loss,candidate_count=len(y),known_count=int(np.sum(y>=0)),**update))
            arrays={f'parameter_{k}':v for k,v in model.parameters.items()}
            arrays.update({f'first_{k}':v for k,v in adam.first.items()});arrays.update({f'second_{k}':v for k,v in adam.second.items()})
            arrays.update(updates=np.array(count,dtype=np.int64),mean=mean,scale=scale)
            atomic_binary_write(dest/'selector_resume_checkpoint.npz',lambda stream:np.savez(stream,**arrays))
            write(dest/'selector_resume_checkpoint.json',dict(status='INCOMPATIBILITY_HEAD_ONLY_CHECKPOINT',updates=count,registered_updates=112,
                checkpoint_sha256=digest(dest/'selector_resume_checkpoint.npz'),bindings=restored['bindings'],trace=trace,
                base_checkpoint_sha256=restored['checkpoint_sha256'],frozen_residual_checkpoint_sha256=restored['frozen_residual_checkpoint_sha256'],
                frozen_duplicate_checkpoint_sha256=restored['frozen_duplicate_checkpoint_sha256'],
                initialization_rng_state=model.resume_rng.bit_generator.state,random_draws_after_initialization=0,
                next_position=None if count==112 else dict(epoch_zero_based=count//7,source=KEYS[count%7])))
            write(dest/'selector_progress.json',dict(status='PARTIAL_INCOMPATIBILITY_HEAD_NOT_FINAL',updates=count,registered_updates=112,last=trace[-1]))
            print('INCOMPATIBILITY_EPOCH',epoch+1,'/16','PIECE',key,'NEW_HEAD_UPDATE',count,'/112','LOSS',loss,flush=True)
    assert adam.step_count==112 and len(trace)==112
    for source,field in (('run367','checkpoint_sha256'),('run373','frozen_residual_checkpoint_sha256'),('run375','frozen_duplicate_checkpoint_sha256')):
        assert digest(OUT/source/'selector_resume_checkpoint.npz')==restored[field]
    assert digest(dest/'base_selector.npz')==model.base_sha==digest(OUT/'run367/selector.npz')
    assert digest(dest/'frozen_residual.npz')==model.frozen.frozen_sha==digest(OUT/'run373/selector.npz')
    assert digest(dest/'frozen_duplicate.npz')==model.frozen_sha==digest(OUT/'run375/selector.npz')
    model.save(dest/'selector.npz');loaded=IncompatibilityScorer.load(dest/'selector.npz',dest/'frozen_duplicate.npz',dest/'frozen_residual.npz',dest/'base_selector.npz')
    assert all(np.array_equal(v,loaded.parameters[k]) for k,v in model.parameters.items())
    classification=[]
    for key,(u,y) in data.items():
        z,_=loaded.logits(u);known=y>=0
        classification.append(dict(source=key,known=int(known.sum()),compatible_false_bad=int(np.sum((y==0)&(z>0))),
            incompatible_nonpositive=int(np.sum((y==1)&(z<=0))),loss=candidate_loss(z,y)[0],not_extraction_metric=True))
    write(dest/'candidate_classification_final.json',dict(status='FINAL_HEAD_SIGN_DIAGNOSTIC_ONLY',cases=classification))
    write(dest/'selector_training.json',dict(status='FINAL_INCOMPATIBILITY_HEAD_TRAINED',updates=112,new_updates=112,trainable_parameters=2593,
        base_Adam_step=560,frozen_residual_Adam_step=112,frozen_duplicate_Adam_step=168,all_previous_parameters_and_moments_frozen=True,
        base_selector_sha256=model.base_sha,frozen_residual_sha256=model.frozen.frozen_sha,frozen_duplicate_sha256=model.frozen_sha,
        base_checkpoint_sha256=restored['checkpoint_sha256'],frozen_residual_checkpoint_sha256=restored['frozen_residual_checkpoint_sha256'],
        frozen_duplicate_checkpoint_sha256=restored['frozen_duplicate_checkpoint_sha256'],trace=trace,bindings=restored['bindings'],
        selector_sha256=digest(dest/'selector.npz'),normalization_sha256=digest(dest/'selector_normalization.npz'),
        seconds=time.perf_counter()-start,loss_kind='full_candidate_balanced_raw_logit_BCE',DEV_reads=0,trajectory_generation_calls=0,
        only_final_weights_evaluated=True,geometric_labels_only=True,direction_and_E16_still_unconfirmed=True))
    write(dest/'source_column_integrity_after.json',verify_columns_unchanged(OUT/'run375/training_relation_cache',KEYS))


def select(dest,key,limit):
    start=time.perf_counter();folder=dest/f'test_{key}';training=read(dest/'selector_training.json')
    assert digest(dest/'selector.npz')==training['selector_sha256'] and digest(dest/'selector_normalization.npz')==training['normalization_sha256']
    model=IncompatibilityScorer.load(dest/'selector.npz',dest/'frozen_duplicate.npz',dest/'frozen_residual.npz',dest/'base_selector.npz')
    with np.load(dest/'selector_normalization.npz') as z:u=(np.load(folder/'unary.npy')-z['mean'])/z['scale']
    logits,_=model.logits(u);parents,points=load_parents(folder);context=DuplicateRelations(parents,read(folder/'parent_metadata.json'),model.frozen)
    for step in range(len(parents)+1):
        check_resources()
        if time.perf_counter()-start>min(120.,limit):raise TimeoutError('Selection development time exhausted')
        scores=model.forward(u,context.features(),context.selected,logits);chosen=model.choose_next(scores,context.selected)
        if chosen is None:break
        context.add(chosen)
    else:raise RuntimeError('Missing STOP after all candidates')
    indices=list(context.selected);del context,parents;points._mmap.close()
    original=read(folder/'decoded.json');result=dict(original);result['parents']=[original['parents'][i] for i in indices]
    result['total_prediction_children']=sum(p['child_count'] for p in result['parents'])+sum(v['counted_false_children'] for v in original['invalid_predictions'])
    result.update(runtime_model='FROZEN375_WITH_COMPLETE_PARENT_INCOMPATIBILITY_HEAD',GT_reads=0)
    write(folder/'final_decoded.json',result)
    write(folder/'final_prediction.json',dict(status='LABEL_FREE_INCOMPATIBILITY_PREDICTION_FROZEN',source=key,selected_candidate_indices=indices,
        candidate_count=len(original['parents']),decoded_sha256=digest(folder/'final_decoded.json'),selector_sha256=training['selector_sha256'],
        frozen_duplicate_sha256=model.frozen_sha,incompatibility_logit_evaluations=1,GT_reads=0,seconds=time.perf_counter()-start))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--stage',required=True)
    ap.add_argument('--source',default='');ap.add_argument('--limit-seconds',type=float,required=True)
    a=ap.parse_args();dest=OUT/a.run;start=time.perf_counter()
    try:
        if a.stage=='incompatibility_control':controls(dest)
        elif a.stage=='incompatibility_train':train(dest,a.limit_seconds)
        elif a.stage=='select':select(dest,a.source,a.limit_seconds)
        else:raise ValueError(a.stage)
        write(dest/f'stage_{a.stage}_{a.source}_complete.json',dict(status='STAGE_COMPLETE',seconds=time.perf_counter()-start,worker_source_sha256=digest(__file__)))
    except BaseException:
        write(dest/f'stage_{a.stage}_{a.source}_failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-start));raise

if __name__=='__main__':main()
