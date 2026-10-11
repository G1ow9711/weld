"""Pro363 single-factor pairwise loss control and 112 further Adam updates."""
import argparse,gc,time,traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_recall_set_worker_1162 import load_parents
from weld_pipeline.sjg_parent_set_scorer_1159 import ParentSetScorer,AdamState
from weld_pipeline.sjg_parent_set_oracle_1158 import ParentSetOracle
from weld_pipeline.sjg_parent_set_features_1160 import ParentRelations
from weld_pipeline.sjg_pairwise_oracle_loss_1181 import pairwise_oracle_loss,check_frozen_state
from weld_pipeline.sjg_selector_resume_state_1178 import restore,reuse_columns
from weld_pipeline.sjg_relation_column_cache_1173 import RelationColumnCache,CachedParentRelations
from weld_pipeline.sjg_selector_trial_worker_1174 import bindings
from weld_pipeline.sjg_seven_relative_prepare_1146 import KEYS


def frozen_inputs(source):
    r=read(source/'selector_training.json');assert r['updates']==112
    assert digest(source/'selector.npz')==r['selector_sha256']
    assert digest(source/'selector_normalization.npz')==r['normalization_sha256']
    with np.load(source/'selector_normalization.npz') as z:mean=z['mean'].copy();scale=z['scale'].copy()
    return ParentSetScorer.load(source/'selector.npz'),mean,scale,r


def control(dest):
    source=OUT/'run363';model,mean,scale,training=frozen_inputs(source);records=[]
    for key in ('21','BM79A_native_local','10'):
        folder=source/f'test_{key}';frozen=read(folder/'final_prediction.json')['selected_candidate_indices']
        c=read(source/f'source_{key}/compatibility.json');assert digest(folder/'decoded.json')==c['prediction_sha256']
        parents,points=load_parents(folder);context=ParentRelations(parents,read(folder/'parent_metadata.json'))
        oracle=ParentSetOracle(c['compatible_targets'],c['original_parents']);unary=(np.load(folder/'unary.npy')-mean)/scale
        recorded_repeat=False
        for step in range(len(frozen)+1):
            check_resources();scores,_=model.forward(unary,context.features(),context.selected)
            chosen=model.choose_next(scores,context.selected);assert chosen==(frozen[step] if step<len(frozen) else None)
            available,allowed=oracle.action_masks();kind=None
            if key=='21' and step==0:kind='REAL_21_INITIAL_PREMATURE_STOP'
            elif key=='BM79A_native_local' and chosen is None:kind='REAL_BM_CORRECT_FINAL_STOP'
            elif key=='10' and chosen is None:kind='REAL_10_PREMATURE_STOP'
            elif key=='10' and not recorded_repeat and c['compatible_targets'][chosen] and not allowed[chosen]:
                kind='REAL_10_REPEAT_COMPATIBLE_PREFIX';recorded_repeat=True
            if kind:
                result=check_frozen_state(scores,available,allowed)
                if allowed[-1]:assert result['best_legal']==len(scores)-1
                if chosen is None and not allowed[-1]:assert result['best_illegal']==len(scores)-1
                records.append(dict(case=key,prefix=step,kind=kind,**result))
            if chosen is None:break
            oracle.select(chosen);context.add(chosen)
        del context,parents;points._mmap.close();gc.collect()
    assert len(records)==4
    loss,gradient=pairwise_oracle_loss(np.array([0.]),np.array([True]),np.array([True]))
    assert loss==0. and np.array_equal(gradient,np.zeros(1))
    write(dest/'pairwise_loss_control.json',dict(status='REAL_FROZEN_PREFIX_PAIRWISE_GRADIENTS_PASSED',records=records,
        no_illegal_action_zero_loss=True,actual_model_updates=0,parameter_files_unchanged=True,
        labels_for_post_prediction_diagnosis_only=True,source_model_sha256=training['selector_sha256']))
    print('PAIRWISE_LOSS_CONTROL_PASSED',records,flush=True)


def train(dest,limit):
    source=OUT/'run363';begin=time.perf_counter()
    assert read(dest/'pairwise_loss_control.json')['status']=='REAL_FROZEN_PREFIX_PAIRWISE_GRADIENTS_PASSED'
    frozen,mean,scale,old_training=frozen_inputs(source);model=ParentSetScorer(1031);optimizer=AdamState(model)
    binding={}
    for key in KEYS:
        folder=dest/f'source_{key}';r=read(folder/'receipt.json')
        assert digest(folder/'unary.npy')==r['unary_sha256']
        assert digest(folder/'decoded.json')==r['decoded_sha256']
        binding[key]=dict(candidate_sha256=r['decoded_sha256'],compatibility_sha256=digest(folder/'compatibility.json'))
    restored=restore(source,model,optimizer,mean,scale,binding,allow_completed_stage=True)
    assert optimizer.step_count==112 and all(np.array_equal(model.parameters[k],frozen.parameters[k]) for k in model.parameters)
    del frozen;trace=list(restored['trace']);np.savez(dest/'selector_normalization.npz',mean=mean,scale=scale)
    write(dest/'pairwise_initialization.json',dict(source='run363',parameters_from_final_model=True,
        initial_adam_step=112,moments_retained=True,source_checkpoint_sha256=restored['checkpoint_sha256'],
        source_model_sha256=old_training['selector_sha256'],normalization_refit=False,
        optimizer_definition_sha256=digest(ROOT/'weld_pipeline/sjg_parent_set_scorer_1159.py')))
    cache=RelationColumnCache(dest/'training_relation_cache',bindings(dest,KEYS))
    reused=reuse_columns(source/'training_relation_cache',cache)
    assert reused['disk_bytes']==read(source/'training_relation_cache_receipt.json')['disk_bytes']
    write(dest/'relation_cache_reuse.json',reused);print('PAIRWISE_INITIAL_STATE_AND_CACHE',reused,'ADAM_STEP',optimizer.step_count,flush=True)
    try:
        for stage_epoch in range(16):
            for key in KEYS:
                if time.perf_counter()-begin>=limit:raise TimeoutError('Shared pairwise trial execution deadline')
                check_resources();folder=dest/f'source_{key}';u=(np.load(folder/'unary.npy')-mean)/scale
                c=read(folder/'compatibility.json');parents,points=load_parents(folder)
                context=CachedParentRelations(parents,read(folder/'parent_metadata.json'),key,cache)
                oracle=ParentSetOracle(c['compatible_targets'],c['original_parents'])
                accumulated={k:np.zeros_like(v) for k,v in model.parameters.items()};loss_sum=0.;steps=0
                while True:
                    if time.perf_counter()-begin>=limit:raise TimeoutError('Shared pairwise trial execution deadline')
                    scores,forward=model.forward(u,context.features(),context.selected)
                    available,allowed=oracle.action_masks();loss,delta=pairwise_oracle_loss(scores,available,allowed)
                    gradients=model.backward(forward,delta)
                    for k in accumulated:accumulated[k]+=gradients[k]
                    loss_sum+=loss;steps+=1
                    if allowed[-1]:break
                    legal=np.flatnonzero(allowed[:-1]);chosen=int(legal[np.argmax(scores[legal])])
                    assert oracle.select(chosen);context.add(chosen);check_resources()
                update=optimizer.update(model,{k:v/steps for k,v in accumulated.items()})
                trace.append(dict(epoch=17+stage_epoch,stage_epoch=1+stage_epoch,source=key,prefixes=steps,
                    mean_loss=loss_sum/steps,loss_kind='best_legal_vs_best_illegal_softplus',**update))
                count=optimizer.step_count;assert 113<=count<=224
                checkpoint={f'parameter_{k}':v for k,v in model.parameters.items()}
                checkpoint.update({f'first_{k}':v for k,v in optimizer.first.items()})
                checkpoint.update({f'second_{k}':v for k,v in optimizer.second.items()})
                checkpoint.update(updates=np.array(count,dtype=np.int64),mean=mean,scale=scale)
                atomic_binary_write(dest/'selector_resume_checkpoint.npz',lambda stream:np.savez(stream,**checkpoint))
                write(dest/'selector_resume_checkpoint.json',dict(status='RESUME_ONLY_NOT_FINAL_MODEL',updates=count,
                    registered_updates=224,stage_initial_update=112,next_epoch=count//len(KEYS)+1 if count<224 else None,
                    next_source=KEYS[count%len(KEYS)] if count<224 else None,
                    initialization_rng_state=model.resume_rng.bit_generator.state,random_draws_after_initialization=0,
                    checkpoint_sha256=digest(dest/'selector_resume_checkpoint.npz'),bindings=binding,trace=trace,
                    training_seconds=restored['training_seconds']+time.perf_counter()-begin,current_process_seconds=time.perf_counter()-begin))
                write(dest/'selector_progress.json',dict(status='PARTIAL_STAGE_UPDATES_NOT_FINAL_MODEL',updates=count,
                    registered_updates=224,new_updates=count-112,registered_new_updates=112,last=trace[-1],seconds=time.perf_counter()-begin))
                print('PAIRWISE_EPOCH',stage_epoch+1,'/16','PIECE',key,'NEW_UPDATE',count-112,'/112','ADAM_STEP',count,'LOSS',loss_sum/steps,flush=True)
                del context,parents;points._mmap.close();gc.collect()
        assert optimizer.step_count==224
        model.save(dest/'selector.npz');loaded=ParentSetScorer.load(dest/'selector.npz')
        assert all(np.array_equal(v,loaded.parameters[k]) for k,v in model.parameters.items())
        write(dest/'selector_training.json',dict(status='FINAL_SET_SELECTOR_TRAINED',updates=224,new_updates=112,
            completed_before_stage=112,stage_epochs=16,total_epochs=32,trace=trace,bindings=binding,
            loss_kind='best_legal_vs_best_illegal_softplus',selector_sha256=digest(dest/'selector.npz'),
            normalization_sha256=digest(dest/'selector_normalization.npz'),seconds=time.perf_counter()-begin,
            DEV_reads=0,only_final_weights_evaluated=True,optimizer_moments_reset=False))
    finally:write(dest/'training_relation_cache_receipt.json',cache.receipt())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--stage',required=True)
    ap.add_argument('--source',default='');ap.add_argument('--limit-seconds',type=float,required=True)
    args=ap.parse_args();dest=OUT/args.run;begin=time.perf_counter()
    try:
        check_resources()
        if args.stage=='pairwise_control':control(dest)
        elif args.stage=='pairwise_train':train(dest,args.limit_seconds)
        else:raise ValueError(args.stage)
        write(dest/f'stage_{args.stage}_{args.source}_complete.json',dict(status='STAGE_COMPLETE',seconds=time.perf_counter()-begin,
            worker_source_sha256=digest(__file__)))
    except BaseException:
        write(dest/f'stage_{args.stage}_{args.source}_failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-begin));raise


if __name__=='__main__':main()
