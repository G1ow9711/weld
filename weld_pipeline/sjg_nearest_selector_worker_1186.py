"""Pro364 nearest whole-parent feature continuation and final pure-input selection."""
import argparse,gc,time,traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_recall_set_worker_1162 import load_parents
from weld_pipeline.sjg_nearest_parent_scorer_1185 import NearestParentScorer as ParentSetScorer
from weld_pipeline.sjg_nearest_parent_features_1184 import NearestParentRelations
from weld_pipeline.sjg_nearest_migration_control_1187 import initial,control
from weld_pipeline.sjg_parent_set_oracle_1158 import ParentSetOracle
from weld_pipeline.sjg_parent_set_features_1160 import ParentRelations
from weld_pipeline.sjg_pairwise_oracle_loss_1181 import pairwise_oracle_loss,check_frozen_state
from weld_pipeline.sjg_selector_resume_state_1178 import restore,reuse_columns
from weld_pipeline.sjg_relation_column_cache_1173 import RelationColumnCache,CachedParentRelations
from weld_pipeline.sjg_selector_trial_worker_1174 import bindings
from weld_pipeline.sjg_seven_relative_prepare_1146 import KEYS


def train(dest,limit):
    source=OUT/'run364';begin=time.perf_counter()
    assert read(dest/'nearest_migration_control.json')['status']=='NEAREST_AND_COMPLETE_FROZEN_MIGRATION_PASSED'
    old,old_adam,model,optimizer,mean,scale,restored=initial(source)
    old_training=read(source/'selector_training.json')
    binding={}
    for key in KEYS:
        folder=dest/f'source_{key}';r=read(folder/'receipt.json')
        assert digest(folder/'unary.npy')==r['unary_sha256']
        assert digest(folder/'decoded.json')==r['decoded_sha256']
        binding[key]=dict(candidate_sha256=r['decoded_sha256'],compatibility_sha256=digest(folder/'compatibility.json'))
    assert restored['bindings']==binding and optimizer.step_count==224
    del old,old_adam;trace=list(restored['trace']);np.savez(dest/'selector_normalization.npz',mean=mean,scale=scale)
    write(dest/'nearest_initialization.json',dict(source='run364',parameters_from_final_model=True,
        initial_adam_step=224,moments_retained=True,source_checkpoint_sha256=restored['checkpoint_sha256'],
        source_model_sha256=old_training['selector_sha256'],normalization_refit=False,
        optimizer_definition_sha256=digest(ROOT/'weld_pipeline/sjg_nearest_parent_scorer_1185.py'),function_preserving_migration=True,new_parameter_rows_zero=True))
    cache=RelationColumnCache(dest/'training_relation_cache',bindings(dest,KEYS))
    reused=reuse_columns(dest/'control_relation_cache',cache)
    assert reused['disk_bytes']==read(dest/'control_relation_cache_receipt.json')['disk_bytes']
    write(dest/'relation_cache_reuse.json',reused);print('NEAREST_INITIAL_STATE_AND_CACHE',reused,'ADAM_STEP',optimizer.step_count,flush=True)
    try:
        for stage_epoch in range(16):
            for key in KEYS:
                if time.perf_counter()-begin>=limit:raise TimeoutError('Shared pairwise trial execution deadline')
                check_resources();folder=dest/f'source_{key}';u=(np.load(folder/'unary.npy')-mean)/scale
                c=read(folder/'compatibility.json');parents,points=load_parents(folder)
                context=NearestParentRelations(parents,read(folder/'parent_metadata.json'),key,cache)
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
                trace.append(dict(epoch=33+stage_epoch,stage_epoch=1+stage_epoch,source=key,prefixes=steps,
                    mean_loss=loss_sum/steps,loss_kind='best_legal_vs_best_illegal_softplus',**update))
                count=optimizer.step_count;assert 225<=count<=336
                checkpoint={f'parameter_{k}':v for k,v in model.parameters.items()}
                checkpoint.update({f'first_{k}':v for k,v in optimizer.first.items()})
                checkpoint.update({f'second_{k}':v for k,v in optimizer.second.items()})
                checkpoint.update(updates=np.array(count,dtype=np.int64),mean=mean,scale=scale)
                atomic_binary_write(dest/'selector_resume_checkpoint.npz',lambda stream:np.savez(stream,**checkpoint))
                write(dest/'selector_resume_checkpoint.json',dict(status='RESUME_ONLY_NOT_FINAL_MODEL',updates=count,
                    registered_updates=336,stage_initial_update=224,next_epoch=count//len(KEYS)+1 if count<336 else None,
                    next_source=KEYS[count%len(KEYS)] if count<336 else None,
                    initialization_rng_state=model.resume_rng.bit_generator.state,random_draws_after_initialization=0,
                    checkpoint_sha256=digest(dest/'selector_resume_checkpoint.npz'),bindings=binding,trace=trace,
                    training_seconds=restored['training_seconds']+time.perf_counter()-begin,current_process_seconds=time.perf_counter()-begin))
                write(dest/'selector_progress.json',dict(status='PARTIAL_STAGE_UPDATES_NOT_FINAL_MODEL',updates=count,
                    registered_updates=336,new_updates=count-224,registered_new_updates=112,last=trace[-1],seconds=time.perf_counter()-begin))
                print('NEAREST_EPOCH',stage_epoch+1,'/16','PIECE',key,'NEW_UPDATE',count-224,'/112','ADAM_STEP',count,'LOSS',loss_sum/steps,flush=True)
                del context,parents;points._mmap.close();gc.collect()
        assert optimizer.step_count==336
        model.save(dest/'selector.npz');loaded=ParentSetScorer.load(dest/'selector.npz')
        assert all(np.array_equal(v,loaded.parameters[k]) for k,v in model.parameters.items())
        write(dest/'selector_training.json',dict(status='FINAL_SET_SELECTOR_TRAINED',updates=336,new_updates=112,
            completed_before_stage=224,stage_epochs=16,total_epochs=48,trace=trace,bindings=binding,
            loss_kind='best_legal_vs_best_illegal_softplus',selector_sha256=digest(dest/'selector.npz'),
            normalization_sha256=digest(dest/'selector_normalization.npz'),seconds=time.perf_counter()-begin,
            DEV_reads=0,only_final_weights_evaluated=True,optimizer_moments_reset=False))
    finally:write(dest/'training_relation_cache_receipt.json',cache.receipt())


def select_final(dest,key,folder):
    from weld_pipeline.sjg_nearest_parent_scorer_1185 import NearestParentScorer as ParentSetScorer
    from weld_pipeline.sjg_nearest_parent_features_1184 import NearestParentRelations as ParentRelations
    begin=time.perf_counter();training=read(dest/'selector_training.json')
    assert digest(dest/'selector.npz')==training['selector_sha256']
    assert digest(dest/'selector_normalization.npz')==training['normalization_sha256']
    model=ParentSetScorer.load(dest/'selector.npz')
    with np.load(dest/'selector_normalization.npz') as z:u=(np.load(folder/'unary.npy')-z['mean'])/z['scale']
    parents,points=load_parents(folder);context=ParentRelations(parents,read(folder/'parent_metadata.json'))
    for step in range(len(parents)+1):
        if time.perf_counter()-begin>120:raise TimeoutError('Complete parent selection 120 second development limit')
        check_resources();scores,_=model.forward(u,context.features(),context.selected)
        chosen=model.choose_next(scores,context.selected)
        if chosen is None:break
        context.add(chosen)
    else:raise RuntimeError('No STOP after all candidates')
    indices=list(context.selected);del parents,context;points._mmap.close()
    original=read(folder/'decoded.json');result=dict(original)
    result['parents']=[original['parents'][i] for i in indices]
    result['total_prediction_children']=sum(p['child_count'] for p in result['parents'])+sum(v['counted_false_children'] for v in original['invalid_predictions'])
    result.update(runtime_model='PRO364_RECALL_AND_NEAREST_PARENT_SET',GT_reads=0)
    write(folder/'final_decoded.json',result)
    write(folder/'final_prediction.json',dict(status='LABEL_FREE_SET_PREDICTION_FROZEN',source=key,
        selected_candidate_indices=indices,candidate_count=len(original['parents']),
        decoded_sha256=digest(folder/'final_decoded.json'),selector_sha256=training['selector_sha256'],
        invalid_ledger_unchanged=True,GT_reads=0,seconds=time.perf_counter()-begin))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--stage',required=True)
    ap.add_argument('--source',default='');ap.add_argument('--limit-seconds',type=float,required=True)
    args=ap.parse_args();dest=OUT/args.run;begin=time.perf_counter()
    try:
        check_resources()
        if args.stage=='nearest_control':control(dest)
        elif args.stage=='nearest_train':train(dest,args.limit_seconds)
        elif args.stage=='select':select_final(dest,args.source,dest/f'test_{args.source}')
        else:raise ValueError(args.stage)
        write(dest/f'stage_{args.stage}_{args.source}_complete.json',dict(status='STAGE_COMPLETE',seconds=time.perf_counter()-begin,
            worker_source_sha256=digest(__file__)))
    except BaseException:
        write(dest/f'stage_{args.stage}_{args.source}_failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-begin));raise


if __name__=='__main__':main()
