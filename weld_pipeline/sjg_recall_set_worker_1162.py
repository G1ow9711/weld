"""One isolated stage of the registered Pro355 joint recall/set experiment."""
import argparse, copy, gc, os, pickle, time, traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT, read, write, digest, check_resources, utc
from weld_pipeline.sjg_staged_relative_worker_1154 import SETTINGS, save_component, load_component
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_forest_state_hash_1116 import forest_state_sha256
from weld_pipeline.sjg_seven_relative_prepare_1146 import KEYS, feature_path


def fit_object(dest):
    from sklearn.ensemble import ExtraTreesClassifier
    old=OUT/'run349'; layout=read(old/'layout.json'); entry=layout['features']['object_x']
    assert digest(old/entry['file'])==entry['sha256']
    binding=read(OUT/'run345/saved_tables_exact_control.json')
    for name in ('object_y','object_w'):
        assert digest(OUT/'run345'/(name+'.bin'))==binding['files'][name]['sha256']
    x=np.load(old/entry['file'],mmap_mode='r')
    y=np.memmap(OUT/'run345/object_y.bin',dtype='<i2',mode='r',shape=(1134182,))
    original=np.memmap(OUT/'run345/object_w.bin',dtype='<f8',mode='r',shape=(1134182,))
    assert x.shape==(len(y),152) and x.flags.f_contiguous and np.all(y>=0)
    w=np.array(original); positive=y>0; negative=y==0
    assert int(positive.sum())==11959 and int(negative.sum())==1122223
    assert np.isclose(w[positive].sum(),363.,rtol=0,atol=1e-10) and w[negative].sum()==1122223.
    factor=363./1122223.; w[negative]*=factor
    assert np.array_equal(w[positive],original[positive])
    assert np.isclose(w[negative].sum(),363.,rtol=0,atol=1e-10)
    for first in range(0,len(x),4096): assert np.isfinite(x[first:first+4096]).all()
    details=dict(negative_factor=factor,positive_weight=float(w[positive].sum()),negative_weight=float(w[negative].sum()),
        rows=len(y),positive_rows=int(positive.sum()),negative_rows=int(negative.sum()),
        old_input_sha256=entry['sha256'],old_labels_sha256=binding['files']['object_y']['sha256'],
        all_known_rows_retained=True,positive_internal_responsibility_unchanged=True,unknown_masks_unchanged=True,
        calibrated_natural_probability_claim=False,real_fit_calls=1)
    write(dest/'object_fit_started.json',dict(details,utc=utc()))
    print('OBJECT_WEIGHT_REPAIR_FIT 1 OF 1 | 64 TREES | POSITIVE MASS 363 NEGATIVE MASS 363',flush=True)
    model=ExtraTreesClassifier(**SETTINGS).fit(x,y,sample_weight=w)
    save_component(dest,'object',model,details)
    x._mmap.close(); y._mmap.close(); original._mmap.close()


def package(dest):
    obj,r=load_component(dest,'object',verified=False)
    write(dest/'object_verified.json',dict(r,status='COMPONENT_VERIFIED',verification_process_independent=True))
    models=dict(object=obj); states={'object':r['numeric_state_sha256']}; files={'object':r['component_sha256']}
    for name in ('endpoint','connection'):
        models[name],v=load_component(OUT/'run349',name)
        states[name]=v['numeric_state_sha256']; files[name]=v['component_sha256']
    models.update(K=12,feature_columns=152,feature_run='run329',input_request_run='run312',
        feature_runs={k:feature_path(k).parent.name for k in (*KEYS,'64')},training_sources=list(KEYS),
        model_family='PRO355_REWEIGHTED_COMPATIBLE_FORESTS',supervision_run='run318',compatible_run='run317')
    atomic_binary_write(dest/'model.pkl',lambda stream:pickle.dump(models,stream,protocol=5))
    write(dest/'upstream_saved.json',dict(status='UPSTREAM_SAVED_PENDING_VERIFY',model_sha256=digest(dest/'model.pkl'),
        component_state_sha256=states,component_file_sha256=files,new_forest_fit_calls=1,frozen_source_run='run349'))


def load_upstream(dest):
    r=read(dest/'upstream_verified.json')
    assert digest(dest/'model.pkl')==r['model_sha256']
    with (dest/'model.pkl').open('rb') as stream: models=pickle.load(stream)
    assert {k:forest_state_sha256(models[k]) for k in r['component_state_sha256']}==r['component_state_sha256']
    for name in ('object','endpoint','connection'):models[name].verbose=0
    return models,r


def freeze_compact(folder, decoded):
    parents=[]; blocks=[]; offset=0
    for parent in decoded['parents']:
        segments=[]
        for segment in parent['segments']:
            points=np.asarray(segment['points'],np.float64); blocks.append(points)
            item={k:v for k,v in segment.items() if k!='points'}
            item['point_range']=[offset,offset+len(points)]; offset+=len(points); segments.append(item)
        parents.append(dict(parent,segments=segments))
    np.save(folder/'parent_points.npy',np.concatenate(blocks) if blocks else np.empty((0,3)))
    write(folder/'compact_parents.json',dict(parents=parents))


def load_parents(folder):
    points=np.load(folder/'parent_points.npy',mmap_mode='r')
    parents=read(folder/'compact_parents.json')['parents']
    for parent in parents:
        for segment in parent['segments']:
            a,b=segment.pop('point_range'); segment['points']=points[a:b]
    return parents,points


def upstream(dest,key,folder):
    from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case,predict_instances,relative_link_blocks
    from weld_pipeline.sjg_joint_link_runtime_1107 import select_paths
    from weld_pipeline.sjg_request_competition_1125 import competition_resources
    from weld_pipeline.sjg_joint_forest_infer_1111 import decode_paths
    from weld_pipeline.sjg_parent_set_features_1160 import unary_features
    folder.mkdir(exist_ok=False); started=time.perf_counter(); models,binding=load_upstream(dest)
    case=load_case(key);screen=predict_instances(case,models);nodes=screen['nodes']
    print('FREE_CANDIDATES',key,'ADMITTED',screen['admitted_instances'],'VALID_NODES',len(nodes),
          'DIRECTED_PAIR_UPPER_BOUND',len(nodes)*max(0,len(nodes)-1),flush=True)
    write(folder/'screen.json',dict(source=key,admitted_instances=screen['admitted_instances'],valid_nodes=len(nodes),
        invalid_nodes=len(screen['invalid']),maximum_probability=screen['maximum_probability'],GT_reads=0))
    edges=[];masks=[];probabilities=[];edge_count=0
    for links,features,legal in relative_link_blocks(nodes,case):
        p=np.zeros((len(links),4));p[:,models['connection'].classes_.astype(int)]=models['connection'].predict_proba(features)
        edges.append(links);masks.append(legal);probabilities.append(p);edge_count+=len(links);check_resources()
    links=np.concatenate(edges) if edges else np.empty((0,2),np.int32)
    legal=np.concatenate(masks) if masks else np.empty((0,3),bool)
    p=np.concatenate(probabilities) if probabilities else np.empty((0,4))
    del edges,masks,probabilities
    print('FREE_GRAPH',key,'ALL_LEGAL_DIRECTED_EDGES',edge_count,flush=True)
    req=OUT/models['input_request_run']/f'source_{key}_input_requests.json'
    bound=next(v for v in read(OUT/models['input_request_run']/'all_input_requests_frozen_before_labels.json')['cases'] if v['source']==key)
    assert digest(req)==bound['sha256']
    resources=competition_resources(nodes,case['graph'],read(req))
    selected=select_paths(nodes,links,p,legal,wall_seconds=10.,competition_resources=resources,allow_node_selection=True)
    decoded=decode_paths(nodes,selected,case['graph'],screen['invalid'])
    decoded.update(runtime_model='PRO355_FREE_UPSTREAM_CANDIDATES',GT_reads=0)
    write(folder/'decoded.json',decoded)
    write(folder/'selected_graph.json',dict(nodes=nodes,selection=selected,competition_resources=resources))
    u,metadata=unary_features(decoded,nodes,selected,case)
    np.save(folder/'unary.npy',u);write(folder/'parent_metadata.json',metadata);freeze_compact(folder,decoded)
    write(folder/'receipt.json',dict(status='PURE_INPUT_COMPLETE_PARENT_CANDIDATES_FROZEN',source=key,
        upstream_model_sha256=binding['model_sha256'],decoded_sha256=digest(folder/'decoded.json'),
        unary_sha256=digest(folder/'unary.npy'),input_feature_receipt_sha256=case['feature_receipt_sha256'],
        candidate_parents=len(decoded['parents']),candidate_children=decoded['total_prediction_children'],
        conditional_edges=edge_count,graph_selection_seconds=selected['seconds'],
        cached_pipeline_seconds=time.perf_counter()-started,GT_reads=0,STEP_reads=0,teacher_nodes_added=0))
    print('FREE_PARENT_OUTPUT',key,len(decoded['parents']),decoded['total_prediction_children'],flush=True)


def score_value(decoded,target,key):
    from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents
    result=score_native_parents(decoded,target,[False]*len(target['parents']))
    value=result['categories']['overall'];errors=[]
    for m in result['matches']:
        errors.extend(m['ordered_endpoint_errors_after_allowed_whole_correspondence_mm'] or [])
        errors.append(m['bidirectional_sampled_maximum_mm'])
    full=sum(sum(m['original_parent_index']==i for m in result['matches'])==len(t['children']) for i,t in enumerate(target['parents']))
    tp,g,fp=value['TP'],value['G'],value['FP']
    strict=tp==g and fp==0 and full==len(target['parents']) and len(decoded['parents'])==len(target['parents']) and not decoded['invalid_predictions'] and bool(errors) and max(errors)<=3.
    report=dict(source=key,TP=tp,G=g,FP=fp,FN=g-tp,extraction_correct_rate_percent=100*tp/g,
        output_precision_percent=100*tp/(tp+fp) if tp+fp else None,matched_maximum_geometry_error_mm=max(errors) if errors else None,
        fully_matched_parents=full,original_parents=len(target['parents']),output_parents=len(decoded['parents']),
        output_children=value['predicted'],strict_TRAIN_gate_pass=bool(strict),full_contract_pass=False)
    return dict(report=report,score=result)


def target_for(key):
    binding=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==key)
    assert digest(OUT/binding['target_file'])==binding['target_sha256']
    return read(OUT/binding['target_file']),binding


def reuse_frozen_upstream(dest,key,folder):
    """Pro355 permits exact current-upstream caches for independent set evaluation."""
    assert key in KEYS
    source=dest/f'source_{key}'
    assert read(dest/f'stage_match_{key}_complete.json')['status']=='STAGE_COMPLETE'
    r=read(source/'receipt.json');model=read(dest/'upstream_verified.json')
    assert r['upstream_model_sha256']==model['model_sha256']==digest(dest/'model.pkl')
    assert digest(source/'decoded.json')==r['decoded_sha256']
    assert digest(source/'unary.npy')==r['unary_sha256']
    assert digest(feature_path(key)/'receipt.json')==r['input_feature_receipt_sha256']
    folder.mkdir(exist_ok=False);bindings={}
    # Prediction artifacts only. No compatibility table, targets or scores copied.
    for name in ('decoded.json','selected_graph.json','unary.npy','parent_metadata.json','compact_parents.json','parent_points.npy'):
        before=digest(source/name);os.link(source/name,folder/name)
        assert digest(folder/name)==before
        bindings[name]=before
    write(folder/'receipt.json',dict(r,reused_same_round_same_upstream_cache=True,cache_source=str(source),
        reused_artifact_sha256=bindings,no_new_raw_input_inference_claim=True,GT_reads=0))
    print('EXACT_SAME_UPSTREAM_CANDIDATES_REUSED',key,'MODEL_SHA',model['model_sha256'],flush=True)


def match_train(folder,key):
    from weld_pipeline.sjg_parent_compatibility_1161 import compile_compatibility
    assert key in KEYS
    r=read(folder/'receipt.json');assert digest(folder/'decoded.json')==r['decoded_sha256']
    decoded=read(folder/'decoded.json');target,binding=target_for(key)
    compatibility=compile_compatibility(decoded,target)
    compatibility.update(source=key,prediction_sha256=r['decoded_sha256'],target_sha256=binding['target_sha256'])
    write(folder/'compatibility.json',compatibility)
    write(folder/'upstream_unselected_evaluation.json',score_value(decoded,target,key))
    print('CANDIDATE_UPPER_BOUND_NOT_ACCURACY',key,'PARENTS',compatibility['maximum_matched_parents'],
          '/',compatibility['original_parents'],'CHILDREN',compatibility['maximum_matched_children'],'/',compatibility['original_children'],flush=True)


def train_selector(dest,relation_factory=None,resume_from=None,time_limit_seconds=180.):
    from weld_pipeline.sjg_parent_set_scorer_1159 import ParentSetScorer,AdamState
    from weld_pipeline.sjg_parent_set_oracle_1158 import ParentSetOracle,set_action_loss
    from weld_pipeline.sjg_parent_set_features_1160 import ParentRelations
    begin=time.perf_counter();means=[];second=[];bindings={}
    if not np.isfinite(time_limit_seconds) or time_limit_seconds<=0:raise ValueError('Invalid training execution limit')
    for key in KEYS:
        folder=dest/f'source_{key}';r=read(folder/'receipt.json')
        assert r['upstream_model_sha256']==read(dest/'upstream_verified.json')['model_sha256']
        assert digest(folder/'unary.npy')==r['unary_sha256']
        if resume_from is None:
            u=np.load(folder/'unary.npy');means.append(u.mean(0) if len(u) else np.zeros(160))
            second.append((u*u).mean(0) if len(u) else np.zeros(160))
        bindings[key]=dict(candidate_sha256=r['decoded_sha256'],compatibility_sha256=digest(folder/'compatibility.json'))
    if resume_from is None:
        mean=np.mean(means,axis=0);scale=np.sqrt(np.maximum(0,np.mean(second,axis=0)-mean*mean));scale[scale<1e-8]=1.
    else:
        # Preserve the exact frozen normalization; never refit it on resume.
        with np.load(resume_from/'selector_normalization.npz',allow_pickle=False) as normalization:
            mean=normalization['mean'].copy();scale=normalization['scale'].copy()
        assert mean.shape==scale.shape==(160,) and np.isfinite(mean).all() and np.isfinite(scale).all() and np.all(scale>0)
    np.savez(dest/'selector_normalization.npz',mean=mean,scale=scale)
    model=ParentSetScorer(1031);optimizer=AdamState(model);trace=[]
    resume_updates=0;prior_training_seconds=0.
    if resume_from is not None:
        from weld_pipeline.sjg_selector_resume_state_1178 import restore
        restored=restore(resume_from,model,optimizer,mean,scale,bindings)
        resume_updates=optimizer.step_count;trace=list(restored['trace']);prior_training_seconds=restored['training_seconds']
        print('SET_TRAIN_RESTORED',resume_updates,'/112','NEXT',restored['next_epoch'],restored['next_source'],flush=True)
    for epoch in range(16):
        for case_index,key in enumerate(KEYS):
            if epoch*len(KEYS)+case_index<resume_updates:continue
            if time.perf_counter()-begin>time_limit_seconds:raise TimeoutError('Selector registered execution limit')
            check_resources();folder=dest/f'source_{key}';u=(np.load(folder/'unary.npy')-mean)/scale
            compat=read(folder/'compatibility.json');parents,points=load_parents(folder)
            metadata=read(folder/'parent_metadata.json')
            context=ParentRelations(parents,metadata) if relation_factory is None else relation_factory(parents,metadata,key)
            oracle=ParentSetOracle(compat['compatible_targets'],compat['original_parents'])
            accumulated={k:np.zeros_like(v) for k,v in model.parameters.items()};loss_sum=0.;steps=0
            while True:
                if time.perf_counter()-begin>time_limit_seconds:raise TimeoutError('Selector registered execution limit')
                scores,cache=model.forward(u,context.features(),context.selected)
                available,allowed=oracle.action_masks();loss,delta=set_action_loss(scores,available,allowed)
                gradients=model.backward(cache,delta)
                for k in accumulated:accumulated[k]+=gradients[k]
                loss_sum+=loss;steps+=1
                if allowed[-1]:break
                legal=np.flatnonzero(allowed[:-1]);chosen=int(legal[np.argmax(scores[legal])])
                assert oracle.select(chosen);context.add(chosen);check_resources()
            update=optimizer.update(model,{k:v/steps for k,v in accumulated.items()})
            trace.append(dict(epoch=epoch+1,source=key,prefixes=steps,mean_loss=loss_sum/steps,**update))
            # Resume evidence only. This checkpoint is never accepted as final weights.
            from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
            checkpoint={f'parameter_{k}':v for k,v in model.parameters.items()}
            checkpoint.update({f'first_{k}':v for k,v in optimizer.first.items()})
            checkpoint.update({f'second_{k}':v for k,v in optimizer.second.items()})
            checkpoint.update(updates=np.array(optimizer.step_count,dtype=np.int64),mean=mean,scale=scale)
            atomic_binary_write(dest/'selector_resume_checkpoint.npz',lambda stream:np.savez(stream,**checkpoint))
            position=optimizer.step_count
            write(dest/'selector_resume_checkpoint.json',dict(status='RESUME_ONLY_NOT_FINAL_MODEL',updates=position,
                next_epoch=position//len(KEYS)+1 if position<112 else None,
                next_source=KEYS[position%len(KEYS)] if position<112 else None,
                initialization_rng_state=model.initialization_rng_state,random_draws_after_initialization=0,
                checkpoint_sha256=digest(dest/'selector_resume_checkpoint.npz'),bindings=bindings,
                trace=trace,training_seconds=prior_training_seconds+time.perf_counter()-begin,
                completed_before_resume=resume_updates,current_process_seconds=time.perf_counter()-begin))
            write(dest/'selector_progress.json',dict(status='PARTIAL_UPDATES_NOT_FINAL_MODEL',updates=optimizer.step_count,
                registered_updates=112,last=trace[-1],seconds=time.perf_counter()-begin,
                completed_before_resume=resume_updates,new_updates=optimizer.step_count-resume_updates))
            print('SET_EPOCH',epoch+1,'/16','PIECE',key,'UPDATE',optimizer.step_count,'/112','LOSS',loss_sum/steps,flush=True)
            del parents,context;points._mmap.close();gc.collect()
    assert optimizer.step_count==112
    model.save(dest/'selector.npz');loaded=ParentSetScorer.load(dest/'selector.npz')
    assert all(np.array_equal(v,loaded.parameters[k]) for k,v in model.parameters.items())
    write(dest/'selector_training.json',dict(status='FINAL_SET_SELECTOR_TRAINED',updates=112,epochs=16,trace=trace,
        selector_sha256=digest(dest/'selector.npz'),normalization_sha256=digest(dest/'selector_normalization.npz'),
        bindings=bindings,seconds=time.perf_counter()-begin,DEV_reads=0,only_final_weights_evaluated=True,
        completed_before_resume=resume_updates,new_updates=112-resume_updates,
        prior_elapsed_at_completed_checkpoint=prior_training_seconds,
        resume_checkpoint_sha256=digest(resume_from/'selector_resume_checkpoint.npz') if resume_from is not None else None))


def select_final(dest,key,folder):
    from weld_pipeline.sjg_parent_set_scorer_1159 import ParentSetScorer
    from weld_pipeline.sjg_parent_set_features_1160 import ParentRelations
    begin=time.perf_counter();training=read(dest/'selector_training.json')
    assert digest(dest/'selector.npz')==training['selector_sha256']
    assert digest(dest/'selector_normalization.npz')==training['normalization_sha256']
    model=ParentSetScorer.load(dest/'selector.npz')
    with np.load(dest/'selector_normalization.npz') as z:u=(np.load(folder/'unary.npy')-z['mean'])/z['scale']
    parents,points=load_parents(folder);context=ParentRelations(parents,read(folder/'parent_metadata.json'))
    for step in range(len(parents)+1):
        if time.perf_counter()-begin>60:raise TimeoutError('Complete parent selection 60 second limit')
        check_resources();scores,_=model.forward(u,context.features(),context.selected)
        chosen=model.choose_next(scores,context.selected)
        if chosen is None:break
        context.add(chosen)
    else:raise RuntimeError('No STOP after all candidates')
    indices=list(context.selected);del parents,context;points._mmap.close()
    original=read(folder/'decoded.json');result=dict(original)
    result['parents']=[original['parents'][i] for i in indices]
    result['total_prediction_children']=sum(p['child_count'] for p in result['parents'])+sum(v['counted_false_children'] for v in original['invalid_predictions'])
    result.update(runtime_model='PRO355_RECALL_AND_LEARNED_PARENT_SET',GT_reads=0)
    write(folder/'final_decoded.json',result)
    write(folder/'final_prediction.json',dict(status='LABEL_FREE_SET_PREDICTION_FROZEN',source=key,
        selected_candidate_indices=indices,candidate_count=len(original['parents']),
        decoded_sha256=digest(folder/'final_decoded.json'),selector_sha256=training['selector_sha256'],
        invalid_ledger_unchanged=True,GT_reads=0,seconds=time.perf_counter()-begin))


def final_score(folder,key,candidate_compatibility_path=None):
    from weld_pipeline.sjg_parent_compatibility_1161 import compile_compatibility
    from weld_pipeline.sjg_parent_set_oracle_1158 import ParentSetOracle
    r=read(folder/'final_prediction.json');assert digest(folder/'final_decoded.json')==r['decoded_sha256']
    target,binding=target_for(key);decoded=read(folder/'final_decoded.json')
    value=score_value(decoded,target,key);value['report'].update(scope='TRAIN' if key in KEYS else 'DEV',
        candidate_count=r['candidate_count'],target_sha256=binding['target_sha256'],prediction_sha256=r['decoded_sha256'])
    before=read(folder/'decoded.json')
    if candidate_compatibility_path is None:
        compat=compile_compatibility(before,target)
    else:
        # Post-prediction TRAIN diagnostics only; the final D3 score above is
        # still independently recomputed from the newly frozen selected output.
        assert key in KEYS
        compat=read(candidate_compatibility_path)
        assert compat['prediction_sha256']==digest(folder/'decoded.json')
        assert compat['target_sha256']==binding['target_sha256']
    oracle=ParentSetOracle(compat['compatible_targets'],compat['original_parents']);duplicates=0;wrong=0
    for index in r['selected_candidate_indices']:
        augmented=oracle.select(index)
        if not augmented:
            if compat['compatible_targets'][index]:duplicates+=1
            else:wrong+=1
    value['report'].update(upstream_uncovered_GT_parents=len(compat['uncovered_target_parents']),
        repeated_compatible_parent_selections=duplicates,incompatible_parent_selections=wrong,
        coverable_parents_missed_at_stop=compat['maximum_matched_parents']-oracle.match_count)
    write(folder/'evaluation.json',value)
    print('ACTUAL_EXTRACTION_METRIC',value['report'],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--stage',required=True);ap.add_argument('--source',default='')
    args=ap.parse_args();dest=OUT/args.run;begin=time.perf_counter()
    folder=dest/(('test_' if args.stage in ('test_upstream','select','score') else 'source_')+args.source)
    try:
        check_resources()
        if args.stage=='object':fit_object(dest)
        elif args.stage=='package':package(dest)
        elif args.stage=='verify':
            r=read(dest/'upstream_saved.json');assert digest(dest/'model.pkl')==r['model_sha256']
            with (dest/'model.pkl').open('rb') as stream:models=pickle.load(stream)
            assert r['component_state_sha256']=={k:forest_state_sha256(models[k]) for k in r['component_state_sha256']}
            write(dest/'upstream_verified.json',dict(r,status='UPSTREAM_VERIFIED',independent_process=True))
        elif args.stage=='test_upstream' and args.source in KEYS:reuse_frozen_upstream(dest,args.source,folder)
        elif args.stage in ('upstream','test_upstream'):upstream(dest,args.source,folder)
        elif args.stage=='match':match_train(folder,args.source)
        elif args.stage=='train_selector':train_selector(dest)
        elif args.stage=='select':select_final(dest,args.source,folder)
        elif args.stage=='score':final_score(folder,args.source)
        else:raise ValueError(args.stage)
        write(dest/f'stage_{args.stage}_{args.source}_complete.json',dict(status='STAGE_COMPLETE',seconds=time.perf_counter()-begin,
            worker_source_sha256=digest(__file__)))
    except BaseException:
        write(dest/f'stage_{args.stage}_{args.source}_failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-begin));raise


if __name__=='__main__':main()
