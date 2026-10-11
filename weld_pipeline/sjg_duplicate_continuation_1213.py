"""Pro374: restore full head state, exact clip repair, fixed-state deduction audit."""
import copy,importlib.util
import numpy as np
from types import SimpleNamespace
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources
from weld_pipeline.sjg_duplicate_head_1209 import DuplicateScorer,DuplicateAdam,DuplicateRelations,SHAPES,reverse_column
from weld_pipeline.sjg_duplicate_supervision_1210 import initialize
from weld_pipeline.sjg_prefix_minibatch_1189 import parameter_digest


def restore_head(dest):
    fresh,unused,mean,scale,receipt=initialize(dest)
    old=OUT/'run374';t=read(old/'selector_training.json');r=read(old/'selector_resume_checkpoint.json')
    assert r['updates']==t['updates']==r['registered_updates']==112 and r['next_position'] is None
    assert len(r['trace'])==112 and [v['step'] for v in r['trace']]==list(range(1,113))
    assert r['bindings']==receipt['bindings']==t['bindings']
    assert digest(old/'selector.npz')==t['selector_sha256']
    assert digest(old/'selector_resume_checkpoint.npz')==r['checkpoint_sha256']
    assert r['base_checkpoint_sha256']==receipt['checkpoint_sha256']
    assert r['frozen_residual_checkpoint_sha256']==receipt['frozen_residual_checkpoint_sha256']
    model=DuplicateScorer.load(old/'selector.npz',old/'frozen_residual.npz',old/'base_selector.npz');adam=DuplicateAdam(model)
    with np.load(old/'selector_resume_checkpoint.npz') as z:
        assert set(z.files)=={f'{prefix}_{key}' for prefix in ('parameter','first','second') for key in SHAPES}|{'updates','mean','scale'}
        assert int(z['updates'])==112 and np.array_equal(z['mean'],mean) and np.array_equal(z['scale'],scale)
        for key,shape in SHAPES.items():
            assert np.array_equal(z[f'parameter_{key}'],model.parameters[key])
            for prefix,target in (('first',adam.first),('second',adam.second)):
                v=z[f'{prefix}_{key}'];assert v.shape==shape and v.dtype==np.float64 and np.isfinite(v).all()
                if prefix=='second':assert np.all(v>=0)
                target[key]=v.copy()
    assert r['initialization_rng_state']==model.initialization_rng_state and r['random_draws_after_initialization']==0
    model.resume_rng.bit_generator.state=r['initialization_rng_state'];adam.step_count=112
    receipt.update(head_resume=r,head_source_sha256=t['selector_sha256'],head_source_checkpoint_sha256=r['checkpoint_sha256'])
    return model,adam,mean,scale,receipt


def old_module():
    path=OUT/'run374/source_snapshot/sjg_duplicate_head_1209.py'
    spec=importlib.util.spec_from_file_location('run374_frozen_duplicate_head',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def restore_and_clip_control(dest):
    model,adam,mean,scale,receipt=restore_head(dest);old=old_module();norms=[]
    with np.errstate(over='ignore',divide='ignore'):
        for row in receipt['head_resume']['trace']:
            norm=row['raw_gradient_norm'];previous=min(1.,5./max(norm,np.finfo(float).tiny));current=1. if norm<=5. else 5./norm
            assert previous==current==row['clip_scale'];norms.append(norm)
    a=copy.deepcopy(model);b=copy.deepcopy(model);oa=old.DuplicateAdam(a);ob=copy.deepcopy(adam)
    oa.step_count=adam.step_count;oa.first={k:v.copy() for k,v in adam.first.items()};oa.second={k:v.copy() for k,v in adam.second.items()}
    rng=np.random.Generator(np.random.PCG64(1213));tests=[]
    for name,target in (('zero_with_historical_moments',0.),('nonzero_unclipped',.25),('nonzero_clipped',10.)):
        gradients={k:rng.normal(size=v.shape) for k,v in a.parameters.items()}
        norm=np.sqrt(sum(np.sum(v*v) for v in gradients.values()))
        gradients={k:v*(target/norm) for k,v in gradients.items()}
        before={k:v.copy() for k,v in a.parameters.items()}
        with np.errstate(over='ignore',divide='ignore'):ra=oa.update(a,gradients)
        rb=ob.update(b,gradients);assert ra==rb and oa.step_count==ob.step_count
        assert all(np.array_equal(v,b.parameters[k]) for k,v in a.parameters.items())
        assert all(np.array_equal(v,ob.first[k]) for k,v in oa.first.items())
        assert all(np.array_equal(v,ob.second[k]) for k,v in oa.second.items())
        changed=any(not np.array_equal(v,a.parameters[k]) for k,v in before.items())
        if target==0:assert changed
        tests.append(dict(kind=name,parameter_and_moment_arrays_exact=True,parameters_changed=changed,**rb))
    assert adam.step_count==112 and parameter_digest(model)==parameter_digest(restore_head(dest)[0])
    write(dest/'resume_clip_control.json',dict(status='COMPLETE_HEAD112_RESTORED_AND_CLIP_REPAIR_EXACT',
        recorded_norms_checked=len(norms),zero_norms_checked=sum(v==0 for v in norms),control_tests=tests,
        discarded_control_optimizer_calls=3,formal_training_updates=0,zero_gradient_updates_not_skipped=True,
        checkpoint_sha256=receipt['head_source_checkpoint_sha256'],head_sha256=receipt['head_source_sha256']))


def seed_prior_pair_checks(dest,supervision,cache,keys):
    """Rebuild the prior exact-input conflict registry from frozen trajectories."""
    old=OUT/'run374';files=list((old/'trajectories').glob('*.json'));assert len(files)==28
    columns={key:set() for key in keys};training=read(old/'selector_training.json')
    for path in files:
        pack=read(path);sha=digest(path);key=pack['clean']['source']
        assert any(row['trajectory_sha256']==sha for row in training['trace'])
        for kind in ('clean','recovery'):columns[key].update(pack[kind]['choices'])
    count=0
    for key in keys:
        metadata=read(dest/f'source_{key}/parent_metadata.json');lengths=[m['length_mm'] for m in metadata]
        context=SimpleNamespace(parents=[None]*len(metadata))
        for index in sorted(columns[key]):
            check_resources();assert (cache.root/key/(str(index)+'.npy')).exists()
            a=cache.column(key,index,context);b=reverse_column(a,index,lengths)
            supervision.column(key,index,a,b);count+=1
    previous=read(old/'pair_supervision_receipt.json');current=supervision.receipt()
    assert current['columns']==previous['columns']==count
    assert current['unique_symmetric_inputs']==previous['unique_symmetric_inputs']
    assert current['labelled_directional_pairs_checked']==previous['labelled_directional_pairs_checked']
    write(dest/'prior_pair_conflict_registry_restored.json',dict(current,status='ALL_PRIOR_CONSTRUCTED_PAIR_INPUTS_RECHECKED',previous_run='run374'))


def fixed_actual_diagnostic(dest,model,mean,scale,cache,phase):
    from weld_pipeline.sjg_history_replay_worker_1198 import inputs
    from weld_pipeline.sjg_parent_set_oracle_1158 import ParentSetOracle
    key='10';old=OUT/'run374';saved_path=old/f'test_{key}/final_prediction.json';saved=read(saved_path)
    choices=saved['selected_candidate_indices'];u,parents,metadata,compat,points=inputs(dest,key,mean,scale)
    ctx=DuplicateRelations(parents,metadata,model,key,cache);oracle=ParentSetOracle(compat['compatible_targets'],compat['original_parents'])
    reference=None;ref_ctx=None
    if phase=='before':
        module=old_module();reference=module.DuplicateScorer.load(old/'selector.npz',old/'frozen_residual.npz',old/'base_selector.npz')
        ref_ctx=module.DuplicateRelations(parents,metadata,reference,key,cache)
    raw_max=np.full(len(parents),-np.inf);rows=[];equal_prefixes=[]
    try:
        for prefix in range(len(choices)+1):
            check_resources();scores,_=model.forward(u,ctx.features(),ctx.selected)
            if phase=='before':
                if prefix in (0,1,len(choices)):
                    reference_scores,_=reference.forward(u,ref_ctx.features(),ref_ctx.selected)
                    assert np.array_equal(scores,reference_scores);equal_prefixes.append(prefix)
                expected=choices[prefix] if prefix<len(choices) else None
                assert model.choose_next(scores,ctx.selected)==expected
            if prefix==len(choices):break
            chosen=choices[prefix];available,allowed=oracle.action_masks();assert available[chosen]
            if not allowed[chosen]:
                kind='duplicate' if compat['compatible_targets'][chosen] else 'incompatible'
                frozen_scores,_=model.frozen.forward(u,ctx.features()[0],ctx.selected)
                b=float(frozen_scores[chosen]-frozen_scores[-1]);c=float(ctx.head.maximum[chosen]);m=float(scores[chosen]-scores[-1])
                assert abs((b-c)-m)<1e-12
                hmax=float(raw_max[chosen]) if ctx.selected else None
                rows.append(dict(prefix_length=prefix,candidate_index=chosen,kind=kind,child_count=parents[chosen]['child_count'],
                    frozen_base_relative_STOP=b,deduction=c,remaining_relative_STOP=m,maximum_raw_pair_score=hmax,
                    deduction_active=c>0,all_pair_scores_strictly_negative=bool(ctx.selected) and hmax<0,
                    empty_selected_set=not bool(ctx.selected),set_loss_local_head_route_zero=c==0,
                    current_argmax_candidate=model.choose_next(scores,ctx.selected)))
            oracle.select(chosen)
            a=ctx.raw_column(chosen);reverse=reverse_column(a,chosen,ctx.lengths)
            h,_=model.pair(a,reverse);raw_max=np.maximum(raw_max,h)
            ctx.add(chosen)
            if ref_ctx is not None:ref_ctx.add(chosen)
        groups={}
        for kind in ('duplicate','incompatible'):
            selected=[v for v in rows if v['kind']==kind];margins=[v['remaining_relative_STOP'] for v in selected]
            groups[kind]=dict(actions=len(selected),children=sum(v['child_count'] for v in selected),
                positive_remaining_margin=sum(v>0 for v in margins),active_deductions=sum(v['deduction_active'] for v in selected),
                all_pairs_negative=sum(v['all_pair_scores_strictly_negative'] for v in selected),
                zero_local_route=sum(v['set_loss_local_head_route_zero'] for v in selected),
                margin_min=min(margins) if margins else None,margin_mean=float(np.mean(margins)) if margins else None,
                margin_max=max(margins) if margins else None)
        assert groups['duplicate']['actions']==18 and groups['incompatible']['actions']==8
        result=dict(status='FROZEN_RUN374_ACTUAL_ERROR_ACTIONS_REPLAYED',phase=phase,groups=groups,actions=rows,
            frozen_prediction_sha256=digest(saved_path),checkpoint_parameter_digest=parameter_digest(model),
            restored_forward_equal_prefixes=equal_prefixes,actual_model_updates=0,diagnostic_not_extraction_score=True)
        if phase=='after':
            before=read(dest/'frozen374_actual_before.json');assert before['frozen_prediction_sha256']==result['frozen_prediction_sha256']
            assert [(v['prefix_length'],v['candidate_index'],v['kind'],v['frozen_base_relative_STOP']) for v in before['actions']]==[(v['prefix_length'],v['candidate_index'],v['kind'],v['frozen_base_relative_STOP']) for v in rows]
        write(dest/f'frozen374_actual_{phase}.json',result)
        print('FIXED374_ERROR_ROUTE_DIAGNOSTIC',phase,groups,flush=True)
    finally:
        del ctx,ref_ctx,parents
        points._mmap.close()
