"""Pro376: full run376 restoration and five fixed incompatible actions."""
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources
from weld_pipeline.sjg_incompatibility_head_1216 import IncompatibilityScorer,IncompatibilityAdam,SHAPES
from weld_pipeline.sjg_incompatibility_worker_1217 import initialize as initialize_frozen
from weld_pipeline.sjg_duplicate_head_1209 import DuplicateRelations
from weld_pipeline.sjg_prefix_minibatch_1189 import parameter_digest
from weld_pipeline.sjg_history_replay_worker_1198 import inputs
from weld_pipeline.sjg_synchronous_pool_1200 import ReadOnlyColumns
from weld_pipeline.sjg_selector_resume_state_1178 import reuse_columns
from weld_pipeline.sjg_selector_trial_worker_1174 import bindings
from weld_pipeline.sjg_seven_relative_prepare_1146 import KEYS


def restore(dest):
    unused,unused_adam,mean,scale,receipt=initialize_frozen(dest)
    old=OUT/'run376';t=read(old/'selector_training.json');r=read(old/'selector_resume_checkpoint.json')
    assert t['updates']==r['updates']==r['registered_updates']==112 and r['next_position'] is None
    assert r['trace']==t['trace'] and [v['step'] for v in r['trace']]==list(range(1,113))
    assert t['bindings']==r['bindings']==receipt['bindings']
    assert digest(old/'selector.npz')==t['selector_sha256'] and digest(old/'selector_resume_checkpoint.npz')==r['checkpoint_sha256']
    assert digest(old/'selector_normalization.npz')==t['normalization_sha256']==digest(OUT/'run367/selector_normalization.npz')
    assert r['base_checkpoint_sha256']==receipt['checkpoint_sha256']
    for name in ('frozen_residual_checkpoint_sha256','frozen_duplicate_checkpoint_sha256'):assert r[name]==receipt[name]
    model=IncompatibilityScorer.load(old/'selector.npz',old/'frozen_duplicate.npz',old/'frozen_residual.npz',old/'base_selector.npz')
    adam=IncompatibilityAdam(model)
    with np.load(old/'selector_resume_checkpoint.npz') as z:
        assert set(z.files)=={f'{a}_{b}' for a in ('parameter','first','second') for b in SHAPES}|{'updates','mean','scale'}
        assert int(z['updates'])==112 and np.array_equal(z['mean'],mean) and np.array_equal(z['scale'],scale)
        for key,shape in SHAPES.items():
            assert np.array_equal(z['parameter_'+key],model.parameters[key])
            for prefix,target in (('first',adam.first),('second',adam.second)):
                value=z[prefix+'_'+key];assert value.shape==shape and value.dtype==np.float64 and np.isfinite(value).all()
                if prefix=='second':assert np.all(value>=0)
                target[key]=value.copy()
    assert r['initialization_rng_state']==model.initialization_rng_state and r['random_draws_after_initialization']==0
    model.resume_rng.bit_generator.state=r['initialization_rng_state'];adam.step_count=112
    receipt.update(head_resume=r,head_source_sha256=t['selector_sha256'],head_source_checkpoint_sha256=r['checkpoint_sha256'])
    return model,adam,mean,scale,receipt


def control(dest):
    model,adam,mean,scale,receipt=restore(dest);old=OUT/'run376'
    assert digest(ROOT/'weld_pipeline/sjg_incompatibility_head_1216.py')==digest(old/'source_snapshot/sjg_incompatibility_head_1216.py')
    cache=ReadOnlyColumns(dest/'control_relation_cache',bindings(dest,KEYS));reuse_columns(old/'control_relation_cache',cache)
    reference=IncompatibilityScorer.load(old/'selector.npz',old/'frozen_duplicate.npz',old/'frozen_residual.npz',old/'base_selector.npz')
    key='10';u,parents,metadata,compat,points=inputs(dest,key,mean,scale)
    a=DuplicateRelations(parents,metadata,reference.frozen,key,cache);b=DuplicateRelations(parents,metadata,model.frozen,key,cache)
    try:
        choices=read(old/f'test_{key}/final_prediction.json')['selected_candidate_indices'];assert len(choices)>1
        a.add(choices[0]);b.add(choices[0]);za,_=reference.logits(u);zb,_=model.logits(u)
        sa=reference.forward(u,a.features(),a.selected,za);sb=model.forward(u,b.features(),b.selected,zb)
        assert np.array_equal(za,zb) and np.array_equal(sa,sb)
        assert reference.choose_next(sa,a.selected)==model.choose_next(sb,b.selected)==choices[1]
        assert adam.step_count==112
        write(dest/'resume_incompatibility_control.json',dict(status='HEAD112_COMPLETE_STATE_AND_NONEMPTY_PREFIX_EXACT',
            source=key,prefix_length=1,score_reference='Original frozen run376 implementation and final weights',
            parameters_moments_rng_normalization_bindings_verified=True,head_sha256=receipt['head_source_sha256'],
            head_checkpoint_sha256=receipt['head_source_checkpoint_sha256'],model_updates=0))
        print('FULL_INCOMPATIBILITY_HEAD112_RESTORE_PASSED',flush=True)
    finally:
        del a,b,parents
        points._mmap.close()
        write(dest/'control_relation_cache_receipt.json',cache.receipt())


def fixed_diagnostic(dest,model,mean,scale,cache,phase):
    key='10';saved=OUT/'run376/test_10/final_prediction.json';choices=read(saved)['selected_candidate_indices']
    u,parents,metadata,compat,points=inputs(dest,key,mean,scale);logits,_=model.logits(u)
    context=DuplicateRelations(parents,metadata,model.frozen,key,cache);rows=[]
    try:
        for prefix,chosen in enumerate(choices):
            check_resources()
            if not compat['compatible_targets'][chosen]:
                frozen_scores,_=model.frozen.forward(u,context.features(),context.selected)
                b=float(frozen_scores[chosen]-frozen_scores[-1]);z=float(logits[chosen]);c=max(0.,z);m=b-c
                kind='inactive_wrong_sign' if z<=0 else ('active_but_not_below_STOP' if m>=0 else 'below_STOP')
                if phase=='before':assert m>=-1e-12
                rows.append(dict(prefix_length=prefix,candidate_index=chosen,child_count=parents[chosen]['child_count'],
                    frozen375_relative_STOP=b,raw_logit=z,deduction=c,remaining_relative_STOP=m,category=kind))
            context.add(chosen)
        assert len(rows)==5 and sum(v['child_count'] for v in rows)==9
        if phase=='after':
            before=read(dest/'frozen376_incompatible_before.json')
            assert before['frozen_prediction_sha256']==digest(saved)
            assert [(v['prefix_length'],v['candidate_index'],v['frozen375_relative_STOP']) for v in before['actions']]==[(v['prefix_length'],v['candidate_index'],v['frozen375_relative_STOP']) for v in rows]
        groups={kind:dict(parents=sum(v['category']==kind for v in rows),children=sum(v['child_count'] for v in rows if v['category']==kind))
            for kind in ('inactive_wrong_sign','active_but_not_below_STOP','below_STOP')}
        result=dict(status='FIVE_RUN376_INCOMPATIBLE_ACTIONS_FIXED_STATE_REPLAYED',phase=phase,actions=rows,groups=groups,
            frozen_prediction_sha256=digest(saved),parameter_digest=parameter_digest(model),model_updates=0,not_extraction_metric=True)
        write(dest/f'frozen376_incompatible_{phase}.json',result)
        print('FIXED376_INCOMPATIBLE_DIAGNOSTIC',phase,groups,flush=True)
    finally:
        del context,parents
        points._mmap.close()
