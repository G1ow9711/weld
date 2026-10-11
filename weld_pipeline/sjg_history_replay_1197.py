"""Pro368 fixed 50% fresh / 50% two-round history, no deploy-time change."""
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest
from weld_pipeline.sjg_prefix_minibatch_1189 import evaluate_prefixes,batch_weights,partition,parameter_digest


def history_pool(dest,keys,candidate_bindings):
    manifest=[]
    for run in ('run367','run368'):
        folder=OUT/run;training=read(folder/'selector_training.json')
        assert training['bindings']==candidate_bindings
        assert digest(folder/'selector.npz')==training['selector_sha256']
        trace=training['trace'][-112:]
        for e in range(4):
            for key in keys:
                path=folder/f'trajectories/pass_{e+1:02}_{key}.json'
                pair=read(path);sha=digest(path)
                assert pair['candidate_binding']==candidate_bindings[key]
                rows=[r for r in trace if r['stage_epoch']==e+1 and r['source']==key]
                assert len(rows)==4 and all(r['trajectory_sha256']==sha for r in rows)
                for name in ('clean','recovery'):
                    t=pair[name];assert t['source']==key and t['traversal_zero_based']==e
                    assert len(t['choices'])+1==t['prefix_count'] and len(set(t['choices']))==len(t['choices'])
                    batches,counts=partition(t['prefix_count'],e)
                    assert batches==t['batches'] and counts.tolist()==t['prefix_occurrences']
                manifest.append(dict(run=run,source=key,traversal=e,path=str(path),sha256=sha))
    assert len(manifest)==56
    write(dest/'history_pool_manifest.json',dict(status='ALL_56_VISITS_112_TRAJECTORIES_BOUND',entries=manifest,
        selection_by_metrics=False,trajectory_answers_recomputed_by_dynamic_oracle=True))
    return manifest


def bundle(dest,key,e,clean,recovery):
    assert clean['model_state_sha256']==recovery['model_state_sha256']
    entries=[dict(origin='fresh',kind='clean',alpha=.25,trajectory=clean),
             dict(origin='fresh',kind='recovery',alpha=.25,trajectory=recovery)]
    for row in read(dest/'history_pool_manifest.json')['entries']:
        if row['source']!=key or row['traversal']!=e:continue
        assert digest(row['path'])==row['sha256'];pair=read(row['path'])
        for name in ('clean','recovery'):
            entries.append(dict(origin=row['run'],kind=name,alpha=.125,trajectory=pair[name],
                historical_file=row['path'],historical_sha256=row['sha256']))
    assert len(entries)==6 and sum(v['alpha'] for v in entries)==1
    return dict(source=key,traversal=e,entries=entries,
        historical_actions_are_states_not_supervision=True,actual_model_updates=0)


def replay_loss(model,u,parents,metadata,compat,key,cache,pack,batch=None):
    loss=0.;gradient={k:np.zeros_like(v) for k,v in model.parameters.items()}
    for item in pack['entries']:
        t=item['trajectory'];alpha=item['alpha'];T=t['prefix_count']
        assert T<=len(parents)+1
        weights={p:alpha/T for p in range(T)} if batch is None else {p:alpha*w for p,w in batch_weights(t,batch).items()}
        value,g=evaluate_prefixes(model,u,parents,metadata,compat,key,cache,t,weights)
        loss+=value
        for k in gradient:gradient[k]+=g[k]
    return loss,gradient


def replay_control(model,u,parents,metadata,compat,key,cache,pack):
    before=parameter_digest(model)
    loss,g=replay_loss(model,u,parents,metadata,compat,key,cache,pack)
    batches=[replay_loss(model,u,parents,metadata,compat,key,cache,pack,b) for b in range(4)]
    loss_error=abs(sum(v[0] for v in batches)/4-loss)
    assert np.isclose(sum(v[0] for v in batches)/4,loss,rtol=1e-11,atol=1e-12)
    errors={}
    for k in g:
        average=sum(v[1][k] for v in batches)/4
        errors[k]=float(np.abs(average-g[k]).max())
        assert np.allclose(average,g[k],rtol=1e-11,atol=1e-12)
    t=pack['entries'][0]['trajectory'];T=t['prefix_count']
    same={'entries':[dict(alpha=item['alpha'],trajectory=t) for item in pack['entries']]}
    same_loss,same_g=replay_loss(model,u,parents,metadata,compat,key,cache,same)
    clean_loss,clean_g=evaluate_prefixes(model,u,parents,metadata,compat,key,cache,t,{p:1./T for p in range(T)})
    assert np.isclose(same_loss,clean_loss,rtol=1e-11,atol=1e-12)
    assert all(np.allclose(same_g[k],v,rtol=1e-11,atol=1e-12) for k,v in clean_g.items())
    assert parameter_digest(model)==before
    return dict(status='SIX_TRAJECTORY_WEIGHT_AND_ORDER_CONTROL_PASSED',loss_error=float(loss_error),
        gradient_max_errors=errors,all_historical_masks_rebuilt_in_original_selection_order=True,
        identical_six_trajectories_reduce_to_original=True,actual_training_updates=0)
