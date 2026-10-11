"""Three sequential TRAIN sources, bounded native AdamW and child risk loss.

Fixed and rematched objectives share 734. Every source retains its own complete
denominators. Replay is an equal mean, never extra independent ground truth.
"""
from copy import deepcopy
import gc
import math
import torch
from weld_pipeline.bounded_native_direction_730 import flat,prepare_bounded_blocks,bounded_direction
from weld_pipeline.child_risk_parent_set_loss_734 import event_loss,conditioned_nodes,resolved_views,world_positions,VERSION as LOSS_VERSION
from weld_pipeline.legacy_scale_count_training_loop_488 import analytic_adamw_proposal
from weld_pipeline.legal_allocation_finite_update_521 import rng_state,restore_rng,trial_submit

VERSION='BOUNDED_NATIVE_CHILD_RISK_SEQUENTIAL_THREE_SOURCE_REPLAY_V1'


def replay_indices(main):
    assert type(main) is int and 0<=main<65
    return (main,(main+22)%65,(main+43)%65)


def fixed_observables(output,source,plan,events,group):
    mask=torch.as_tensor(source['group_ids']==group,device=output['presence_logits'].device)
    logits=torch.cat([value[mask].double().reshape(-1) for key,value in output.items()
        if isinstance(value,torch.Tensor) and value.ndim and len(value)==len(mask)
        and 'logits' in key and 'allocation' not in key])
    nodes=[];cache={}
    for i,pairs in plan['pairs'].items():
        if int(source['group_ids'][i])!=group:continue
        for q,p,v in pairs:
            view=list(resolved_views(source['parents'][i][p]))[v]
            values,_=conditioned_nodes(output,i,view,events,cache)
            nodes.append(world_positions(values[q],view['candidate_geometry']).double().reshape(-1,3))
    endpoints=torch.cat(nodes) if nodes else logits.new_empty((0,3))
    assert torch.isfinite(logits).all() and torch.isfinite(endpoints).all()
    return logits.detach().cpu(),endpoints.detach().cpu()


def replay_event_transaction(model,optimizer,source_loader,main_source,branch,device='cuda',*,commit=True,progress=None):
    group={'H':0,'V':1}[branch];indices=replay_indices(main_source)
    parameters=list(getattr(model.events,branch).parameters())
    before=[p.detach().clone() for p in parameters]
    old_grad=[None if p.grad is None else p.grad.detach().clone() for p in parameters]
    saved_state=deepcopy(optimizer.state_dict());original_rng=rng_state();native_calls=0
    baselines={};loader_calls=0
    def emit(stage,**kwargs):
        if progress is not None:progress(dict(stage=stage,branch=branch,main_source=main_source,**kwargs))
    def restore(*,gradients=False):
        with torch.no_grad():
            for p,old in zip(parameters,before):p.copy_(old)
        optimizer.load_state_dict(deepcopy(saved_state));restore_rng(original_rng)
        if gradients:
            for p,g in zip(parameters,old_grad):p.grad=None if g is None else g.clone()
    def one_source(index,*,backward=False,baseline=None):
        nonlocal loader_calls
        emit('LOAD_REPLAY_SOURCE',replay_source=index)
        source=source_loader(index);loader_calls+=1
        base={k:torch.as_tensor(v,device=device) for k,v in source['base'].items()}
        features=torch.as_tensor(source['features'],device=device)
        args=(source['parents'],base,torch.as_tensor(source['background_known'],device=device),
            torch.as_tensor(source['group_ids'],device=device),torch.as_tensor(source['hard_negative_weights'],device=device),
            source['original_G'],source['original_G_by_group'])
        output=model.events(features,base)
        _,fixed=event_loss(output,*args,events=model.events,transaction_plan=None if baseline is None else baseline['plan'])
        plan=fixed['transaction_plan_TRAIN_only']
        loss=fixed['loss_by_group'][group];value=float(loss.detach());assert math.isfinite(value)
        observed=fixed_observables(output,source,plan,model.events,group)
        if backward:
            grads=torch.autograd.grad(loss/3.,parameters,allow_unused=True)
            for p,g in zip(parameters,grads):
                if g is not None:p.grad=g.detach().clone() if p.grad is None else p.grad.add_(g.detach())
        result=dict(source=index,fixed_loss=value,plan=plan,observed=observed,
            original_G=source['original_G'],original_G_by_group=source['original_G_by_group'],
            original_E16_records=source['original_E16_records'],original_local_points=source['original_local_points'],
            components={k:float(v.detach()) for k,v in fixed['group_components'][group].items()},
            effective_supervision=sum(len(rows) for i,rows in enumerate(source['parents']) if int(source['group_ids'][i])==group)
                +int(((source['group_ids']==group)&source['background_known']).sum()))
        if baseline is not None:
            _,dynamic=event_loss(output,*args,events=model.events)
            dv=float(dynamic['loss_by_group'][group]);assert math.isfinite(dv)
            assert dv<=value+1e-9*max(1.,abs(value),abs(dv)), 'Unified rematching objective exceeds fixed pairing'
            for key in ('original_G','original_G_by_group','original_E16_records','original_local_points'):
                assert result[key]==baseline[key],key
            assert plan==baseline['plan']
            logits,endpoints=observed;old_logits,old_endpoints=baseline['observed']
            assert logits.shape==old_logits.shape and endpoints.shape==old_endpoints.shape
            result.update(dynamic_loss=dv,dynamic_components={k:float(v) for k,v in dynamic['group_components'][group].items()},
                max_classification_logit_change=float((logits-old_logits).abs().max()) if logits.numel() else 0.,
                max_fixed_endpoint_move_mm=float(torch.linalg.vector_norm(endpoints-old_endpoints,dim=-1).max()) if endpoints.numel() else 0.)
        # Only small CPU plans and observables survive. No prepared source or
        # autograd graph from a different workpiece is retained by this routine.
        return result
    try:
        model.eval();optimizer.zero_grad(set_to_none=True)
        for index in indices:
            baselines[index]=one_source(index,backward=True);gc.collect()
        initial=sum(r['fixed_loss'] for r in baselines.values())/3.
        if not sum(r['effective_supervision'] for r in baselines.values()):
            restore(gradients=True)
            return dict(version=VERSION,branch=branch,replay_sources=list(indices),accepted=False,no_effective_supervision=True,
                committed_optimizer_steps=0,native_proposal_step_calls=0,own_parameters_optimizer_RNG_restored=True)
        raw=flat([torch.zeros_like(p) if p.grad is None else p.grad for p in parameters])
        assert torch.isfinite(raw).all()
        preclip=float(torch.nn.utils.clip_grad_norm_(parameters,5.));assert math.isfinite(preclip)
        blocks=prepare_bounded_blocks(optimizer,parameters,raw)
        proposal=analytic_adamw_proposal(optimizer,parameters)
        bounded,direction=bounded_direction(proposal,raw,blocks)
        @torch.no_grad()
        def evaluate():
            results=[]
            for index in indices:
                r=one_source(index,baseline=baselines[index]);r.pop('plan');r.pop('observed');results.append(r);gc.collect()
            fv=sum(r['fixed_loss'] for r in results)/3.;dv=sum(r['dynamic_loss'] for r in results)/3.
            delta=flat([p.double()-old.double() for p,old in zip(parameters,before)])
            dot=float(raw@delta);nonzero=bool((delta!=0).any())
            logits=max(r['max_classification_logit_change'] for r in results)
            endpoint=max(r['max_fixed_endpoint_move_mm'] for r in results)
            return dict(fixed_before=initial,fixed_after=fv,dynamic_before=initial,dynamic_after=dv,
                g_dot_actual_delta=dot,actual_nonzero=nonzero,actual_delta_norm=float(torch.linalg.vector_norm(delta)),
                armijo_bound=initial+1e-4*dot,max_classification_logit_change=logits,max_fixed_endpoint_move_mm=endpoint,
                pass_all=nonzero and dot<0. and fv<=initial+1e-4*dot and dv<initial and logits<=.1 and endpoint<=.5,
                rematching_not_worse_than_fixed=True,per_source=results)
        zero=evaluate()
        assert zero['fixed_after']==initial and zero['dynamic_after']==initial
        assert zero['max_classification_logit_change']==zero['max_fixed_endpoint_move_mm']==0.
        trial_submit(parameters,before,bounded,1.);full=evaluate()
        alpha=min(1.,.1/full['max_classification_logit_change'] if full['max_classification_logit_change'] else 1.,
            .5/full['max_fixed_endpoint_move_mm'] if full['max_fixed_endpoint_move_mm'] else 1.)
        assert math.isfinite(alpha) and alpha>0.
        trials=[];chosen=None
        for divisor in (1,2,4,8,16,32):
            fraction=alpha/divisor;trial_submit(parameters,before,bounded,fraction)
            result=evaluate();trials.append(dict(fraction=fraction,**result))
            emit('BOUNDED_REPLAY_TRIAL',fraction=fraction,loss_before=initial,loss_after=result['dynamic_after'],accepted=result['pass_all'])
            if result['pass_all']:chosen=fraction;break
        evidence=dict(version=VERSION,loss_version=LOSS_VERSION,branch=branch,main_source=main_source,replay_sources=list(indices),
            replay_weights=[1./3]*3,replay_is_not_extra_independent_G=True,source_loader_calls=loader_calls,
            prepared_sources_resident_at_once=1,preclip_norm=preclip,direction=direction,trust_initial_fraction=alpha,
            full_bounded_probe=full,zero_step=zero,trials=trials,chosen_fraction=chosen,
            per_step_old2098_hard_veto=False,original_denominators_by_source=[{k:r[k] for k in
                ('source','original_G','original_G_by_group','original_E16_records','original_local_points')} for r in baselines.values()])
        restore()
        if chosen is None or not commit:
            restore(gradients=True)
            return dict(**evidence,accepted=chosen is not None,committed_optimizer_steps=0,native_proposal_step_calls=0,
                normal_six_trial_rejection=chosen is None,stop_branch=False,own_parameters_optimizer_RNG_restored=True)
        native_calls=1;optimizer.step()
        native=flat([p.double()-old.double() for p,old in zip(parameters,before)])
        correction,native_direction=bounded_direction(native,raw,blocks)
        trial_submit(parameters,before,correction,chosen);final=evaluate()
        evidence['source_loader_calls']=loader_calls
        if not final['pass_all']:
            restore(gradients=True)
            return dict(**evidence,accepted=False,committed_optimizer_steps=0,native_proposal_step_calls=1,
                normal_native_recheck_rejection=True,stop_branch=False,final_native_guard=final,own_parameters_optimizer_RNG_restored=True)
        model.verify_frozen()
        return dict(**evidence,accepted=True,committed_optimizer_steps=1,native_proposal_step_calls=1,
            final_actual_guard=final,native_direction=native_direction,
            actual_minus_analytic_maximum=float((native-proposal).abs().max()))
    except BaseException as error:
        restore(gradients=True);error.native_proposal_step_calls=native_calls
        error.transaction_parameters_optimizer_RNG_restored=True
        raise
