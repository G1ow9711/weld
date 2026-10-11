"""TRAIN-only state-preserving descent guard and independent block scales."""
from copy import deepcopy
import math
import torch
from weld_pipeline.legal_allocation_finite_update_521 import rng_state,restore_rng,trial_submit
from weld_pipeline.physical_trial_transactions_585 import physical_start_fraction
from weld_pipeline.legacy_scale_count_training_loop_488 import analytic_adamw_proposal
from weld_pipeline.fixed_transaction_event_loss_641 import event_loss as fixed_loss
from weld_pipeline.count_conditioned_event_loss_581 import event_loss as dynamic_loss

VERSION="STATE_PRESERVING_DESCENT_GUARD_SPLIT_BLOCK_FIXED_AND_DYNAMIC_ARMIJO_v1"

def flat(values):return torch.cat([v.detach().double().reshape(-1) for v in values])

class SigmoidPresenceGuard:
    """Reuse world/topology guard with the exact runtime presence predicate."""
    def __init__(self,events):self.events=events
    def forward_cached(self,*args):
        out=dict(self.events.forward_cached(*args));p=out["presence_logits"]
        out["presence_logits"]=torch.where(p.sigmoid()>=.5,p.clamp_min(0.),p)
        return out
    def allocation_for_requested_counts(self,*args):return self.events.allocation_for_requested_counts(*args)

def prepare_blocks(optimizer,parameters,raw_gradient):
    groups={id(p):group for group in optimizer.param_groups for p in group["params"]}
    blocks={};offset=0
    for parameter in parameters:
        group=groups[id(parameter)];name=group["phase_name"]
        row=blocks.setdefault(name,dict(indices=[],preconditioned_gradient=[]))
        n=parameter.numel();g=raw_gradient[offset:offset+n].reshape(parameter.shape)
        state=optimizer.state.get(parameter,{})
        if parameter.grad is None:
            pg=torch.zeros_like(g)
        else:
            beta2=group["betas"][1];step=float(state["step"])+1 if state else 1.
            variance=state["exp_avg_sq"] if state else torch.zeros_like(parameter)
            next_variance=variance*beta2+parameter.grad.square()*(1-beta2)
            denominator=next_variance.double().sqrt()/math.sqrt(1-beta2**step)+group["eps"]
            pg=g*group["lr"]/denominator
        row["indices"].extend(range(offset,offset+n));row["preconditioned_gradient"].append(pg.reshape(-1))
        offset+=n
    for row in blocks.values():
        row["indices"]=torch.tensor(row["indices"],dtype=torch.long,device=raw_gradient.device)
        row["preconditioned_gradient"]=torch.cat(row["preconditioned_gradient"])
    return blocks

def correct_direction(proposal,gradient,blocks):
    corrected=proposal.clone();facts={}
    for name,row in blocks.items():
        index=row["indices"];g=gradient[index];pg=row["preconditioned_gradient"]
        D=float(g@pg);original=float(g@proposal[index]);assert D>=0 and math.isfinite(D)
        coefficient=max(0.,(original+.1*D)/D) if D else 0.
        corrected[index]=proposal[index]-coefficient*pg if D else torch.zeros_like(pg)
        actual=float(g@corrected[index])
        assert actual<=-.1*D+max(1e-12,abs(D)*1e-10)
        facts[name]=dict(g_dot_proposal=original,D=D,coefficient=coefficient,g_dot_corrected=actual)
    return corrected,facts

def typed_event_transaction(model,optimizer,source,pool,branch,device="cuda",*,commit=True):
    group={"H":0,"V":1}[branch];head=getattr(model.events,branch);parameters=list(head.parameters())
    supervision=sum(len(rows) for i,rows in enumerate(source["parents"]) if int(source["group_ids"][i])==group)
    supervision+=int(((source["group_ids"]==group)&source["background_known"]).sum())
    if not supervision:return dict(accepted=False,no_effective_supervision=True,committed_optimizer_steps=0,native_proposal_step_calls=0)
    initial_rng=rng_state();before=[p.detach().clone() for p in parameters];old_state=deepcopy(optimizer.state_dict())
    native_calls=0
    def restore():
        with torch.no_grad():
            for p,old in zip(parameters,before):p.copy_(old)
        optimizer.load_state_dict(old_state);restore_rng(initial_rng)
    base={k:torch.as_tensor(v,device=device) for k,v in source["base"].items()}
    features=torch.as_tensor(source["features"],device=device)
    args=(source["parents"],base,torch.as_tensor(source["background_known"],device=device),
        torch.as_tensor(source["group_ids"],device=device),torch.as_tensor(source["hard_negative_weights"],device=device),
        source["original_G"],source["original_G_by_group"])
    guard_events=SigmoidPresenceGuard(model.events)
    try:
        model.eval();optimizer.zero_grad(set_to_none=True)
        output=model.events(features,base)
        _,report=fixed_loss(output,*args,events=model.events)
        plan=report["transaction_plan_TRAIN_only"];loss=report["loss_by_group"][group];value=float(loss.detach())
        with torch.no_grad():
            _,dynamic=dynamic_loss(model.events(features,base),*args,events=model.events)
            dynamic_before=float(dynamic["loss_by_group"][group])
        assert abs(value-dynamic_before)<=1e-10*max(1.,abs(value)),"Zero-step fixed/dynamic loss mismatch"
        gradients=torch.autograd.grad(loss,parameters,allow_unused=True)
        gradient=flat([torch.zeros_like(p) if g is None else g for p,g in zip(parameters,gradients)])
        for p,g in zip(parameters,gradients):p.grad=g
        preclip=float(torch.nn.utils.clip_grad_norm_(parameters,5.));assert math.isfinite(preclip)
        blocks=prepare_blocks(optimizer,parameters,gradient)
        proposal=analytic_adamw_proposal(optimizer,parameters)
        corrected,direction=correct_direction(proposal,gradient,blocks)
        scaling=physical_start_fraction(guard_events,pool,parameters,before,corrected,device)
        scales={}
        for name,row in blocks.items():
            bound=scaling["geometry_start_bound" if name.endswith("allocation") else "classification_start_bound"]
            scales[name]=min(1.,bound) if bound is not None else 1.
            assert scales[name]>=0. and math.isfinite(scales[name])
        def scaled(d):
            result=d.clone()
            for name,row in blocks.items():result[row["indices"]]*=scales[name]
            return result
        @torch.no_grad()
        def check():
            delta=flat([p.double()-old.double() for p,old in zip(parameters,before)])
            dot=float(gradient@delta);nonzero=bool((delta!=0).any())
            guard=pool.check(guard_events,device)
            if not guard["pass_all"] or not nonzero or dot>=0.:
                return dict(pass_all=False,protection=guard,actual_nonzero=nonzero,g_dot_actual_delta=dot)
            current=model.events(features,base)
            _,fixed=fixed_loss(current,*args,events=model.events,transaction_plan=plan)
            _,dynamic=dynamic_loss(current,*args,events=model.events)
            fv,dv=float(fixed["loss_by_group"][group]),float(dynamic["loss_by_group"][group])
            return dict(pass_all=math.isfinite(fv) and math.isfinite(dv) and fv<=value+1e-4*dot and dv<dynamic_before,
                fixed_before=value,fixed_after=fv,armijo_bound=value+1e-4*dot,dynamic_before=dynamic_before,dynamic_after=dv,
                g_dot_actual_delta=dot,actual_nonzero=nonzero,protection=guard)
        trials=[];chosen=None
        for fraction in (1.,.5,.25,.125):
            trial_submit(parameters,before,scaled(corrected),fraction)
            result=check();trials.append(dict(fraction=fraction,**result))
            if result["pass_all"]:chosen=fraction;break
        evidence=dict(branch=branch,version=VERSION,preclip_norm=preclip,direction=direction,block_scales=scales,
            physical_selector=scaling,trials=trials,typed_loss_before=value,chosen_fraction=chosen)
        restore()
        if chosen is None or not commit:
            return dict(**evidence,accepted=chosen is not None,committed_optimizer_steps=0,native_proposal_step_calls=0,
                normal_four_trial_rejection=chosen is None,own_parameters_optimizer_RNG_restored=True)
        optimizer.step();native_calls=1
        native=flat([p.double()-old.double() for p,old in zip(parameters,before)])
        native_corrected,native_direction=correct_direction(native,gradient,blocks)
        trial_submit(parameters,before,scaled(native_corrected),chosen)
        final=check()
        if not final["pass_all"]:
            restore()
            return dict(**evidence,accepted=False,committed_optimizer_steps=0,native_proposal_step_calls=1,
                final_native_guard=final,stop_branch=True,own_parameters_optimizer_RNG_restored=True)
        model.verify_frozen()
        return dict(**evidence,accepted=True,committed_optimizer_steps=1,native_proposal_step_calls=1,
            final_actual_guard=final,native_direction=native_direction,actual_minus_analytic_maximum=float((native-proposal).abs().max()))
    except BaseException as error:
        restore()
        error.native_proposal_step_calls=native_calls
        error.transaction_parameters_optimizer_RNG_restored=True
        raise
