"""State-preserving Armijo transactions without the old2098 hard veto."""
from copy import deepcopy
import math
import torch
from weld_pipeline.descent_guarded_transactions_642 import flat,prepare_blocks,correct_direction
from weld_pipeline.legal_allocation_finite_update_521 import rng_state,restore_rng,trial_submit
from weld_pipeline.legacy_scale_count_training_loop_488 import analytic_adamw_proposal
from weld_pipeline.unified_parent_set_loss_691 import event_loss,VERSION


def typed_event_transaction(model,optimizer,source,branch,device='cuda',*,commit=True):
    group = {'H':0,'V':1}[branch]
    parameters = list(getattr(model.events,branch).parameters())
    supervision = sum(len(rows) for i,rows in enumerate(source['parents']) if int(source['group_ids'][i])==group)
    supervision += int(((source['group_ids']==group)&source['background_known']).sum())
    if not supervision:
        return dict(accepted=False,no_effective_supervision=True,committed_optimizer_steps=0,native_proposal_step_calls=0)
    original_rng = rng_state(); before = [p.detach().clone() for p in parameters]
    saved_state = deepcopy(optimizer.state_dict()); native_calls = 0
    base = {k:torch.as_tensor(v,device=device) for k,v in source['base'].items()}
    features = torch.as_tensor(source['features'],device=device)
    args = (source['parents'],base,torch.as_tensor(source['background_known'],device=device),
        torch.as_tensor(source['group_ids'],device=device),torch.as_tensor(source['hard_negative_weights'],device=device),
        source['original_G'],source['original_G_by_group'])
    def restore():
        with torch.no_grad():
            for parameter,value in zip(parameters,before):
                parameter.copy_(value)
        optimizer.load_state_dict(saved_state); restore_rng(original_rng)
    try:
        model.eval(); optimizer.zero_grad(set_to_none=True)
        output = model.events(features,base)
        _,report = event_loss(output,*args,events=model.events)
        plan = report['transaction_plan_TRAIN_only']; loss = report['loss_by_group'][group]
        initial = float(loss.detach())
        gradients = torch.autograd.grad(loss,parameters,allow_unused=True)
        gradient = flat([torch.zeros_like(p) if g is None else g for p,g in zip(parameters,gradients)])
        for parameter,value in zip(parameters,gradients):
            parameter.grad = value
        preclip = float(torch.nn.utils.clip_grad_norm_(parameters,5.)); assert math.isfinite(preclip)
        blocks = prepare_blocks(optimizer,parameters,gradient)
        proposal = analytic_adamw_proposal(optimizer,parameters)
        corrected,direction = correct_direction(proposal,gradient,blocks)
        @torch.no_grad()
        def check():
            delta = flat([p.double()-old.double() for p,old in zip(parameters,before)])
            dot = float(gradient@delta)
            nonzero = bool((delta!=0).any())
            output = model.events(features,base)
            _,fixed = event_loss(output,*args,events=model.events,transaction_plan=plan)
            _,dynamic = event_loss(output,*args,events=model.events)
            fv,dv = float(fixed['loss_by_group'][group]),float(dynamic['loss_by_group'][group])
            # Floating summation tolerance only; never an acceptance geometry tolerance.
            numeric_tolerance = 1e-9*max(1.,abs(fv),abs(dv))
            assert dv <= fv+numeric_tolerance, 'Unified assignment optimum exceeds the same fixed-pair objective'
            return dict(pass_all=nonzero and dot<0. and math.isfinite(fv) and math.isfinite(dv)
                and fv<=initial+1e-4*dot and dv<initial,
                fixed_before=initial,fixed_after=fv,dynamic_before=initial,dynamic_after=dv,
                rematching_not_worse_than_fixed=True,armijo_bound=initial+1e-4*dot,
                g_dot_actual_delta=dot,actual_nonzero=nonzero,
                fixed_components={k:float(v) for k,v in fixed['group_components'][group].items()},
                dynamic_components={k:float(v) for k,v in dynamic['group_components'][group].items()},
                per_step_old2098_veto_applied=False)
        trials = []; chosen = None
        for fraction in (1.,.5,.25,.125):
            trial_submit(parameters,before,corrected,fraction)
            result = check(); trials.append(dict(fraction=fraction,**result))
            if result['pass_all']:
                chosen = fraction; break
        evidence = dict(branch=branch,version=VERSION,preclip_norm=preclip,direction=direction,
            trials=trials,chosen_fraction=chosen,typed_loss_before=initial,
            block_scales={name:1. for name in blocks},old2098_minimum_slack_scaling_applied=False,
            original_G=source['original_G'],reserved_train_anchor_count=0,
            old_query_anchors_ignored=report['old_query_anchors_ignored'])
        restore()
        if chosen is None or not commit:
            return dict(**evidence,accepted=chosen is not None,committed_optimizer_steps=0,native_proposal_step_calls=0,
                normal_four_trial_rejection=chosen is None,own_parameters_optimizer_RNG_restored=True)
        optimizer.step(); native_calls = 1
        native = flat([p.double()-old.double() for p,old in zip(parameters,before)])
        native_corrected,native_direction = correct_direction(native,gradient,blocks)
        trial_submit(parameters,before,native_corrected,chosen)
        final = check()
        if not final['pass_all']:
            restore()
            return dict(**evidence,accepted=False,committed_optimizer_steps=0,native_proposal_step_calls=1,
                final_native_guard=final,stop_branch=True,own_parameters_optimizer_RNG_restored=True)
        model.verify_frozen()
        return dict(**evidence,accepted=True,committed_optimizer_steps=1,native_proposal_step_calls=1,
            final_actual_guard=final,native_direction=native_direction,
            actual_minus_analytic_maximum=float((native-proposal).abs().max()))
    except BaseException as error:
        restore()
        error.native_proposal_step_calls = native_calls
        error.transaction_parameters_optimizer_RNG_restored = True
        raise
