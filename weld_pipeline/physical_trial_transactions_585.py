"""TRAIN-only independent H/V/attribute proposals with physical step selection.

The selector bounds fixed-topology motion. Actual all2098 world and topology
checks still decide acceptance, including a permitted change to the true count.
Four finite trials; no extra epochs, zero submissions, or removed identities.
"""
from copy import deepcopy
import math

import torch

from weld_pipeline.legal_allocation_finite_update_521 import (
    rng_state, restore_rng, trial_submit,
)
from weld_pipeline.legacy_scale_count_training_loop_488 import analytic_adamw_proposal
from weld_pipeline.legacy_legal_allocation_model_518 import margin_logits_for_counts

VERSION = "U000_PHYSICAL_SCALE_INDEPENDENT_TRANSACTIONS_ONE65_v1"


def make_independent_optimizers(model):
    result, bindings = {}, {}
    for branch in ("H", "V"):
        head = getattr(model.events, branch)
        groups = [dict(params=head.classification_parameters(), lr=1e-4,
                       phase_name=branch + "_classification"),
                  dict(params=list(head.allocation.parameters()), lr=3e-4,
                       phase_name=branch + "_allocation")]
        result[branch] = torch.optim.AdamW(groups, weight_decay=1e-4)
        bindings[branch] = ["events." + branch + "." + name
                            for name, _ in head.named_parameters()]
    attrs = [("attributes." + name, p) for name, p in model.attributes.named_parameters()]
    attrs += [("path_attributes." + name, p) for name, p in model.path_attributes.named_parameters()]
    result["attributes"] = torch.optim.AdamW([p for _, p in attrs], lr=3e-4, weight_decay=1e-4)
    bindings["attributes"] = [name for name, _ in attrs]
    assert len(set(sum(bindings.values(), []))) == sum(map(len, bindings.values()))
    assert not any(opt.state for opt in result.values())
    return result, bindings


def bound_from_slack(slack, change):
    """Exact current slack; no invented epsilon at an adverse boundary."""
    slack, change = float(slack), float(change)
    if not math.isfinite(slack) or not math.isfinite(change) or slack < 0.:
        return 0.
    if change >= 0.:
        return math.inf
    return slack / -change


def active_allocation_components(output, candidate, query, n):
    mask = torch.zeros(output["allocation_increment_logits"].shape[-1],
                       dtype=torch.bool, device=output["presence_logits"].device)
    mask[1:2*n:2] = True
    if n > 1:
        mask[2:2*n:2] = output["gap_logits"][candidate, query, :n-1] >= 0.
    margins = margin_logits_for_counts(output, output["count_logits"].argmax(-1))[candidate, query] >= 0.
    mask[0], mask[2*n] = margins[0], margins[1]
    return mask


@torch.no_grad()
def physical_start_fraction(events, pool, parameters, before, proposal, device="cuda"):
    current = []
    for batch in pool.batches:
        prior = {k: v.to(device) for k, v in batch["prior"].items()}
        h = batch["h"].to(device)
        out = events.forward_cached(h, prior)
        n = out["count_logits"].argmax(-1)
        allocation = events.allocation_for_requested_counts(out, n)
        current.append((h, prior, out, n, allocation, batch["records"]))
    limits = []
    geometry_limits, class_limits, current_max = [], [], 0.
    trial_submit(parameters, before, proposal, 1.)
    try:
        for h, prior, old, counts, old_allocation, rows in current:
            proposed = events.forward_cached(h, prior)
            new_allocation = events.allocation_for_requested_counts(proposed, counts)
            old_margins = margin_logits_for_counts(old, counts)
            new_margins = margin_logits_for_counts(proposed, counts)
            for row in rows:
                ci, qi = row["candidate"], row["query"]
                n = int(counts[ci, qi])
                assert n in (row["old_n"], row["target_n"]) and n > 0
                slots = row["old_slots"] if n == row["old_n"] else row["target_slots"]
                nodes = old["positions_mm"][ci, qi, slots]
                world = nodes[..., None] * nodes.new_tensor(row["direction"])[None, None] + nodes.new_tensor(row["origin"])[None, None]
                error = float(torch.linalg.vector_norm(world - nodes.new_tensor(row["world_truth"]), dim=-1).max())
                assert error <= 3., "Current protected object already outside original gate"
                budget = min(.25, (3. - error) / 2.)
                active = active_allocation_components(old, ci, qi, n)
                dz = (new_allocation[ci, qi] - old_allocation[ci, qi]).double()[active]
                # Softmax is invariant to a common active-component offset.
                # Subtracting its midrange tightens the SAME fixed-node bound.
                centered = dz - (dz.max() + dz.min()) / 2.
                norm = float(centered.abs().max())
                span = float(old["allocation_domain_mm"][ci, 1] - old["allocation_domain_mm"][ci, 0])
                geom = math.inf if norm == 0. else 2. * budget / (span * norm)
                geometry_limits.append(geom)
                current_max = max(current_max, error)
                if math.isfinite(geom):
                    limits.append((geom, "geometry", row["source"], qi, budget, norm))
                for name in ("presence", "closed", "direction"):
                    if name == "direction" and not row["direction_semantics_confirmed"]:
                        continue
                    a = float(old[name + "_logits"][ci, qi])
                    z = float(proposed[name + "_logits"][ci, qi])
                    sign = 1. if a >= 0. else -1.
                    limit = bound_from_slack(sign * a, sign * (z - a))
                    class_limits.append(limit)
                    if math.isfinite(limit):
                        limits.append((limit, name, row["source"], qi, sign*a, z-a))
                for name, aa, zz in (
                        ("gap", old["gap_logits"][ci, qi, :n-1], proposed["gap_logits"][ci, qi, :n-1]),
                        ("margin", old_margins[ci, qi], new_margins[ci, qi])):
                    for a, z in zip(aa.tolist(), zz.tolist()):
                        sign = 1. if a >= 0. else -1.
                        limit = bound_from_slack(sign*a, sign*(z-a))
                        class_limits.append(limit)
                        if math.isfinite(limit):
                            limits.append((limit, name, row["source"], qi, sign*a, z-a))
                aa, zz = old["count_logits"][ci, qi], proposed["count_logits"][ci, qi]
                for competitor in range(len(aa)):
                    # The original hard gate explicitly permits old OR true N.
                    # A true-N switch has no fixed-topology certificate: it MUST
                    # pass the actual all2098 world check at the trial fraction.
                    if competitor in (row["old_n"], row["target_n"]):
                        continue
                    slack = float(aa[n] - aa[competitor])
                    change = float((zz[n] - zz[competitor]) - (aa[n] - aa[competitor]))
                    limit = bound_from_slack(slack, change)
                    class_limits.append(limit)
                    if math.isfinite(limit):
                        limits.append((limit, "count_forbidden_class", row["source"], qi, slack, change))
    finally:
        for p, old in zip(parameters, before):
            p.copy_(old)
    finite = sorted(limits, key=lambda v: v[0])
    alpha0 = min(1., finite[0][0]) if finite else 1.
    assert math.isfinite(alpha0) and alpha0 >= 0.
    return dict(alpha0=alpha0, fractions=[alpha0/(2**k) for k in range(4)],
                tightest_limits=[dict(bound=r[0], reason=r[1], source_TRAIN_only=r[2], query=r[3],
                                      current_budget_or_slack=r[4], proposal_change=r[5]) for r in finite[:8]],
                fixed_topology_softmax_bound_only_selects_step=True,
                permitted_true_count_switch_requires_actual_full_world_guard=True,
                common_active_logit_offset_removed_by_softmax_invariance=True,
                protected_child_identities=pool.records,
                current_maximum_world_error_mm=current_max,
                geometry_start_bound=(min(geometry_limits) if geometry_limits and math.isfinite(min(geometry_limits)) else None),
                classification_start_bound=(min(class_limits) if class_limits and math.isfinite(min(class_limits)) else None))


def finite_independent_transaction(optimizer, parameters, check, *, selector=None,
                                   initial_rng=None, commit=True):
    """Own parameter/AdamW/RNG snapshot; never restores another transaction."""
    initial_rng = initial_rng or rng_state()
    before = [p.detach().clone() for p in parameters]
    state = deepcopy(optimizer.state_dict())
    proposal = analytic_adamw_proposal(optimizer, parameters)
    assert bool(torch.isfinite(proposal).all())

    def restore():
        with torch.no_grad():
            for p, old in zip(parameters, before):
                p.copy_(old)
        optimizer.load_state_dict(state)
        restore_rng(initial_rng)

    if float(torch.linalg.vector_norm(proposal)) == 0.:
        restore()
        return dict(accepted=False, zero_proposal_not_counted=True,
                    committed_optimizer_steps=0, native_proposal_step_calls=0)
    try:
        scaling = selector(before, proposal) if selector else dict(alpha0=1., fractions=[1., .5, .25, .125])
    except (AssertionError, ValueError, ArithmeticError, RuntimeError) as error:
        restore()
        return dict(accepted=False, committed_optimizer_steps=0, native_proposal_step_calls=0,
                    explicit_selector_failure=str(error), stop_branch=True,
                    own_parameters_optimizer_RNG_restored=True)
    if scaling["alpha0"] <= 0.:
        restore()
        return dict(accepted=False, committed_optimizer_steps=0, native_proposal_step_calls=0,
                    no_positive_physical_or_classification_slack=True,
                    scaling=scaling, stop_branch=True, own_parameters_optimizer_RNG_restored=True)
    trials, chosen = [], None
    for fraction in scaling["fractions"]:
        trial_submit(parameters, before, proposal, fraction)
        actual_nonzero = any(not torch.equal(p.detach(), old) for p, old in zip(parameters, before))
        if actual_nonzero:
            try:
                result = check()
            except (AssertionError, ValueError, ArithmeticError, RuntimeError) as error:
                result = dict(pass_all=False, explicit_numerical_failure=str(error))
        else:
            result = dict(pass_all=False, actual_parameter_change_zero=True)
        trials.append(dict(fraction=fraction, result=result))
        if result["pass_all"]:
            chosen = fraction
            break
    with torch.no_grad():
        for p, old in zip(parameters, before):
            p.copy_(old)
    if chosen is None or not commit:
        restore()
        return dict(accepted=chosen is not None, committed_optimizer_steps=0, native_proposal_step_calls=0,
                    scaling=scaling, trials=trials, chosen_fraction=chosen,
                    rejected_all_four=chosen is None, stop_branch=chosen is None,
                    own_parameters_optimizer_RNG_restored=True, copied_control_only=not commit)
    optimizer.step()
    native = torch.cat([(p.detach().double()-old.double()).reshape(-1) for p, old in zip(parameters,before)])
    trial_submit(parameters, before, native, chosen)
    try:
        final = check()
    except (AssertionError, ValueError, ArithmeticError, RuntimeError) as error:
        final = dict(pass_all=False, explicit_numerical_failure=str(error))
    actual_nonzero = any(not torch.equal(p.detach(), old) for p, old in zip(parameters,before))
    if not final["pass_all"] or not actual_nonzero:
        restore()
        return dict(accepted=False, committed_optimizer_steps=0, native_proposal_step_calls=1,
                    scaling=scaling, trials=trials, final_native_guard=final,
                    actual_parameter_change_zero=not actual_nonzero, stop_branch=True,
                    own_parameters_optimizer_RNG_restored=True)
    return dict(accepted=True, committed_optimizer_steps=1, native_proposal_step_calls=1,
                scaling=scaling, trials=trials, final_actual_guard=final, chosen_fraction=chosen,
                submitted_norm=float(torch.linalg.vector_norm(native)*chosen),
                actual_minus_analytic_maximum=float((native-proposal).abs().max()),
                constrained_fraction_AdamW_moments_from_one_native_proposal=True,
                no_zero_submission=True)


def typed_event_transaction(model, optimizer, source, pool, branch, device="cuda", *, commit=True):
    from weld_pipeline.count_conditioned_event_loss_581 import event_loss
    group = {"H": 0, "V": 1}[branch]
    parameters = list(getattr(model.events,branch).parameters())
    supervision = (sum(len(rows) for ci,rows in enumerate(source["parents"]) if int(source["group_ids"][ci])==group)
        + int(((source["group_ids"]==group) & source["background_known"]).sum()))
    if not supervision:
        return dict(accepted=False, committed_optimizer_steps=0, native_proposal_step_calls=0,
                    no_effective_supervision=True, branch=branch)
    initial_rng = rng_state()
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    base = {k:torch.as_tensor(v,device=device) for k,v in source["base"].items()}

    def objective():
        out = model.events(torch.as_tensor(source["features"],device=device),base)
        _, report = event_loss(out,source["parents"],base,
            torch.as_tensor(source["background_known"],device=device),
            torch.as_tensor(source["group_ids"],device=device),
            torch.as_tensor(source["hard_negative_weights"],device=device),
            source["original_G"],source["original_G_by_group"],events=model.events)
        return report["loss_by_group"][group], report

    loss, report = objective()
    value = float(loss.detach())
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    for p, grad in zip(parameters,gradients):
        p.grad = grad
    preclip = float(torch.nn.utils.clip_grad_norm_(parameters,5.))
    assert math.isfinite(preclip)

    @torch.no_grad()
    def check():
        guard = pool.check(model.events,device)
        if not guard["pass_all"]:
            return dict(pass_all=False, protection=guard)
        now, _ = objective()
        after = float(now)
        return dict(pass_all=math.isfinite(after) and after<value,
                    typed_loss_before=value,typed_loss_after=after,protection=guard)

    result = finite_independent_transaction(optimizer,parameters,check,
        selector=lambda before,proposal:physical_start_fraction(model.events,pool,parameters,before,proposal,device),
        initial_rng=initial_rng,commit=commit)
    result.update(branch=branch,typed_loss_before=value,preclip_norm=preclip,
                  effective_supervision_records=supervision,original_G=source["original_G"],
                  training_correspondence_failure_counts=report["training_correspondence_failure_counts"])
    model.verify_frozen()
    return result
