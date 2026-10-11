"""Projected actual AdamW proposals; old moments retained by parameter name."""
import itertools
import math
import numpy as np
import torch
from weld_pipeline.legacy_incremental_corrective_training_loop_425 import tensor
from weld_pipeline.legacy_incremental_emission_training_loop_457 import state_digest, norm, group_shared_gradient_evidence
from weld_pipeline.legacy_scale_count_loss_486 import event_loss, attribute_loss
from weld_pipeline.contract_field_acceptance_297 import FIELDS
from weld_pipeline.legacy_frozen_incremental_model_392 import tensor_state_digest


def migrate_optimizer_from325(model, recovery, original_migration):
    named = dict(model.named_parameters())
    saved = recovery["optimizer_state"]
    assert len(saved["param_groups"]) == 6
    names_by_group = original_migration["parameter_names_by_new_group"]
    reconstructed, bindings = [], []
    for group in saved["param_groups"]:
        phase = group["phase_name"]
        names = names_by_group[phase]
        assert len(names) == len(group["params"])
        parameters = [named[name] for name in names]
        options = {key: value for key, value in group.items() if key != "params"}
        reconstructed.append(dict(options, params=parameters))
        bindings += [(name, parameter, key) for name, parameter, key in zip(names, parameters, group["params"])]
    assert len({name for name, p, key in bindings}) == len(bindings)
    for name, parameter, key in bindings:
        if key in saved["state"]:
            assert tuple(saved["state"][key]["exp_avg"].shape) == tuple(parameter.shape), name
            assert tuple(saved["state"][key]["exp_avg_sq"].shape) == tuple(parameter.shape), name
    old = torch.optim.AdamW(reconstructed)
    old.load_state_dict(saved)
    before = {name: state_digest(old.state[p]) for name, p, key in bindings if p in old.state}
    assert len(before) == len(saved["state"]) == 50
    for name, p, key in bindings:
        if p in old.state:
            assert state_digest(old.state[p]) == state_digest(saved["state"][key]), name
    common = {key: value for key, value in old.param_groups[0].items() if key not in ("params", "lr", "phase_name")}
    numeric, states = [], []
    for module in (model.attributes, model.path_attributes):
        numeric += list(module.shared.parameters()) + list(module.scalars.parameters()) + list(module.normals.parameters()) + list(module.metric_projection.parameters())
        states += list(module.states.parameters())
    groups = [
        dict(common, params=list(model.events.shared.parameters()), lr=3e-5, phase_name="event_shared"),
        dict(common, params=list(model.events.presence.parameters()), lr=1e-4, phase_name="presence"),
        dict(common, params=list(model.events.count.parameters()) + list(model.events.gap.parameters()) + list(model.events.direction.parameters()) + list(model.events.closed.parameters()), lr=1e-4, phase_name="count_gap_direction_closed"),
        dict(common, params=list(model.events.positions.parameters()), lr=3e-4, phase_name="dimensionless_positions"),
        dict(common, params=numeric, lr=3e-4, phase_name="attribute_shared_numeric_normal_metric"),
        dict(common, params=states, lr=1e-4, phase_name="attribute_states"),
        dict({key: value for key, value in old.param_groups[-1].items() if key != "params"}, params=list(model.horizontal_residual.parameters()), phase_name="frozen_pointwise_H_residual"),
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    optimizer.state.update(old.state)
    assert {id(p) for group in groups for p in group["params"]} == {id(p) for p in model.incremental_parameters()}
    assert len(optimizer.state) == 50
    after = {name: state_digest(optimizer.state[p]) for name, p, key in bindings if p in optimizer.state}
    assert after == before
    for parameter in model.horizontal_residual.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    identity_names = {id(p): name for name, p in named.items()}
    new = [name for name, p in named.items() if id(p) in {id(v) for group in groups for v in group["params"]}
           and name not in {n for n, p, key in bindings}]
    assert sorted(new) == ["attributes.metric_projection.weight", "path_attributes.metric_projection.weight"]
    assert all(named[name] not in optimizer.state for name in new)
    return optimizer, dict(compatible_state_entries=50, old_names_shapes_moments_step_exact=True,
        optimizer_state_by_parameter_name_sha256=after,
        saved_optimizer_id_by_parameter_name={name: key for name, p, key in bindings},
        new_zero_initialized_parameter_names=new,
        new_parameter_state_entries_before_first_step=0,
        parameter_names_by_new_group={group["phase_name"]: [identity_names[id(p)] for p in group["params"]] for group in groups},
        frozen_H_parameter_sha256=tensor_state_digest(model.horizontal_residual),
        frozen_H_optimizer_state_sha256={name: state_digest(optimizer.state[p]) for name, p in named.items() if name.startswith("horizontal_residual.") and p in optimizer.state},
        actual_saved_source_layout="325 six groups from verified457 migration; not260 three-group layout",
        projected_proposal_is_modified_AdamW_not_equivalent_old_optimization=True)


def flatten(values):
    return torch.cat([value.detach().double().reshape(-1) for value in values])


def difference_vector(parameters, before):
    return torch.cat([(parameter.detach().double() - old.double()).reshape(-1)
                      for parameter, old in zip(parameters, before)])


def project_update(proposal, gradients, bounds=None):
    """Closest Euclidean vector under at most two normalized halfspaces."""
    rows = []
    names = []
    for name, gradient in gradients.items():
        length = float(torch.linalg.vector_norm(gradient))
        if length == 0.:
            continue
        assert math.isfinite(length)
        rows.append(gradient / length)
        names.append(name)
    if not rows:
        return proposal.clone(), dict(active_constraints=[], gradient_groups=[], before_dot={}, after_dot={}, projection_distance=0.)
    assert len(rows) <= 2
    matrix = torch.stack(rows)
    gram = (matrix @ matrix.T).cpu().numpy()
    dots = (matrix @ proposal).cpu().numpy()
    boundary = np.zeros(len(rows)) if bounds is None else np.asarray(bounds, np.float64)
    assert boundary.shape == dots.shape and np.isfinite(gram).all() and np.isfinite(dots).all()
    candidates = []
    for count in range(len(rows) + 1):
        for active in itertools.combinations(range(len(rows)), count):
            if not active:
                candidate = proposal.clone()
                multipliers = np.empty(0)
            else:
                chosen = list(active)
                subsystem = gram[np.ix_(chosen, chosen)]
                multipliers = np.linalg.lstsq(subsystem, dots[chosen] - boundary[chosen], rcond=1e-13)[0]
                if (multipliers < -1e-12).any():
                    continue
                if not np.allclose(subsystem @ multipliers, dots[chosen] - boundary[chosen], rtol=1e-10, atol=1e-12):
                    continue
                candidate = proposal - proposal.new_tensor(multipliers) @ matrix[chosen]
            residual = (matrix @ candidate).cpu().numpy() - boundary
            if (residual > 1e-12).any():
                continue
            distance = float((candidate - proposal).square().sum())
            candidates.append((distance, candidate, list(active)))
    if not candidates:
        raise ArithmeticError("No feasible finite active-set projection under registered constraints")
    distance, result, active = min(candidates, key=lambda item: item[0])
    final = (matrix @ result).cpu().numpy()
    return result, dict(active_constraints=[names[i] for i in active], gradient_groups=names,
        before_dot={name: float(value) for name, value in zip(names, dots)},
        after_dot={name: float(value) for name, value in zip(names, final)},
        projection_distance=math.sqrt(distance), normalized_constraints_matrix=matrix)


@torch.no_grad()
def commit_projected_update(parameters, before, proposal, gradients):
    """Float64 nearest proposal plus bounded float32 feasibility correction.

Roundoff can turn a boundary constraint positive after casting. Up to four
inward corrections are allowed; if infeasible, submit exact zero and record it.
AdamW moments/step remain those computed for the original proposal.
"""
    projected, info = project_update(proposal, gradients)
    matrix = info.pop("normalized_constraints_matrix", None)
    corrections, used_zero = 0, False
    def submit(update):
        start = 0
        for parameter, old in zip(parameters, before):
            end = start + parameter.numel()
            parameter.copy_((old.double() + update[start:end].reshape(old.shape)).to(parameter.dtype))
            start = end
        assert start == len(update)
        return difference_vector(parameters, before)
    actual = submit(projected)
    if matrix is not None:
        for attempt in range(4):
            dots = matrix @ actual
            if bool((dots <= 0.).all()):
                break
            margin = max(1e-12, 2. * float(dots.max()))
            try:
                projected, correction_info = project_update(proposal, gradients, [-margin] * len(matrix))
            except ArithmeticError:
                break
            actual = submit(projected)
            corrections += 1
        if not bool(((matrix @ actual) <= 0.).all()):
            used_zero = True
            for parameter, old in zip(parameters, before):
                parameter.copy_(old)
            actual = difference_vector(parameters, before)
        assert bool(((matrix @ actual) <= 0.).all()), "Actual submitted update violated keep gradients"
        info["actual_submitted_dot"] = {name: float(value) for name, value in zip(info["gradient_groups"], (matrix @ actual).cpu().numpy())}
    else:
        info["actual_submitted_dot"] = {}
    initial_norm, final_norm = float(torch.linalg.vector_norm(proposal)), float(torch.linalg.vector_norm(actual))
    info.update(proposal_norm=initial_norm, submitted_norm=final_norm,
                submitted_over_proposal_norm=final_norm / initial_norm if initial_norm else None,
                bounded_float32_inward_corrections=corrections, quantization_zero_safeguard_used=used_zero,
                nearly_zero_update=bool(initial_norm and final_norm / initial_norm < 1e-3),
                actual_submitted_all_nonzero_keep_constraints_satisfied=True,
                AdamW_computed_moments_and_step_not_reverted=True)
    return info


def analytic_adamw_proposal(optimizer, parameters):
    """Read-only one-step proposal for controls, in saved native tensor dtype."""
    group_by_id = {id(p): group for group in optimizer.param_groups for p in group["params"]}
    result = []
    for parameter in parameters:
        if parameter.grad is None:
            result.append(torch.zeros_like(parameter))
            continue
        group = group_by_id[id(parameter)]
        assert not group["amsgrad"] and not group["maximize"] and not group["capturable"]
        state = optimizer.state.get(parameter, {})
        step = float(state["step"]) + 1. if state else 1.
        beta1, beta2 = group["betas"]
        gradient = parameter.grad.detach()
        old_m = state["exp_avg"] if state else torch.zeros_like(parameter)
        old_v = state["exp_avg_sq"] if state else torch.zeros_like(parameter)
        moment = old_m * beta1 + gradient * (1. - beta1)
        variance = old_v * beta2 + gradient.square() * (1. - beta2)
        denominator = variance.sqrt() / math.sqrt(1. - beta2 ** step) + group["eps"]
        candidate = parameter.detach() * (1. - group["lr"] * group["weight_decay"])
        candidate = candidate - group["lr"] / (1. - beta1 ** step) * moment / denominator
        result.append(candidate.double() - parameter.detach().double())
    return flatten(result)


def attribute_backward(module, features, context, target, available, device, weight=.25, chunk=128):
    indices = np.flatnonzero(available)
    if not len(indices):
        return 0., 0, {}
    totals = {name: np.asarray(target[name][indices]).sum(axis=0) for name in ("state_known", "scalar_known", "normal_known")}
    diagnostic = {name: dict(known_numeric=0, absolute_error_sum=0., passed_numeric_3mm=0,
        prediction_min=None, prediction_max=None, target_min=None, target_max=None) for name in FIELDS}
    loss_sum = 0.
    for start in range(0, len(indices), chunk):
        chosen = indices[start:start + chunk]
        wanted = {name: tensor(values[chosen], device) for name, values in target.items()}
        output = module(tensor(features[chosen], device), tensor(context[chosen], device))
        loss = attribute_loss(output, wanted, totals) * weight
        regularization = output["numeric_increment_dimensionless"].square().mean() * (.001 * weight * len(chosen) / len(indices))
        (loss + regularization).backward()
        loss_sum += float((loss + regularization).detach().cpu())
        values = output["scalar_values"].detach().cpu().numpy()
        for j, name in enumerate(FIELDS):
            if name in ("has_eave", "side_plate_split") or name.startswith("normal_"):
                continue
            known = target["scalar_known"][chosen, j]
            if not known.any():
                continue
            actual, expected = values[known, j], target["scalar_values"][chosen, j][known]
            errors = np.abs(actual - expected)
            row = diagnostic[name]
            row["known_numeric"] += len(actual)
            row["absolute_error_sum"] += float(errors.sum())
            row["passed_numeric_3mm"] += int((errors <= (1. if name.startswith("thickness_") else 3.)).sum())
            for label, array in (("prediction", actual), ("target", expected)):
                lo, hi = float(array.min()), float(array.max())
                row[label + "_min"] = lo if row[label + "_min"] is None else min(lo, row[label + "_min"])
                row[label + "_max"] = hi if row[label + "_max"] is None else max(hi, row[label + "_max"])
    return loss_sum, len(indices), diagnostic


def source_update(model, optimizer, source, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    frozen_H = tensor_state_digest(model.horizontal_residual)
    H_state = {id(p): state_digest(optimizer.state[p]) for p in model.horizontal_residual.parameters() if p in optimizer.state}
    reference_state = model.reference_digests()
    base = {name: tensor(value, device) for name, value in source["base"].items()}
    output = model.events(tensor(source["features"], device), base)
    loss, detail = event_loss(output, source["parents"], base, tensor(source["background_known"], device),
        tensor(source["group_ids"], device), tensor(source["hard_negative_weights"], device),
        source["original_G"], source["original_G_by_group"])
    event_parameters = [p for p in model.events.parameters() if p.requires_grad]
    keep_gradients = {}
    for group, keep in detail["protection_losses"].items():
        gradients = torch.autograd.grad(keep, event_parameters, retain_graph=True, allow_unused=True)
        keep_gradients[str(group)] = flatten([torch.zeros_like(p) if gradient is None else gradient for p, gradient in zip(event_parameters, gradients)])
    conflict = group_shared_gradient_evidence(model, detail)
    loss.backward()
    ordinary, path_rows, path_diagnostic = attribute_backward(model.path_attributes, source["path_features"],
        source["path_metric_context"], source["path_targets"], source["path_available"], device)
    local, local_rows, local_diagnostic = attribute_backward(model.attributes, source["local_features"],
        source["local_metric_context"], source["local_targets"], source["local_available"], device)
    parameters = model.incremental_parameters()
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)
    before = {group["phase_name"]: [p.detach().clone() for p in group["params"]] for group in optimizer.param_groups}
    event_before = [p.detach().clone() for p in event_parameters]
    group_norms = {group["phase_name"]: norm(group["params"]) for group in optimizer.param_groups}
    preclip = torch.nn.utils.clip_grad_norm_(parameters, 10.)
    assert bool(torch.isfinite(preclip))
    optimizer.step()  # Caller counts this actual proposal even if a later check fails.
    proposal = difference_vector(event_parameters, event_before)
    projection = commit_projected_update(event_parameters, event_before, proposal, keep_gradients)
    changes = {group["phase_name"]: max(float((p.detach() - old).abs().max().cpu()) for p, old in zip(group["params"], before[group["phase_name"]])) for group in optimizer.param_groups}
    assert changes["frozen_pointwise_H_residual"] == 0.
    assert tensor_state_digest(model.horizontal_residual) == frozen_H
    assert {id(p): state_digest(optimizer.state[p]) for p in model.horizontal_residual.parameters() if p in optimizer.state} == H_state
    assert model.reference_digests() == reference_state
    model.verify_frozen()
    return dict(losses=dict(event_total=float(loss.detach().cpu()), whole_path_attribute_weighted=ordinary,
        local_attribute_weighted=local, preclip_gradient_norm=float(preclip.cpu())),
        training_correspondence_failure_counts=detail["training_correspondence_failure_counts"],
        actual_projected_update=projection, group_presence_shared_gradient_evidence=conflict,
        parameter_group_preclip_gradient_norm=group_norms, parameter_group_maximum_absolute_update=changes,
        combined_clip10_triggered=bool(float(preclip) > 10.),
        whole_numeric_TRAIN_diagnostic=path_diagnostic, local_numeric_TRAIN_diagnostic=local_diagnostic,
        available_whole_path_rows=path_rows, available_local_reference_rows=local_rows,
        original_G=source["original_G"], original_E16_records=source["original_E16_records"],
        original_local_points=source["original_local_points"], representation_miss_children=source["representation_miss_children"],
        protected130_V_TRAIN_reference_count=source["current_correct_V_ids_used_only_in_TRAIN"],
        E3_new_H_TRAIN_reference_count=source["E3_new_H_correct_id_count_training_only"],
        geometry_supervised_children=detail["geometry_supervised_children"],
        event_NN_forwards=1, additional_frozen325_event_forwards=1, horizontal_shadow_NN_forwards=0,
        optimizer_steps=1, current_raw_XYZ_bounds_are_features_not_truth=True,
        approved_whole_truth_fields_created=0, formal_E_records_created=0,
        TRAIN_diagnostic_not_fixed_free_test_score=True, all_original_denominators_retained=True)
