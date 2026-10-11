"""Finite emission-guard updates; optimizer identity/state and frozen H retained."""
import hashlib
import math
import numpy as np
import torch
from weld_pipeline.legacy_incremental_emission_loss_455 import event_loss
from weld_pipeline.legacy_incremental_corrective_training_loop_425 import tensor, attribute_backward
from weld_pipeline.legacy_frozen_incremental_model_392 import tensor_state_digest


def state_digest(state):
    result = hashlib.sha256()
    for key, value in sorted(state.items()):
        result.update(str(key).encode())
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().contiguous().numpy()
            result.update(str(array.dtype).encode())
            result.update(str(array.shape).encode())
            result.update(array.tobytes())
        else:
            result.update(repr(value).encode())
    return result.hexdigest()


def migrate_optimizer_from260(model, recovery):
    named = dict(model.named_parameters())
    name_by_identity = {id(value): key for key, value in named.items()}
    events = list(model.events.parameters())
    horizontal = list(model.horizontal_residual.parameters())
    attributes = list(model.attributes.parameters()) + list(model.path_attributes.parameters())
    # Reconstruct the verified427 saved layout before any regrouping.
    groups = [dict(params=events, lr=1e-4), dict(params=horizontal, lr=1e-4), dict(params=attributes, lr=3e-4)]
    saved = recovery['optimizer_state']
    assert len(saved['param_groups']) == len(groups)
    bindings = []
    for actual, serialized in zip(groups, saved['param_groups']):
        assert len(actual['params']) == len(serialized['params'])
        bindings.extend((name_by_identity[id(p)], p, original_id) for p, original_id in zip(actual['params'], serialized['params']))
    assert len({name for name, p, original_id in bindings}) == len(bindings)
    assert {original_id for name, p, original_id in bindings} == {v for group in saved['param_groups'] for v in group['params']}
    for name, p, original_id in bindings:
        for key in ('exp_avg', 'exp_avg_sq'):
            if original_id in saved['state']:
                assert tuple(saved['state'][original_id][key].shape) == tuple(p.shape), (name, key)
    old = torch.optim.AdamW(groups, weight_decay=1e-4)
    old.load_state_dict(saved)
    before = {name: state_digest(old.state[p]) for name, p, original_id in bindings if p in old.state}
    assert len(before) == len(saved['state'])
    for name, p, original_id in bindings:
        if p in old.state:
            assert state_digest(old.state[p]) == state_digest(saved['state'][original_id]), name
    options = {key: value for key, value in old.param_groups[0].items() if key not in ('params', 'lr')}
    active_groups = [
        dict(options, params=list(model.events.shared.parameters()), lr=3e-5, phase_name='event_shared'),
        dict(options, params=list(model.events.presence.parameters()), lr=1e-4, phase_name='presence'),
        dict(options, params=list(model.events.count.parameters()) + list(model.events.gap.parameters()) + list(model.events.direction.parameters()) + list(model.events.closed.parameters()), lr=3e-4, phase_name='count_gap_direction_closed'),
        dict(options, params=list(model.events.positions.parameters()), lr=3e-4, phase_name='positions_mm'),
        dict(options, params=attributes, lr=3e-4, phase_name='whole_and_local_attributes'),
        dict({key: value for key, value in old.param_groups[1].items() if key != 'params'}, params=horizontal, phase_name='frozen_pointwise_H_residual'),
    ]
    optimizer = torch.optim.AdamW(active_groups, weight_decay=1e-4)
    optimizer.state.update(old.state)
    assert {id(p) for group in active_groups for p in group['params']} == {id(p) for p in model.incremental_parameters()}
    assert len(optimizer.state) == len(old.state)
    for p in horizontal:
        p.requires_grad_(False)
        p.grad = None
    after = {name: state_digest(optimizer.state[p]) for name, p, original_id in bindings if p in optimizer.state}
    assert after == before
    record = dict(compatible_state_entries=len(before), all_parameter_names_shapes_and_state_contents_verified=True,
        verified_saved_layout='427 events then H residual then attributes; source and recovery bound by caller',
        optimizer_state_by_parameter_name_sha256=after,
        saved_optimizer_id_by_parameter_name={name: original_id for name, p, original_id in bindings},
        pointwise_H_parameter_sha256=tensor_state_digest(model.horizontal_residual),
        pointwise_H_optimizer_state_sha256={name_by_identity[id(p)]: state_digest(optimizer.state[p]) for p in horizontal if p in optimizer.state},
        parameter_names_by_new_group={group['phase_name']: [name_by_identity[id(p)] for p in group['params']] for group in active_groups})
    return optimizer, record


def norm(parameters):
    return math.sqrt(sum(float(p.grad.detach().double().square().sum().cpu()) for p in parameters if p.grad is not None))


def group_shared_gradient_evidence(model, detail):
    parameters = list(model.events.shared.parameters())
    evidence = {}
    for group, terms in detail['group_presence_terms'].items():
        gradients, info = {}, {}
        for name, loss in terms.items():
            gradient = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            gradient = [torch.zeros_like(p) if value is None else value for p, value in zip(parameters, gradient)]
            assert all(bool(torch.isfinite(value).all()) for value in gradient)
            gradients[name] = gradient
            info[name + '_loss'] = float(loss.detach().cpu())
            info[name + '_shared_gradient_norm'] = math.sqrt(sum(float(value.double().square().sum().cpu()) for value in gradient))
        a, b = gradients['negative'], gradients['emission_keep']
        denominator = info['negative_shared_gradient_norm'] * info['emission_keep_shared_gradient_norm']
        info['negative_vs_keep_shared_gradient_cosine'] = sum(float((x.double() * y.double()).sum().cpu()) for x, y in zip(a, b)) / denominator if denominator else None
        evidence[str(group)] = info
    return evidence


def numeric_attribute_evidence(model, source, device):
    # TRAIN diagnostic on current target paths; value loss itself has no predicted-state mask.
    from weld_pipeline.contract_field_acceptance_297 import FIELDS
    indices = np.flatnonzero(source['path_available'])
    fields = {name: dict(known_state=0, correct_state=0, known_numeric=0, conditional_numeric_count=0,
                        numeric_absolute_error_sum=0., conditional_numeric_absolute_error_sum=0.,
                        target_min=None, target_max=None, prediction_min=None, prediction_max=None) for name in FIELDS}
    with torch.no_grad():
        for start in range(0, len(indices), 128):
            rows = indices[start:start + 128]
            output = model.path_attributes(tensor(source['path_features'][rows], device))
            states = output['state_logits'].argmax(-1).cpu().numpy()
            values = output['scalar_values'].cpu().numpy()
            for j, name in enumerate(FIELDS):
                known = source['path_targets']['state_known'][rows, j]
                target_state = source['path_targets']['state_class'][rows, j]
                correct = states[:, j] == target_state
                row = fields[name]
                row['known_state'] += int(known.sum())
                row['correct_state'] += int((known & correct).sum())
                numeric = source['path_targets']['scalar_known'][rows, j]
                if not numeric.any():
                    continue
                wanted = source['path_targets']['scalar_values'][rows, j][numeric]
                predicted = values[:, j][numeric]
                assert np.isfinite(wanted).all() and np.isfinite(predicted).all()
                error = np.abs(predicted - wanted)
                conditional = correct[numeric]
                row['known_numeric'] += len(wanted)
                row['numeric_absolute_error_sum'] += float(error.sum())
                row['conditional_numeric_count'] += int(conditional.sum())
                row['conditional_numeric_absolute_error_sum'] += float(error[conditional].sum())
                for label, array in (('target', wanted), ('prediction', predicted)):
                    lo, hi = float(array.min()), float(array.max())
                    row[label + '_min'] = lo if row[label + '_min'] is None else min(lo, row[label + '_min'])
                    row[label + '_max'] = hi if row[label + '_max'] is None else max(hi, row[label + '_max'])
    return dict(fields=fields, role='TRAIN path diagnostic, not free E16 acceptance',
                known_numeric_loss_not_masked_by_predicted_unknown=True, physical_distance_targets_mm_unchanged=True)


def source_update(model, optimizer, source, device):
    model.train()
    model.horizontal_residual.eval()
    optimizer.zero_grad(set_to_none=True)
    H_before = tensor_state_digest(model.horizontal_residual)
    frozen_state = {id(p): state_digest(optimizer.state[p]) for p in model.horizontal_residual.parameters() if p in optimizer.state}
    base = {key: tensor(value, device) for key, value in source['base'].items()}
    output = model.events(tensor(source['features'], device), base)
    event, detail = event_loss(output, source['parents'], base, tensor(source['background_known'], device),
                             tensor(source['group_ids'], device), tensor(source['hard_negative_weights'], device), source['original_G'])
    available_ids = set(source['path_original_child_ids'])
    assert all(set(row['canonical_original_child_ids']) <= available_ids for row in detail['assignments'])
    shared_gradients = group_shared_gradient_evidence(model, detail)
    attribute_diagnostic = numeric_attribute_evidence(model, source, device)
    event.backward()
    ordinary, path_rows = attribute_backward(model.path_attributes, source['path_features'], source['path_targets'], source['path_available'], device)
    local, local_rows = attribute_backward(model.attributes, source['local_features'], source['local_targets'], source['local_available'], device)
    parameters = model.incremental_parameters()
    assert all(p.grad is None for old in (model.original115, model.original_hybrid_horizontal, model.horizontal_residual) for p in old.parameters())
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)
    before = {group['phase_name']: [p.detach().clone() for p in group['params']] for group in optimizer.param_groups}
    norms = {group['phase_name']: norm(group['params']) for group in optimizer.param_groups}
    combined = torch.nn.utils.clip_grad_norm_(parameters, 10.)
    assert bool(torch.isfinite(combined))
    optimizer.step()
    changes = {group['phase_name']: max(float((p.detach() - old).abs().max().cpu()) for p, old in zip(group['params'], before[group['phase_name']])) for group in optimizer.param_groups}
    assert changes['frozen_pointwise_H_residual'] == 0. and tensor_state_digest(model.horizontal_residual) == H_before
    assert {id(p): state_digest(optimizer.state[p]) for p in model.horizontal_residual.parameters() if p in optimizer.state} == frozen_state
    model.verify_frozen()
    losses = dict(event_total=float(event.detach().cpu()), whole_path_attribute_weighted=ordinary,
                  local_attribute_weighted=local, horizontal_shadow=0., preclip_gradient_norm=float(combined.cpu()))
    assert all(math.isfinite(value) for value in losses.values())
    return dict(losses=losses, event_details={key: float(value.detach().cpu()) for key, value in detail.items() if isinstance(value, torch.Tensor)},
        training_correspondence_failure_counts=detail['training_correspondence_failure_counts'],
        protected_complete_parent_count=detail['protected_complete_parent_count'],
        reserved_train_anchor_count=detail['reserved_train_anchor_count'],
        current_correct_V_TRAIN_reference_count=source['current_correct_V_ids_used_only_in_TRAIN'],
        group_presence_shared_gradient_evidence=shared_gradients, attribute_train_diagnostic=attribute_diagnostic,
        parameter_group_preclip_gradient_norm=norms, parameter_group_maximum_absolute_update=changes,
        combined_clip10_triggered=bool(float(combined) > 10.),
        original_G=source['original_G'], original_E16_records=source['original_E16_records'],
        original_local_points=source['original_local_points'], available_whole_path_rows=path_rows,
        available_local_reference_rows=local_rows, representation_miss_children=source['representation_miss_children'],
        event_NN_forwards=1, horizontal_shadow_NN_forwards=0, optimizer_steps=1,
        frozen_H_parameter_and_optimizer_state_unchanged=True, published_H_unchanged=True, formal_E_records_created=0)
