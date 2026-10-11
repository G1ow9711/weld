"""Resume639 by verified parameter identity; split six event heads and allocation."""
from copy import deepcopy
import torch


EVENT_HEADS = ('presence', 'count', 'gap', 'direction', 'closed', 'margin', 'allocation')


def _copy_state(value, parameter, key=None):
    if isinstance(value, torch.Tensor):
        # Adam's non-capturable scalar step remains on its saved CPU device.
        return value.detach().clone() if key == 'step' else value.detach().to(parameter.device).clone()
    if isinstance(value, dict):
        return {k: _copy_state(v, parameter, k) for k, v in value.items()}
    return deepcopy(value)


def _state_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_state_equal(a[k], b[k]) for k in a)
    return a == b


def resume_setloss_optimizers(model, recovery):
    """No step, RNG mutation, state reset, or index-only state reassignment."""
    bundle = recovery['bundle']
    assert bundle['actual_optimizer_updates'] == 639
    parameters = dict(model.named_parameters())
    result, bindings, evidence = {}, {}, {}
    all_owned = []
    for branch in ('H', 'V', 'attributes'):
        saved = recovery['independent_optimizer_states'][branch]
        names = list(recovery['independent_optimizer_named_bindings'][branch])
        ids = [i for group in saved['param_groups'] for i in group['params']]
        assert len(ids) == len(set(ids)) == len(names) == len(set(names))
        assert all(name in parameters and parameters[name].requires_grad for name in names)
        assert set(saved['state']) <= set(ids)
        lookup = {}; offset = 0
        for group in saved['param_groups']:
            for identity in group['params']:
                name = names[offset]; parameter = parameters[name]
                module, relative = name.split('.', 1)
                assert module in bundle['incremental_state']
                assert torch.equal(parameter.detach().cpu(), bundle['incremental_state'][module][relative].cpu()), name
                state = saved['state'].get(identity)
                if state:
                    for slot in ('exp_avg', 'exp_avg_sq'):
                        assert state[slot].shape == parameter.shape and state[slot].dtype == parameter.dtype, (name, slot)
                lookup[name] = (parameter, identity, group, state)
                offset += 1
        groups, new_names = [], []
        if branch in ('H', 'V'):
            expected = {name for name in parameters if name.startswith('events.'+branch+'.') and parameters[name].requires_grad}
            assert set(names) == expected
            for head in EVENT_HEADS:
                selected = [name for name in names if name.startswith('events.'+branch+'.'+head+'.')]
                assert selected, (branch, head)
                template = {k: deepcopy(v) for k,v in lookup[selected[0]][2].items() if k not in ('params','phase_name')}
                assert template['lr'] == (3e-4 if head == 'allocation' else 1e-4)
                for name in selected:
                    options = {k:v for k,v in lookup[name][2].items() if k not in ('params','phase_name')}
                    assert options == template
                groups.append(dict(template,params=[parameters[name] for name in selected],phase_name=branch+'_'+head))
                new_names.extend(selected)
        else:
            expected = {name for name in parameters if name.startswith(('attributes.','path_attributes.')) and parameters[name].requires_grad}
            assert set(names) == expected
            offset = 0
            for old_group in saved['param_groups']:
                selected = names[offset:offset+len(old_group['params'])]; offset += len(selected)
                options = {k:deepcopy(v) for k,v in old_group.items() if k != 'params'}
                groups.append(dict(options, params=[parameters[name] for name in selected]))
                new_names.extend(selected)
        assert set(new_names) == set(names) and len(new_names) == len(names)
        optimizer = torch.optim.AdamW(groups)
        copied = []
        for name, (parameter, identity, old_group, old_state) in lookup.items():
            if old_state is not None:
                optimizer.state[parameter] = _copy_state(old_state,parameter)
                assert _state_equal(old_state,optimizer.state[parameter]), name
                copied.append(name)
        assert len(optimizer.state) == len(saved['state'])
        assert [id(p) for g in optimizer.param_groups for p in g['params']] == [id(parameters[n]) for n in new_names]
        evidence[branch] = dict(parameter_names_in_actual_group_order=new_names,
            copied_state_names=copied, copied_state_count=len(copied),
            saved_missing_states_preserved=[n for n in names if n not in copied],
            groups=[dict(phase_name=g['phase_name'],lr=g['lr'],weight_decay=g['weight_decay'],
                parameter_count=len(g['params'])) for g in optimizer.param_groups],
            all_values_dtypes_shapes_and_steps_exactly_preserved=True)
        result[branch] = optimizer; bindings[branch] = new_names
        all_owned.extend(id(parameters[n]) for n in new_names)
    assert len(all_owned) == len(set(all_owned))
    return result, bindings, evidence
