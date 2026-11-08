"""Fit only98 source-endpoint parameters on an exactly frozen own backbone."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
import pickle
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tools import curveonly_01a1ac7c_experiment as e
from tools import curveonly_01a1ac7c_event_mass_train_v201 as previous
from weld_pipeline import curveonly_01a1ac7c_source_endpoint_v203 as m

SETTINGS = dict(seed=20261106, CPU_threads=2, GPU_memory_fraction=.35,
    observations=111, backbone_epochs=128, backbone_retrained=False,
    source_head_parameters=98, source_solver='CPU_FLOAT64_LBFGS', source_max_iter=128,
    source_L2=1e-4, source_numeric_tolerance_mm=m.codec.reference.PROJECTION_NUMERIC_MM,
    source_class_weighting='equal_case_then_present_role_then_present_class',
    validation_fit_or_threshold_calibration=False)
configure = previous.configure


def bindings():
    frozen = e.read(e.REC/'event_mass_fit_v202/ALL_OUTPUTS_FROZEN.json')
    return frozen['model'], frozen['fit']


def backbone(cases, model_binding=None, fit_binding=None):
    model_binding, fit_binding = (model_binding, fit_binding) if model_binding else bindings()
    for b in (model_binding, fit_binding):
        assert str(Path(b['path']).resolve()).startswith(str(e.REC.resolve())+'/')
        assert e.sha(b['path']) == b['sha256']
    fit = e.read(fit_binding['path'])
    assert {q['source_id'] for q in cases} == {q['source_id'] for q in fit['cases']}
    assert all(q['split'] == 'TRAIN' for q in cases)
    with Path(model_binding['path']).open('rb') as file:
        original = pickle.load(file)
    torch.manual_seed(SETTINGS['seed'])
    torch.set_num_threads(SETTINGS['CPU_threads'])
    torch.use_deterministic_algorithms(True)
    network = m.BoundaryNetwork().cpu().eval()
    network.load_backbone(original.network.state_dict())
    assert sum(q.numel() for q in network.parameters() if q.requires_grad) == 98
    return original, network, model_binding, fit_binding


def extract(cases, network, mean, scale):
    rows, feature_records = [], []
    for case in sorted(cases, key=lambda q: q['source_id']):
        for key in ('features', 'geometry', 'local_targets'):
            assert e.sha(case[key]['path']) == case[key]['sha256']
        packet = e.read(case['geometry']['path'])
        current = m.candidate_rows(np.load(case['features']['path']), packet)
        targets = [m.targets(row, [q for q in case['chosen'] if q['candidate'] == row['candidate']])
                   for row in current]
        for row, target in zip(current, targets):
            row['source_id'] = case['source_id']
            row['source_target'] = target
        rows.extend(current)
        feature_records.append(dict(source_id=case['source_id'], features=case['features'],
                                    geometry=case['geometry'], candidates=len(current)))
    forward, reverse, labels, roles, examples = [], [], [], [], []
    with torch.no_grad():
        for ids in m.base.batches(rows):
            selected = [rows[i] for i in ids]
            batch = m.pack(selected, mean, scale, 'cpu')
            a, b = network.hidden_directions(batch)
            for j, row in enumerate(selected):
                if row['closed']:
                    continue
                target = row['source_target']
                for code in target['codes']:
                    for role, index, boundary in (
                            (0, code['start_index'], 0),
                            (1, code['end_index'], len(row['grid'])-1)):
                        if index != boundary:
                            continue
                        forward.append(a[j, :, index].numpy().copy())
                        reverse.append(b[j, :, index].numpy().copy())
                        value = float(target['fields'][6+role, index])
                        labels.append(value)
                        roles.append(role)
                        examples.append(dict(source_id=row['source_id'], candidate=row['candidate'],
                            truth=code['truth'], role=role, grid_index=index, exact_source_label=value))
    assert examples
    weights = np.zeros(len(examples), float)
    source_ids = np.asarray([q['source_id'] for q in examples])
    labels, roles = np.asarray(labels, float), np.asarray(roles, np.int64)
    case_count = 0
    for sid in sorted(set(source_ids)):
        take = source_ids == sid
        present_roles = sum(bool(np.any(take & (roles == role))) for role in (0, 1))
        for role in (0, 1):
            active = take & (roles == role)
            present_classes = sum(bool(np.any(active & (labels == label))) for label in (0, 1))
            for label in (0, 1):
                same = active & (labels == label)
                if same.any():
                    weights[same] = 1/(present_roles*present_classes*int(same.sum()))
        assert abs(weights[take].sum()-1) < 1e-9
        case_count += 1
    assert abs(weights.sum()-case_count) < 1e-9
    return rows, np.asarray(forward), np.asarray(reverse), labels, roles, weights, examples, feature_records


def head_logits(weight, bias, forward, reverse, roles):
    a = (forward*weight[roles]).sum(1)+bias[roles]
    b = (reverse*weight[1-roles]).sum(1)+bias[1-roles]
    return (a+b)*.5


def assembly_control(cases, destination):
    original, network, model_binding, fit_binding = backbone(cases)
    data = extract(cases, network, original.mean, original.scale)
    _, a, b, labels, roles, weights, examples, records = data
    tensors = [torch.as_tensor(q) for q in (a, b, labels, roles, weights)]
    logits = head_logits(network.source_head.weight[:, :, 0].double(),
        network.source_head.bias.double(), tensors[0].double(), tensors[1].double(), tensors[3])
    loss = (F.binary_cross_entropy_with_logits(logits, tensors[2].double(), reduction='none')*
            tensors[4].double()).sum()/tensors[4].sum()
    loss.backward()
    assert all(p.grad is None for name, p in network.named_parameters() if not name.startswith('source_head.'))
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in network.source_head.parameters())
    e.write(Path(destination)/'ASSEMBLY_CONTROL.json', dict(passed=True,
        TRAIN_inputs=len(cases), supervised_source_endpoint_examples=len(examples),
        source_labels=dict(negative=int((labels == 0).sum()), positive=int((labels == 1).sum())),
        eligible_cases=len({q['source_id'] for q in examples}), case_mass=float(weights.sum()),
        trainable_parameters=98, total_parameters=sum(p.numel() for p in network.parameters()),
        frozen_backbone_gradient_absent=True, new_head_gradient_finite=True,
        backbone_model=model_binding, backbone_fit=fit_binding, feature_records=records,
        no_optimizer_step_or_saved_random_weights=True))


def fit(cases, destination, model_binding=None, fit_binding=None):
    start = time.perf_counter()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    original, network, model_binding, fit_binding = backbone(cases, model_binding, fit_binding)
    rows, a, b, labels, roles, weights, examples, records = extract(
        cases, network, original.mean, original.scale)
    np.savez_compressed(destination/'SOURCE_ENDPOINT_EXAMPLES.npz', forward=a, reverse=b,
                        labels=labels, roles=roles, weights=weights)
    forward, reverse = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    y, role, w = torch.as_tensor(labels), torch.as_tensor(roles), torch.as_tensor(weights)
    weight = torch.nn.Parameter(network.source_head.weight[:, :, 0].detach().double().clone())
    bias = torch.nn.Parameter(network.source_head.bias.detach().double().clone())
    optimizer = torch.optim.LBFGS([weight, bias], lr=1., max_iter=SETTINGS['source_max_iter'],
        tolerance_grad=1e-7, tolerance_change=1e-10, line_search_fn='strong_wolfe')
    trace = []
    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits = head_logits(weight, bias, forward, reverse, role)
        ce = (F.binary_cross_entropy_with_logits(logits, y, reduction='none')*w).sum()/w.sum()
        loss = ce+SETTINGS['source_L2']*weight.square().mean()
        assert torch.isfinite(loss).item()
        loss.backward()
        trace.append(dict(evaluation=len(trace)+1, loss=float(loss.detach()),
                          weighted_BCE=float(ce.detach())))
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        network.source_head.weight.copy_(weight[:, :, None].float())
        network.source_head.bias.copy_(bias.float())
    assert all(torch.equal(p, dict(original.network.named_parameters())[name])
               for name, p in network.named_parameters() if not name.startswith('source_head.'))
    before = m.core.predict_fields(original.network, rows, original.mean, original.scale)
    after = m.predict_fields(network, rows, original.mean, original.scale)
    assert all(np.array_equal(x, z[:6]) for x, z in zip(before[0], after[0]))
    assert np.array_equal(before[1], after[1])
    network.eval()
    torch.save(dict(network=network.state_dict(), mean=original.mean, scale=original.scale,
                    settings=SETTINGS), destination/'weights.pt')
    restored = m.BoundaryNetwork()
    restored.load_state_dict(torch.load(destination/'weights.pt', map_location='cpu', weights_only=False)['network'])
    replay = m.predict_fields(restored, rows, original.mean, original.scale)
    assert all(np.array_equal(x, z) for x, z in zip(after[0], replay[0]))
    assert np.array_equal(after[1], replay[1])
    with torch.no_grad():
        logits = head_logits(network.source_head.weight[:, :, 0].double(), network.source_head.bias.double(),
                             forward, reverse, role)
        probability = torch.sigmoid(logits).numpy()
    state = optimizer.state[weight]
    result = dict(weights=e.bind(destination/'weights.pt'), settings=SETTINGS,
        parameter_count=sum(p.numel() for p in network.parameters()), trainable_parameters=98,
        TRAIN_inputs=len(cases), cases=records, backbone_model=model_binding, backbone_fit=fit_binding,
        backbone_weights_and_normalizer_unchanged=True,
        original_six_fields_and_full_cycle_bit_exact_on_all_fit_inputs=True,
        full_fit_population_CPU_reload_fields_bit_exact=True,
        source_examples=e.bind(destination/'SOURCE_ENDPOINT_EXAMPLES.npz'),
        source_example_metadata=examples, eligible_cases=len({q['source_id'] for q in examples}),
        source_example_probabilities=probability.tolist(),
        source_weighted_accuracy=float(((((probability > .5) == labels)*weights).sum())/weights.sum()),
        optimizer_iterations=int(state['n_iter']), optimizer_function_evaluations=int(state['func_evals']),
        optimizer_trace=trace, elapsed_seconds=time.perf_counter()-start,
        no_new_backbone_training_or_held_or_VALfit=True)
    e.write(destination/'FIT.json', result)
    print('FROZEN_SOURCE_ENDPOINT_HEAD_FIT', len(examples), result['source_weighted_accuracy'], flush=True)
    return network, original.mean, original.scale, result


def mathematical_controls():
    return dict(previous.mathematical_controls(),
                new_source_head_does_not_modify_existing_event_mass_objective=True)
