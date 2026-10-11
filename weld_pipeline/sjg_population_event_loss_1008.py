"""Population-correct event objective for a future reviewed TRAIN experiment.

The sampler may oversample endpoints, but neither sample frequency nor an
extra positive-class weight may change the population posterior optimum.
Denominators are fixed TRAIN population mask masses, never minibatch ratios.
This module does not mutate or retrain run187 and does not calibrate run188.
"""
import numpy as np
import torch
from torch.nn import functional as F
from weld_pipeline.sjg_probability_corrected_queries_1007 import labels_for, grid_positions


def population_statistics(profiles, supervision, layout):
    total = int(layout['ptr'][-1])
    if total <= 0:
        raise ValueError('Nonempty inference grid required')
    known = np.array([total, total, total, total, 0], dtype=np.float64)
    positive = np.zeros(5, np.float64)
    offset_known = np.zeros(2, np.float64)
    # Non-compatible carriers have known negative first-four heads, unknown
    # direction and unknown offsets. Only compatible carriers need evaluation.
    for key in supervision['by_profile']:
        node = int(key)
        count = int(layout['counts'][node])
        for start in range(0, count, 65536):
            local = np.arange(start, min(count, start + 65536))
            q = grid_positions(layout, np.full(len(local), node), local)
            labels = labels_for(profiles, supervision, np.full(len(q), node), q)
            positive += (labels['target'] * labels['target_mask']).sum(0, dtype=np.float64)
            known[4] += labels['target_mask'][:, 4].sum()
            offset_known += labels['offset_mask'].sum(0)
    return dict(total_queries=total, known_fraction=(known / total).tolist(),
                positive_fraction=(positive / total).tolist(),
                conditional_positive_prior=np.divide(positive, known, out=np.zeros(5), where=known > 0).tolist(),
                offset_known_fraction=(offset_known / total).tolist(),
                event_positive_weights=[1, 1, 1, 1, 1], optimizer_updates=0)


def loss_for(logits, offset, labels, importance_weight, statistics):
    n = len(logits)
    if n < 1 or logits.shape != labels['target'].shape or logits.shape[1] != 5:
        raise ValueError('Nonempty five-head query batch required')
    weight = torch.as_tensor(importance_weight, dtype=logits.dtype, device=logits.device)
    if weight.shape != (n,) or not torch.isfinite(weight).all() or not (weight > 0).all():
        raise ValueError('Every query requires a finite positive population weight')
    known = labels['target_mask'].to(logits.dtype)
    mask = labels['offset_mask'].to(offset.dtype)
    event_mass = float(np.sum(statistics['known_fraction']))
    offset_mass = float(np.sum(statistics['offset_known_fraction']))
    if event_mass <= 0 or offset_mass < 0:
        raise ValueError('Invalid full-population mask masses')
    # No pos_weight: inverse sampling probabilities already correct the prior.
    binary = F.binary_cross_entropy_with_logits(logits, labels['target'], reduction='none')
    if 'event_head_scale' in statistics:
        scale = logits.new_tensor(statistics['event_head_scale'])
        if scale.shape != (5,) or not torch.isfinite(scale).all() or (scale < 0).any():
            raise ValueError('Invalid fixed TRAIN head scales')
        binary = binary * scale
    event = (binary * known * weight[:, None]).sum() / (n * event_mass)
    if offset_mass:
        regression = (F.smooth_l1_loss(offset, labels['offset'], reduction='none') * mask * weight[:, None]).sum() / (n * offset_mass)
    else:
        if mask.any():
            raise ValueError('Sample contains known offsets absent from population')
        regression = offset.sum() * 0
    return event + regression, event, regression


def controls():
    from weld_pipeline.sjg_probability_corrected_queries_1007 import mixture_probability, grid_layout
    from weld_pipeline.sjg_local_curve_supervision_915 import compatible_interval
    # Exhaustive mixture expectation, including variable/unknown masks and
    # soft targets: compare values AND gradients with the whole-grid loss.
    n = 101
    x = torch.arange(n, dtype=torch.float64)
    labels = dict(target=torch.stack([(x % 7 == 0).double(), torch.exp(-((x - 3) / 2) ** 2),
                                     torch.exp(-((x - 80) / 2) ** 2), torch.zeros(n), (x % 3 == 0).double()], 1),
                  target_mask=torch.ones(n, 5, dtype=torch.bool),
                  offset=torch.stack([torch.sin(x), torch.cos(x)], 1),
                  offset_mask=torch.stack([x < 8, x > 75], 1))
    labels['target_mask'][:, 4] = (x % 4 == 0)
    stats = dict(known_fraction=labels['target_mask'].double().mean(0).tolist(),
                 offset_known_fraction=labels['offset_mask'].double().mean(0).tolist())
    logits = torch.sin(x[:, None] + torch.arange(5)).requires_grad_()
    offsets = torch.stack([torch.cos(x), torch.sin(x)], 1).requires_grad_()
    full, _, _ = loss_for(logits, offsets, labels, torch.ones(n), stats)
    full_grads = torch.autograd.grad(full, (logits, offsets), retain_graph=True)
    p = mixture_probability(np.arange(n), n, np.array([1, 2, 3, 4, 5, 79, 80]), .5)
    expected = logits.sum() * 0
    for i in range(n):
        mini = {k: v[i:i+1] for k, v in labels.items()}
        value, _, _ = loss_for(logits[i:i+1], offsets[i:i+1], mini, [1 / n / p[i]], stats)
        expected = expected + p[i] * value
    grads = torch.autograd.grad(expected, (logits, offsets))
    assert torch.allclose(full, expected, atol=1e-12, rtol=0)
    errors = [float((a - b).abs().max()) for a, b in zip(full_grads, grads)]
    assert max(errors) < 1e-12
    assert not grads[0][~labels['target_mask']].any()
    assert not grads[1][~labels['offset_mask']].any()
    # A constant logit is stationary at the natural soft-positive prior.
    prior = labels['target'][:, 1].mean()
    bias = torch.logit(prior).detach().requires_grad_()
    risk = F.binary_cross_entropy_with_logits(bias.expand(n), labels['target'][:, 1])
    derivative = float(torch.autograd.grad(risk, bias)[0])
    assert abs(derivative) < 1e-12
    profiles = [dict(representation=dict(kind='line', endpoints_mm=[[0, 0, 0], [100, 0, 0]])),
                dict(representation=dict(kind='line', endpoints_mm=[[0, 30, 0], [100, 30, 0]]))]
    event = compatible_interval(profiles[0], dict(kind='line', endpoints_mm=[[20, 0, 0], [40, 0, 0]]))
    sup = dict(by_profile={'0': [event]})
    layout = grid_layout(profiles, sup)
    measured = population_statistics(profiles, sup, layout)
    nodes = np.repeat(np.arange(2), layout['counts'])
    q = np.tile(np.linspace(0, 100, 51), 2)
    exhaustive = labels_for(profiles, sup, nodes, q)
    assert np.allclose(measured['known_fraction'], exhaustive['target_mask'].mean(0))
    assert np.allclose(measured['offset_known_fraction'], exhaustive['offset_mask'].mean(0))
    assert np.allclose(measured['positive_fraction'], (exhaustive['target'] * exhaustive['target_mask']).astype(np.float64).mean(0))
    blank = dict(labels)
    blank['offset_mask'] = torch.zeros_like(labels['offset_mask'])
    empty_stats = dict(stats, offset_known_fraction=[0, 0])
    _, _, zero = loss_for(logits, offsets, blank, torch.ones(n), empty_stats)
    assert zero == 0
    return dict(exact_mixture_risk_equals_population_risk=True, maximum_gradient_error=max(errors),
                unknown_labels_have_zero_gradient=True, natural_prior_optimum_gradient=derivative,
                population_masks_verified_by_enumeration=True, absent_offsets_finite_zero=True,
                event_positive_weights=[1, 1, 1, 1, 1], optimizer_updates=0,
                current_model_and_thresholds_unchanged=True)
