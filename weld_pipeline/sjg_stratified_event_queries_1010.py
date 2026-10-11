"""TRAIN event-stratified proposal with exact population importance weights.

Each known child endpoint and interior gets a stratum. Shared query points can belong to
several strata; their probabilities add. Half the draws remain uniform over
the complete input grid. No carrier is discarded and no held-out label is read.
"""
import numpy as np
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_probability_corrected_queries_1007 import grid_layout, grid_positions, labels_for


def event_layout(profiles, supervision):
    layout = grid_layout(profiles, supervision)
    groups = {}
    for raw_node, events in supervision['by_profile'].items():
        node = int(raw_node);curve = CurveCarrier(profiles[node]['representation'])
        q = curve.grid(2.)
        labels = labels_for(profiles, supervision, np.full(len(q), node), q)
        for event in events:
            identity = (int(event['parent']), int(event['child']))
            if event['closed_whole']:
                selected = labels['target'][:, 3] > .5
                groups.setdefault(identity + ('closed',), []).append(layout['ptr'][node] + np.flatnonzero(selected))
                continue
            inside = np.zeros(len(q), bool)
            for lo, hi in event['intervals']:
                inside |= (q >= lo) & (q <= hi)
            groups.setdefault(identity + ('interior',), []).append(layout['ptr'][node] + np.flatnonzero(inside))
            for head, name in enumerate(('lo', 'hi')):
                delta = q - event[name]
                offset_error = q + labels['offset'][:, head] * 10 - event[name]
                if curve.kind == 'arc' and curve.arc.is_closed:
                    delta = (delta + curve.length / 2) % curve.length - curve.length / 2
                    offset_error = (offset_error + curve.length / 2) % curve.length - curve.length / 2
                selected = (abs(delta) <= 2.) & labels['offset_mask'][:, head] & (abs(offset_error) <= 1e-5)
                groups.setdefault(identity + (name,), []).append(layout['ptr'][node] + np.flatnonzero(selected))
    keys = sorted(groups)
    members = [np.unique(np.concatenate(groups[key])) for key in keys]
    if supervision['missing'] or any(len(indices) == 0 for indices in members):
        raise ValueError('Cannot promise endpoint exposure for unsupported TRAIN events')
    layout.update(event_keys=keys, event_groups=members)
    return layout


def proposal_probability(indices, total, groups, allocations, count):
    indices = np.asarray(indices, np.int64)
    allocations = np.asarray(allocations, np.int64)
    uniform = count - int(allocations.sum())
    if total <= 0 or uniform <= 0 or len(groups) != len(allocations):
        raise ValueError('A nonzero whole-grid component is required')
    probability = np.full(indices.shape, uniform / count / total)
    if not len(groups):
        return probability
    flat = np.concatenate(groups)
    contributions = np.concatenate([np.full(len(group), allocated / count / len(group))
                                    for group, allocated in zip(groups, allocations)])
    unique, inverse = np.unique(flat, return_inverse=True)
    mass = np.bincount(inverse, weights=contributions, minlength=len(unique))
    position = np.searchsorted(unique, indices)
    clipped = np.minimum(position, len(unique) - 1)
    valid = (position < len(unique)) & (unique[clipped] == indices)
    probability[valid] += mass[clipped[valid]]
    return probability


def draw(layout, seed, count=2048, focus_fraction=.5):
    if count < 1 or not 0 <= focus_fraction < 1:
        raise ValueError('Positive count and uniform support required')
    total = int(layout['ptr'][-1]);groups = layout['event_groups']
    nf = int(count * focus_fraction) if groups else 0
    if groups and nf < len(groups):
        raise ValueError('Registered query budget cannot expose every TRAIN event')
    rng = np.random.default_rng(seed)
    allocations = np.full(len(groups), nf // len(groups), np.int64) if groups else np.empty(0, np.int64)
    if groups:
        allocations[rng.permutation(len(groups))[:nf % len(groups)]] += 1
    pieces = [rng.integers(total, size=count - nf)]
    pieces.extend(rng.choice(group, size=int(allocated), replace=True) for group, allocated in zip(groups, allocations))
    indices = np.concatenate(pieces);rng.shuffle(indices)
    probability = proposal_probability(indices, total, groups, allocations, count)
    nodes = np.searchsorted(layout['ptr'], indices, side='right') - 1
    local = indices - layout['ptr'][nodes]
    return dict(grid_indices=indices, nodes=nodes, q_mm=grid_positions(layout, nodes, local),
                sampling_probability=probability, importance_weight=(1. / total) / probability,
                focus_fraction=nf/count, event_allocations=allocations)


def controls():
    from weld_pipeline.sjg_local_curve_supervision_915 import compatible_interval
    from weld_pipeline.sjg_train_event_coverage_audit_1009 import event_coverage
    profile = dict(representation=dict(kind='line', endpoints_mm=[[0, 0, 0], [100, 0, 0]]))
    children = [];events = []
    for i, (left, right) in enumerate(((20, 40), (40, 60))):
        event = compatible_interval(profile, dict(kind='line', endpoints_mm=[[left, 0, 0], [right, 0, 0]]))
        event.update(parent=0, child=i);events.append(event)
        children.append(dict(parent=0, child=i, kind='line'))
    supervision = dict(original_G=2, children=children, missing=[], by_profile={'0': events})
    layout = event_layout([profile], supervision);sample = draw(layout, 1010, count=64)
    population = np.arange(int(layout['ptr'][-1]))
    p = proposal_probability(population, len(population), layout['event_groups'], sample['event_allocations'], 64)
    assert np.all(p > 0) and abs(p.sum() - 1) < 1e-12
    risk = np.sin(population) ** 2
    assert abs(np.sum(p * (1/len(population)/p) * risk) - risk.mean()) < 1e-12
    # The shared endpoint belongs to two strata. Ignoring this overlap would
    # produce a wrong denominator and biased importance weights.
    intersections = [set(a) & set(b) for i, a in enumerate(layout['event_groups']) for b in layout['event_groups'][i+1:]]
    assert any(intersections)
    labels = labels_for([profile], supervision, sample['nodes'], sample['q_mm'])
    coverage = event_coverage([profile], supervision, sample['nodes'], sample['q_mm'], labels)
    assert coverage['both_endpoint_peaks_seen'] == 2
    assert coverage['occupancy_seen'] == 2
    assert np.array_equal(sample['q_mm'], CurveCarrier(profile['representation']).grid(2.)[sample['grid_indices']])
    background = dict(layout, event_groups=[], event_keys=[])
    assert np.all(draw(background, 1, count=64)['importance_weight'] == 1)
    try:
        draw(layout, 1, count=4)
    except ValueError:
        pass
    else:
        raise AssertionError('Insufficient registered budget must not silently omit events')
    return dict(overlapping_event_probability_sums_exactly=True, full_grid_risk_expectation_preserved=True,
                every_supported_TRAIN_endpoint_sampled=True, every_supported_TRAIN_interior_sampled=True, full_background_support=True,
                insufficient_budget_fails_explicitly=True, no_events_reverts_to_uniform=True,
                optimizer_updates=0, no_model_accuracy_claim=True)
