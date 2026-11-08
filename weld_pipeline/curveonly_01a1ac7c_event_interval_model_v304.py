"""Generate new should-weld intervals from qualified native structural cells.

Material contact remains an observation. A TRAIN-only classifier decides cell
occupancy; original native paths supply shape and exact source provenance.
The frozen older network supplies additional input hypotheses, never labels.
"""
import math
import time
import numpy as np
from . import curveonly_01a1ac7c_direction_spans_v247 as prior
from . import curveonly_01a1ac7c_qualified_event_cells_v301 as qualification
from . import curveonly_01a1ac7c_native_coverage_selection_v294 as competition
from . import curveonly_01a1ac7c_primitive_output_v219 as primitive
from . import curveonly_01a1ac7c_runtime_v51 as exporter
from . import curveonly_01a1ac7c_corner_publication_v256 as corner
from . import curveonly_01a1ac7c_arc_publication_v6 as publication

events = qualification.original
TOL = events.NUMERIC_MM
PHYSICAL = np.r_[np.arange(107), np.arange(111, 134)]
CONTEXT = ('available', 'signed_log_arclength_delta', 'log_finite_distance_min',
    'log_finite_distance_max', 'finite_witness_any', 'all_faces_valid', 'plane_any',
    'cylinder_any', 'sampling_band_extreme_any', 'normal_dot_tangent_min',
    'normal_dot_tangent_max', 'normal_dot_base_max', 'prior_boundary', 'native_end')
NAMES = tuple(prior.NAMES[i] for i in PHYSICAL)+tuple(
    side+':'+name for side in ('left', 'right') for name in CONTEXT)+(
    'log_cell_width_mm', 'closed_native_carrier', 'finite_event_geometry_available')
HYPERPARAMETERS = dict(max_iter=80, max_leaf_nodes=15, max_depth=4,
    min_samples_leaf=20, l2_regularization=1., learning_rate=.1,
    early_stopping=False, random_state=20261106)


def native_only(packet):
    """Unknown finite geometry does not remove its input or score denominator."""
    rows = []
    for ci, c in enumerate(packet['candidates']):
        p = np.asarray(c['points']); length = float(prior.arclength(p)[-1])
        geo = [dict(q_mm=0., kind='NATIVE_REPRESENTATION_START'),
               dict(q_mm=length, kind='NATIVE_REPRESENTATION_END')]
        for part in c.get('source_material_path', []):
            geo += [dict(q_mm=float(part[k]), kind='NATIVE_MATERIAL_PATH_REPRESENTATION_JOIN')
                    for k in ('start_mm', 'end_mm')]
        rows.append(dict(candidate=ci, length_mm=length,
            closed=bool(np.linalg.norm(p[0]-p[-1]) <= 1e-6),
            qualified_geometry_events=geo, qualified_geometry_nodes_mm=events.nodes(geo, length)))
    return dict(available=False, reason='NO_QUALIFIED_FINITE_STEP', faces=[], candidates=rows,
        no_input_or_truth_removed=True, process_state='UNKNOWN')


def build_geometry(packet, solids):
    raw = events.build(packet, solids)
    return qualification.qualify(packet, raw) if raw['available'] else native_only(packet)


def lift_prior_boundaries(packet, states):
    """Lift only literal native parameter identity, never 3D-near curve aliases."""
    lengths = [float(prior.arclength(c['points'])[-1]) for c in packet['candidates']]
    def mapping(ci, offset=0., scale=1., seen=()):
        assert ci not in seen
        path = packet['candidates'][ci].get('source_material_path')
        if not path:
            return [(ci, offset, scale, 0., lengths[ci])]
        result = []
        for p in path:
            child = p['curve']; length = lengths[child]
            assert abs(p['end_mm']-p['start_mm']-length) <= 1e-7*max(1., length)
            origin = p['end_mm'] if p['reversed'] else p['start_mm']
            result += mapping(child, offset+scale*origin,
                              scale*(-1. if p['reversed'] else 1.), seen+(ci,))
        return result
    maps = [mapping(i) for i in range(len(lengths))]
    primitive_roots, original = {}, {}
    for state in states:
        ci = state['candidate']; length = lengths[ci]
        closed = np.linalg.norm(np.asarray(packet['candidates'][ci]['points'])[0]-
                                np.asarray(packet['candidates'][ci]['points'])[-1]) <= 1e-6
        original[ci] = events.model_events(state, length, closed)
        for event in original[ci]:
            q = event['q_mm']
            for leaf, offset, scale, lo, hi in maps[ci]:
                v = (q-offset)/scale
                if lo-TOL <= v <= hi+TOL:
                    primitive_roots.setdefault(leaf, []).append(min(hi, max(lo, v)))
    result = []
    for ci, paths in enumerate(maps):
        out = list(original.get(ci, []))
        for leaf, offset, scale, lo, hi in paths:
            for q in primitive_roots.get(leaf, []):
                out.append(dict(q_mm=float(offset+scale*q), kind='EXACT_NATIVE_ALIAS_PRIOR_BOUNDARY',
                    geometry_or_process_confirmation=False, no_reference_or_proximity_mapping=True))
        result.append(out)
    return result


def normal_world(row, point, rotation, translation):
    if row['status'] == 'CANONICAL_PLANE':
        n = np.asarray(row['normal'], float)
    else:
        p = (point-translation)@rotation
        axis = np.asarray(row['axis'], float); axis /= np.linalg.norm(axis)
        d = p-row['axis_point_mm']; n = d-(d@axis)*axis
    return (n/max(np.linalg.norm(n), 1e-300))@rotation.T


def event_context(q, geom, learned, points, station, faces, rotation, translation, base):
    source = [v for v in geom if v['kind'] == 'FACE_CARRIER_ROOT' and not v.get('same_owner', False)]
    boundary = float(any(abs(v['q_mm']-q) <= TOL for v in learned))
    native = float(q <= TOL or abs(q-station[-1]) <= TOL)
    if not source:
        return [0.]*12+[boundary, native]
    distance = min(abs(v['q_mm']-q) for v in source)
    roots = [v for v in source if abs(abs(v['q_mm']-q)-distance) <= TOL]
    # Equal-distance events at different q are symmetric observations, all kept.
    tangent, vertical, finite = [], [], []
    for event in roots:
        x = event['q_mm']; index = min(len(points)-2, max(0, np.searchsorted(station, x, side='right')-1))
        vector = points[index+1]-points[index]; vector /= max(np.linalg.norm(vector), 1e-300)
        point = np.array([np.interp(x, station, points[:, k]) for k in range(3)])
        n = normal_world(faces[event['face']], point, rotation, translation)
        tangent.append(abs(float(n@vector))); vertical.append(abs(float(n@base)))
        finite.append(float(event['original_finite_face_distance_mm']))
    delta = sum(v['q_mm']-q for v in roots)/len(roots)
    return [1., math.copysign(math.log1p(abs(delta)), delta), math.log1p(min(finite)),
        math.log1p(max(finite)), float(any(v['finite_witness_state'] == 'FINITE_ORIGINAL_FACE_ROOT_OBSERVED' for v in roots)),
        float(all(v['face_valid'] for v in roots)),
        float(any(v['analytic_surface'] == 'CANONICAL_PLANE' for v in roots)),
        float(any(v['analytic_surface'] == 'CANONICAL_CYLINDER' for v in roots)),
        float(any(v.get('qualification', {}).get('status') == 'CO_CARRIED_BAND_EXTREME_HYPOTHESIS_RETAINED' for v in roots)),
        min(tangent), max(tangent), max(vertical), boundary, native]


def graph(features, packet, inventory, states):
    rows = prior.candidate_rows(features, packet)
    learned = lift_prior_boundaries(packet, states)
    pose = np.asarray(packet.get('pose', {}).get('transform_STEP_to_XYZ', np.eye(4)))
    if pose.shape != (4, 4): pose = np.eye(4)
    rotation, translation = pose[:3, :3], pose[:3, 3]
    base = np.asarray(packet['base_axis'], float); base /= max(np.linalg.norm(base), 1e-300)
    observations, cells, offsets = [], [], [0]
    for row, geo, hypotheses in zip(rows, inventory['candidates'], learned):
        ci = row['candidate']; assert geo['candidate'] == ci
        p = np.asarray(packet['candidates'][ci]['points']); s = prior.arclength(p)
        geoevents = geo['qualified_geometry_events']; length = float(s[-1])
        nodes = events.nodes(geoevents+hypotheses, length)
        grid = np.arange(len(row['grid']))*row['grid_step_mm']; values = row['grid'][:, PHYSICAL]
        if row['closed']:
            grid = np.r_[grid, length]; values = np.vstack([values, values[0]])
        context = [event_context(q, geoevents, hypotheses, p, s, inventory['faces'],
            rotation, translation, base) for q in nodes]
        for index, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])):
            middle = np.array([np.interp((a+b)/2, grid, values[:, j]) for j in range(len(PHYSICAL))])
            value = np.r_[middle, context[index], context[index+1], math.log1p(b-a),
                          row['closed'], inventory['available']]
            observations.append(value)
            cells.append(dict(candidate=ci, a=float(a), b=float(b), closed=row['closed'], length_mm=length))
        offsets.append(len(cells))
    x = np.asarray(observations, np.float32).reshape(-1, len(NAMES))
    assert len(inventory['candidates']) == len(rows) and np.isfinite(x).all()
    return x, dict(cells=cells, offsets=offsets, feature_names=list(NAMES),
        no_label_name_or_absolute_XYZ_features=True, exact_native_prior_alias_lifting=True,
        prior_hypotheses_not_should_weld_certification=True)


def overlap_mm(a, b, intervals):
    end, total = -math.inf, 0.
    for lo, hi in sorted(intervals):
        lo, hi = max(a, lo), min(b, hi)
        if hi > max(lo, end): total += hi-max(lo, end)
        end = max(end, hi)
    return total


def train_targets(packet, graph_data, case):
    """Only canonical/strictly certified native aliases transfer positive labels."""
    assert case['split'] == 'TRAIN'
    lengths = [float(prior.arclength(c['points'])[-1]) for c in packet['candidates']]
    canonical = case['canonical_choices']; spans, positive_candidates = {}, set()
    for q in canonical:
        positive_candidates.add(q['candidate'])
        for ci, a, b in competition.provenance.primitive_intervals(q['candidate'], q['start_mm'], q['end_mm'], packet, lengths):
            spans.setdefault(ci, []).append((a, b, q['truth']))
    unknown = set(case.get('ambiguous_zero_candidates', []))
    target, group, categories, fractions = [], [], [], []
    cells = graph_data['cells']
    for item in cells:
        ci, a, b = item['candidate'], item['a'], item['b']
        parts = competition.provenance.primitive_intervals(ci, a, b, packet, lengths)
        covered = sum(overlap_mm(x, y, [(lo, hi) for lo, hi, t in spans.get(pi, [])]) for pi, x, y in parts)
        # Explicit canonical coverage also includes its original nominal bridges.
        direct = overlap_mm(a, b, [(q['start_mm'], q['end_mm']) for q in canonical if q['candidate'] == ci])
        covered = max(covered, direct); width = b-a
        fraction = min(1., covered/width); label = int(fraction > .5)
        known_native = competition.measure(parts)
        carrier_positive = any(pi in spans for pi, x, y in parts) or ci in positive_candidates
        masked = not label and (ci in unknown or known_native < width-TOL and direct <= TOL)
        truth_overlap = {}
        for pi, x, y in parts:
            for lo, hi, ti in spans.get(pi, []):
                truth_overlap[ti] = truth_overlap.get(ti, 0.)+max(0., min(y, hi)-max(x, lo))
        if label:
            truth = max(truth_overlap, key=truth_overlap.get) if truth_overlap else next(
                q['truth'] for q in canonical if q['candidate'] == ci and min(b, q['end_mm']) > max(a, q['start_mm']))
            key = ('positive_truth', truth); category = 'positive'
        else:
            key = ('internal_gap' if carrier_positive else 'background', ci); category = key[0]
        target.append(-1 if masked else label); group.append(key); categories.append(category); fractions.append(fraction)
    # Equal case, positive truth and negative contiguous gap; aliases cannot
    # inflate a truth's training mass. Unknown cells have exactly zero weight.
    masses = {}; keys = []; last = {}; run = {}
    for i, (value, key, category, item) in enumerate(zip(target, group, categories, cells)):
        ci = item['candidate']
        if category == 'internal_gap':
            if last.get(ci) != (i-1, category): run[ci] = run.get(ci, 0)+1
            key = (*key, run[ci])
        last[ci] = (i, category); keys.append((category, key))
        if value >= 0: masses[category, key] = masses.get((category, key), 0.)+item['b']-item['a']
    category_counts = {category:sum(k[0] == category for k in masses) for category in set(categories)}
    weight = np.array([0. if t < 0 else (item['b']-item['a'])/masses[key]/category_counts[key[0]]
        for t, key, item in zip(target, keys, cells)], np.float64)
    if weight.sum(): weight /= weight.sum()
    return np.asarray(target, np.int8), weight, dict(fractions=fractions, categories=categories,
        canonical_truths=sorted({q['truth'] for q in canonical}), known_cells=int(sum(t >= 0 for t in target)),
        unknown_cells=int(sum(t < 0 for t in target)), positive_cells=int(sum(t == 1 for t in target)),
        category_counts=category_counts, positive_native_propagation_only=True,
        no_unknown_or_unproven_alias_as_physical_negative=True)


def runs(graph_data, probabilities):
    result = []
    for ci, (lo, hi) in enumerate(zip(graph_data['offsets'][:-1], graph_data['offsets'][1:])):
        cells = graph_data['cells'][lo:hi]; p = probabilities[lo:hi]
        changes = np.diff(np.r_[False, p > .5, False].astype(np.int8))
        groups = [list(range(a, b)) for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))]
        if cells and cells[0]['closed'] and len(groups) > 1 and groups[0][0] == 0 and groups[-1][-1] == len(cells)-1:
            groups = [groups[-1]+groups[0]]+groups[1:-1]
        for ids in groups:
            a = cells[ids[0]]['a']; b = cells[ids[-1]]['b']; length = cells[0]['length_mm']
            if b <= a: b += length
            weights = np.array([cells[j]['b']-cells[j]['a'] for j in ids])
            cf = float(np.average(p[ids], weights=weights))
            result.append(dict(candidate=ci, a=a, b=b, confidence=cf,
                cell_indices=[lo+j for j in ids], support_density=float(np.average(2*p[ids]-1, weights=weights))))
    return result


def select(proposals, packet):
    rules = competition.original
    lengths = [float(prior.arclength(c['points'])[-1]) for c in packet['candidates']]
    records, spans, eligible = [], [], []
    for i, p in enumerate(proposals):
        feasible, parts = rules.feasibility(p, packet, lengths); spans.append(parts)
        utility = p['support_density']*competition.measure(parts)
        records.append(dict(proposal=i, candidate=p['candidate'], interval=p['interval'], feasibility=feasible,
            support_density=p['support_density'], utility=utility))
        if feasible['feasible']: eligible.append(i)
    adjacency = [0]*len(eligible); conflicts = []
    for i, a in enumerate(eligible):
        for j, b in enumerate(eligible[:i]):
            if proposals[a]['weld_type'] != proposals[b]['weld_type'] or proposals[a].get('weld_side') != proposals[b].get('weld_side'): continue
            shared = rules.overlap(spans[a], spans[b])
            if shared:
                adjacency[i] |= 1 << j; adjacency[j] |= 1 << i
                conflicts.append(dict(proposals=[b, a], shared_finite_intervals=shared))
    chosen, optimization = competition.maximum_compatible([records[i]['utility'] for i in eligible], adjacency)
    kept = sorted((eligible[i] for i in chosen), key=lambda i: (-proposals[i]['confidence'], i))
    return [proposals[i] for i in kept], dict(input_proposals=len(proposals), kept=len(kept), kept_proposals=kept,
        proposals=records, conflicts=conflicts, optimization=optimization,
        source='NEW_CELL_INTERVALS_WITH_FINITE_NATIVE_COMPETITION', no_prior_continuation_gate=True)


class IntervalModel:
    def __init__(self, backbone=None, head=None):
        self.backbone, self.head = backbone, head
        self.last_timing = self.last_diagnostics = None
        self.fixed_teacher = None

    def predict(self, features, packet):
        start = time.perf_counter()
        data = packet['event_interval_graph_v304']
        x = np.asarray(packet['event_interval_observations_v304'], np.float32)
        if self.fixed_teacher is None:
            probability = self.head.predict_proba(x)[:, list(self.head.classes_).index(1)]
        else:
            probability = np.where(np.asarray(self.fixed_teacher) == 1, 1., 0.)
        intervals = runs(data, probability); raw = []
        for i, q in enumerate(intervals):
            raw.append(dict(candidate=q['candidate'], points=rules_native_slice(packet, q).tolist(),
                weld_type='horizontal', confidence=q['confidence'], support_density=q['support_density'],
                cell_indices=q['cell_indices'], interval=dict(start_mm=q['a'], end_mm=q['b'], index=i,
                    predicted_count=len(intervals)), process_reason='TRAIN_MODEL_EVENT_CELL_SHOULD_WELD_INTERVAL',
                process_applicability='MODEL_PREDICTION_NOT_35MM_CERTIFICATE'))
        normalized, normalization = primitive.normalize(raw, packet)
        result, selection = select(normalized, packet)
        self.last_diagnostics = dict(cell_probability=probability.tolist(), generated_intervals=intervals,
            normalized_proposals=normalized, normalization=normalization, selection=selection)
        self.last_timing = dict(cpu_seconds=time.perf_counter()-start, event_cells=len(x),
            generated_intervals=len(raw), selected_intervals=len(result), no_prior_stopping_gate=True,
            classifier_features=len(NAMES), no_fixed_weld_count_or_minimum_weld_or_gap=True)
        return result


def rules_native_slice(packet, item):
    return competition.original.native_slice(packet['candidates'][item['candidate']]['points'], item['a'], item['b'])


def decode(features, packet, model):
    old = publication._fallback; publication._fallback = corner.fallback
    try:
        result = exporter.decode(features, packet, model)
    finally:
        publication._fallback = old
    result['event_interval_model_v304'] = model.last_diagnostics
    result['rule_algorithm'] = 'QUALIFIED_STRUCTURAL_EVENT_SHOULD_WELD_INTERVAL_GENERATION'
    return result


def input_curves(record, model):
    features, packet = prior.input_curves(record)
    step = record.get('STEP'); step = step.get('path') if isinstance(step, dict) else step
    inventory = build_geometry(packet, prior.cad.load_material(step) if step else None)
    model.backbone.predict(features, packet)
    x, graph_data = graph(features, packet, inventory, model.backbone.last_states)
    return features, dict(packet, event_interval_graph_v304=graph_data,
        event_interval_observations_v304=x.tolist())
