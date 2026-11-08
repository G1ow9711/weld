"""Separate should-weld occupancy from true cuts between adjacent weld units.

V304 binary coverage erased mandatory boundaries. This version learns node cuts
separately and preserves native shape. Numeric ties use the existing1e-7 native
projection arithmetic tolerance, never the3mm acceptance tolerance.
"""
import math
import time
import types
from functools import lru_cache
import numpy as np
from . import curveonly_01a1ac7c_event_interval_model_v304 as previous

TOL, NAMES = previous.TOL, previous.NAMES
HYPERPARAMETERS = previous.HYPERPARAMETERS
CUT_PARAMETERS = dict(HYPERPARAMETERS, min_samples_leaf=10)
NODE_NAMES = tuple(prefix+':'+previous.prior.NAMES[i] for prefix in ('mean', 'absolute_change')
    for i in previous.PHYSICAL)+tuple('node:'+s for s in previous.CONTEXT)+(
    'left_log_width_mm', 'right_log_width_mm', 'closed_native',
    'native_material_path_join', 'native_tangent_turn_rad', 'owner_material_change')
assert len(NODE_NAMES) == 280


def node_features(packet, data, observed):
    x = np.asarray(observed); result = np.zeros((len(x), len(NODE_NAMES)), np.float32)
    owner = previous.events.ownership.owner_function(packet)
    for ci, (lo, hi) in enumerate(zip(data['offsets'][:-1], data['offsets'][1:])):
        if hi == lo: continue
        curve = packet['candidates'][ci]; points = np.asarray(curve['points'], float)
        station = previous.prior.arclength(points); vectors = np.diff(points, axis=0)
        norms = np.linalg.norm(vectors, axis=1); valid = np.flatnonzero(norms > 0)
        joins = [part[k] for part in curve.get('source_material_path', []) for k in ('start_mm', 'end_mm')]
        closed = data['cells'][lo]['closed']
        for right in range(lo, hi):
            left = right-1 if right > lo else (hi-1 if closed else right)
            a, b = x[left], x[right]; cell = data['cells'][right]; q = cell['a']
            leftcell = data['cells'][left]
            before = valid[station[valid+1] <= q+TOL]
            after = valid[station[valid] >= q-TOL]
            il = int(before[-1]) if len(before) else (int(valid[-1]) if closed and right == lo else int(valid[0]))
            ir = int(after[0]) if len(after) else int(valid[-1])
            if not (right == lo and closed) and (station[il+1] < q-TOL or station[ir] > q+TOL):
                # q is inside one native segment rather than a vertex.
                middle = min(len(vectors)-1, max(0, int(np.searchsorted(station, q, side='right')-1)))
                if norms[middle] > 0: il = ir = middle
            cosine = float(np.clip(vectors[il]@vectors[ir]/(norms[il]*norms[ir]), -1., 1.))
            turn = math.acos(cosine)
            ma = owner(ci, (leftcell['a']+leftcell['b'])/2, packet)
            mb = owner(ci, (cell['a']+cell['b'])/2, packet)
            result[right] = np.r_[(a[:130]+b[:130])/2, abs(a[:130]-b[:130]), b[130:144],
                math.log1p(leftcell['b']-leftcell['a']), math.log1p(cell['b']-cell['a']),
                closed, any(abs(v-q) <= TOL for v in joins), turn, ma != mb]
    assert np.isfinite(result).all()
    return result


def train_targets(packet, data, case):
    y, weight, receipt = previous.train_targets(packet, data, case)
    lengths = [float(previous.prior.arclength(c['points'])[-1]) for c in packet['candidates']]
    positive = {}
    for item in case['canonical_choices']:
        for ci, a, b in previous.competition.provenance.primitive_intervals(item['candidate'],
                item['start_mm'], item['end_mm'], packet, lengths):
            positive.setdefault(ci, []).append((a, b, item['truth']))
    identity = np.full(len(y), -1, np.int32)
    for index, item in enumerate(data['cells']):
        if y[index] != 1: continue
        ci, a, b = item['candidate'], item['a'], item['b']; scores = {}
        for pi, lo, hi in previous.competition.provenance.primitive_intervals(ci, a, b, packet, lengths):
            for aa, bb, ti in positive.get(pi, []):
                scores[ti] = scores.get(ti, 0.)+max(0., min(hi, bb)-max(lo, aa))
        for q in case['canonical_choices']:
            if q['candidate'] == ci:
                scores[q['truth']] = max(scores.get(q['truth'], 0.),
                    max(0., min(b, q['end_mm'])-max(a, q['start_mm'])))
        assert scores and max(scores.values()) > 0
        identity[index] = min(scores, key=lambda k:(-scores[k], k))
    cut = np.full(len(y), -1, np.int8); groups = {}; keys = {}
    for lo, hi in zip(data['offsets'][:-1], data['offsets'][1:]):
        if hi == lo: continue
        closed = data['cells'][lo]['closed']
        for right in range(lo, hi):
            if right == lo and not closed: continue
            left = right-1 if right > lo else hi-1
            if y[left] != 1 or y[right] != 1: continue
            a, b = int(identity[left]), int(identity[right]); assert a >= 0 and b >= 0
            label = int(a != b); cut[right] = label
            key = (label, min(a, b), max(a, b)); keys[right] = key
            groups[key] = groups.get(key, 0)+1
    cut_weight = np.zeros(len(y), np.float64)
    counts = {value:sum(key[0] == value for key in groups) for value in (0, 1)}
    for i, key in keys.items(): cut_weight[i] = 1./groups[key]/counts[key[0]]
    receipt.update(distinct_native_truth_ids_per_cell=identity.tolist(),
        node_cut_counts={str(value):int(sum(cut == value)) for value in (-1, 0, 1)},
        continuous_different_true_units_not_binary_union=True,
        no_process_cut_inferred_from_geometry_alone=True)
    return y, weight, cut, cut_weight, receipt


def runs(data, probability, cuts):
    result = []
    for ci, (lo, hi) in enumerate(zip(data['offsets'][:-1], data['offsets'][1:])):
        cells = data['cells'][lo:hi]; p = probability[lo:hi]; groups = []
        for j in range(len(cells)):
            if p[j] <= .5: continue
            if j == 0 or p[j-1] <= .5 or cuts[lo+j] > .5: groups.append([])
            groups[-1].append(j)
        if cells and cells[0]['closed'] and len(groups) > 1 and groups[0][0] == 0 and groups[-1][-1] == len(cells)-1 and cuts[lo] <= .5:
            groups = [groups[-1]+groups[0]]+groups[1:-1]
        for ids in groups:
            a, b = cells[ids[0]]['a'], cells[ids[-1]]['b']
            if b <= a: b += cells[0]['length_mm']
            widths = np.array([cells[j]['b']-cells[j]['a'] for j in ids])
            result.append(dict(candidate=ci, a=a, b=b,
                confidence=float(np.average(p[ids], weights=widths)),
                support_density=float(np.average(2*p[ids]-1, weights=widths)),
                cell_indices=[lo+j for j in ids]))
    return result


def maximum_compatible(weights, adjacency):
    calls = [0]
    @lru_cache(None)
    def solve(mask):
        calls[0] += 1
        if not mask: return 0., ()
        first = mask & -mask; component, pending = first, first
        while pending:
            bit = pending & -pending; pending ^= bit; i = bit.bit_length()-1
            new = adjacency[i] & mask & ~component; component |= new; pending |= new
        if component != mask:
            a, aa = solve(component); b, bb = solve(mask^component)
            return a+b, tuple(sorted(aa+bb))
        ids, rest = [], mask
        while rest:
            bit = rest & -rest; rest ^= bit; ids.append(bit.bit_length()-1)
        vertex = max(ids, key=lambda i:((adjacency[i]&mask).bit_count(), weights[i], -i))
        bit = 1 << vertex; a, aa = solve(mask^bit)
        b, bb = solve(mask & ~bit & ~adjacency[vertex]); b += weights[vertex]; bb = tuple(sorted(bb+(vertex,)))
        if b > a+TOL or abs(b-a) <= TOL and (len(bb), bb) < (len(aa), aa): return b, bb
        return a, aa
    value, selected = solve((1 << len(weights))-1)
    assert all(not(adjacency[i]&sum(1 << j for j in selected)) for i in selected)
    return list(selected), dict(objective=value, search_subproblems=calls[0],
        objective_unit='cell_model_support_times_unique_native_mm',
        native_projection_numeric_tie_mm=TOL, fewer_representations_only_within_numeric_tie=True,
        no_always_longest_or_new_acceptance_tolerance=True)


_competition = types.SimpleNamespace(**dict(previous.competition.__dict__, maximum_compatible=maximum_compatible))
select = types.FunctionType(previous.select.__code__, dict(previous.__dict__, competition=_competition))


class IntervalModel(previous.IntervalModel):
    def __init__(self, backbone=None, head=None, cut_head=None):
        super().__init__(backbone, head); self.cut_head = cut_head; self.fixed_cuts = None

    def predict(self, features, packet):
        start = time.perf_counter(); data = packet['event_interval_graph_v304']
        x = np.asarray(packet['event_interval_observations_v304'], np.float32)
        nodes = np.asarray(packet['event_node_observations_v308'], np.float32)
        if self.fixed_teacher is None:
            probability = self.head.predict_proba(x)[:, list(self.head.classes_).index(1)]
            cut_probability = self.cut_head.predict_proba(nodes)[:, list(self.cut_head.classes_).index(1)]
        else:
            probability = np.where(np.asarray(self.fixed_teacher) == 1, 1., 0.)
            cut_probability = np.where(np.asarray(self.fixed_cuts) == 1, 1., 0.)
        intervals = runs(data, probability, cut_probability); raw = []
        for index, q in enumerate(intervals):
            raw.append(dict(candidate=q['candidate'], points=previous.rules_native_slice(packet, q).tolist(),
                weld_type='horizontal', confidence=q['confidence'], support_density=q['support_density'],
                cell_indices=q['cell_indices'], interval=dict(start_mm=q['a'], end_mm=q['b'], index=index,
                    predicted_count=len(intervals)), process_reason='TRAIN_OCCUPANCY_AND_TRUE_NODE_CUT_MODELS',
                process_applicability='MODEL_PREDICTION_NOT_35MM_CERTIFICATE'))
        normalized, normalization = previous.primitive.normalize(raw, packet)
        result, selection = select(normalized, packet)
        self.last_diagnostics = dict(cell_probability=probability.tolist(), node_cut_probability=cut_probability.tolist(),
            generated_intervals=intervals, normalized_proposals=normalized, normalization=normalization, selection=selection)
        self.last_timing = dict(cpu_seconds=time.perf_counter()-start, event_cells=len(x),
            generated_intervals=len(raw), selected_intervals=len(result), cell_features=len(NAMES), node_features=len(NODE_NAMES),
            distinct_true_cuts_separate_from_occupancy=True, no_R35_stopping_gate=True,
            no_fixed_interval_count_or_minimum_weld_or_gap=True)
        return result


decode = previous.decode


def input_curves(record, model):
    x, packet = previous.input_curves(record, model)
    nodes = node_features(packet, packet['event_interval_graph_v304'], packet['event_interval_observations_v304'])
    return x, dict(packet, event_node_observations_v308=nodes.tolist())
