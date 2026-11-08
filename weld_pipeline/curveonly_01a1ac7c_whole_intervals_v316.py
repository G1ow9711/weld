"""All native event-pair ranges, scored as complete weld units.

Qualified geometry/old neural hypotheses supply possible endpoints. Frozen
local models are soft observations. A new whole-range head decides validity;
local foreground thresholds and old stopping decisions never gate generation.
"""
import math
import sys
import time
from functools import lru_cache
import numpy as np
from . import curveonly_01a1ac7c_source_context_model_v312 as profile

base = profile.previous
provenance = base.competition.provenance
TOL = base.TOL
SCALARS = ('log_source_length', 'log_range_length', 'source_fraction', 'native_closed',
    'full_native_cycle', 'endpoint_closed', 'source_circle', 'source_STEP',
    'STEP_pose_passed', 'source_path_part_count', 'interior_native_joins',
    'nominal_clearance_fraction', 'end_clearance_min_fraction', 'end_clearance_max_fraction',
    'chord_arclength_ratio', 'abs_height_change_per_mm', 'endpoint_tangent_dot',
    'turn_per_mm', 'turn_radians', 'cell_support_mean', 'cell_support_std',
    'cell_support_min', 'cell_support_max', 'start_inside_support', 'end_inside_support',
    'start_outside_support', 'end_outside_support', 'start_outside_available', 'end_outside_available',
    'interior_unit_cut_max', 'interior_unit_cut_sum', 'interior_unit_cut_mean')
NAMES = tuple(prefix+':'+n for prefix in ('span_mean', 'span_std') for n in base.NAMES[:130])+tuple(
    side+':'+n for side in ('start_event', 'end_event') for n in base.CONTEXT)+profile.EXTRA_NAMES+SCALARS
SETTINGS = dict(max_iter=120, max_leaf_nodes=31, max_depth=6, min_samples_leaf=15,
    learning_rate=.1, l2_regularization=1., early_stopping=False, random_state=20261106)


def pairs(data, ci):
    lo, hi = data['offsets'][ci:ci+2]; cells = data['cells'][lo:hi]
    nodes = np.array([c['a'] for c in cells]+[cells[-1]['b']], float)
    closed = cells[0]['closed']; m = len(cells)
    if closed:
        a, b = np.meshgrid(np.arange(m), np.arange(m), indexing='ij')
        keep = a != b; a, b = a[keep], b[keep]; b = np.where(b > a, b, b+m)
        a, b = np.r_[a, 0], np.r_[b, m]
    else: a, b = np.triu_indices(m+1, 1)
    grid = np.r_[nodes, nodes[1:]+nodes[-1]] if closed else nodes
    return np.column_stack([a, b]).astype(np.int32), nodes, grid


def extrema(values, a, b, maximum=False, empty=0.):
    """Sparse range query, no scanning/re-sampling per generated interval."""
    op = np.maximum if maximum else np.minimum
    levels = [np.asarray(values, float)]
    width = 1
    while width*2 <= len(values):
        levels.append(op(levels[-1][:-width], levels[-1][width:])); width *= 2
    length = b-a; result = np.full(len(a), empty, float)
    for power in np.unique(np.floor(np.log2(np.maximum(1, length))).astype(int)):
        use = (length > 0)&(length >= 2**power)&(length < 2**(power+1))
        result[use] = op(levels[power][a[use]], levels[power][b[use]-2**power])
    return result


def fields(model, physical, nodes, data, states):
    context = profile.source_context(data, states)
    cell = model.head.predict_proba(np.column_stack([physical, context]))[:, list(model.head.classes_).index(1)]
    cuts = model.cut_head.predict_proba(nodes)[:, list(model.cut_head.classes_).index(1)]
    return cell, cuts, context


def features(packet, data, physical, cell_probability, cut_probability, context, ci):
    indexes, original, grid = pairs(data, ci); a, b = indexes.T
    lo, hi = data['offsets'][ci:ci+2]; candidate = packet['candidates'][ci]
    closed = data['cells'][lo]['closed']; length = original[-1]
    x = np.asarray(physical[lo:hi, :130], float)
    p = np.asarray(cell_probability[lo:hi]); q = np.asarray(cut_probability[lo:hi])
    n = len(p); widths = np.diff(original)
    start_context = physical[lo:hi, 130:144]
    event_context = np.vstack([start_context, physical[hi-1, 144:158]])
    if closed:
        x = np.vstack([x, x]); p = np.r_[p, p]; widths = np.r_[widths, widths]
        q = np.r_[q, q, q[0]]; event_context = np.vstack([event_context[:-1], event_context])
    else: q = np.r_[q, 0.]
    aa, bb = grid[a], grid[b]; width = bb-aa
    prefix = np.vstack([np.zeros(130), np.cumsum(x*widths[:, None], axis=0)])
    second = np.vstack([np.zeros(130), np.cumsum(x*x*widths[:, None], axis=0)])
    mean = (prefix[b]-prefix[a])/width[:, None]
    std = np.sqrt(np.maximum(0., (second[b]-second[a])/width[:, None]-mean*mean))
    points = np.asarray(candidate['points']); station = base.prior.arclength(points)
    xyz = np.column_stack([np.interp(original, station, points[:, k]) for k in range(3)])
    if closed: xyz = np.vstack([xyz[:-1], xyz])
    delta = xyz[b]-xyz[a]; chord = np.linalg.norm(delta, axis=1)
    vectors = np.diff(points, axis=0); norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 0; s = station[:-1][valid]; directions = vectors[valid]/norms[valid, None]
    at = np.clip(np.searchsorted(s, original, side='right')-1, 0, len(s)-1)
    tangent = directions[at]
    if closed: tangent = np.vstack([tangent[:-1], tangent])
    angles = np.arccos(np.clip((directions[:-1]*directions[1:]).sum(1), -1., 1.))
    turns = np.r_[0., np.cumsum(angles)]
    turn_at = np.interp(original, s, turns)
    if closed: turn_at = np.r_[turn_at, turn_at[1:]+turn_at[-1]]
    turning = turn_at[b]-turn_at[a]
    joins = [part[k] for part in candidate.get('source_material_path', []) for k in ('start_mm', 'end_mm')]
    joins = np.array(sorted(set(joins)))
    if closed and len(joins): joins = np.r_[joins, joins+length]
    count = np.searchsorted(joins, bb-TOL)-np.searchsorted(joins, aa+TOL, side='right')
    count = np.maximum(0, count)
    clearance = np.zeros(len(a))
    for part in candidate.get('nominal_clearance_intervals', []):
        for shift in ([0., length] if closed else [0.]):
            clearance += np.maximum(0., np.minimum(bb, part['end_mm']+shift)-np.maximum(aa, part['start_mm']+shift))
    pp = np.r_[0., np.cumsum(p*widths)]; pp2 = np.r_[0., np.cumsum(p*p*widths)]
    pm = (pp[b]-pp[a])/width; ps = np.sqrt(np.maximum(0., (pp2[b]-pp2[a])/width-pm*pm))
    pq = np.r_[0., np.cumsum(q)]
    inside_n = np.maximum(0, b-a-1); internal_sum = pq[b]-pq[a+1]
    outside_left = p[(a-1) % n] if closed else p[np.maximum(0, a-1)]
    outside_right = p[b % n] if closed else p[np.minimum(n-1, b)]
    left_ok = np.ones(len(a), bool) if closed else a > 0
    right_ok = np.ones(len(a), bool) if closed else b < n
    source_parts = max(1, len(candidate.get('source_material_path', [])))
    end1, end2 = (np.zeros(len(a)), np.zeros(len(a))) if closed else (aa/length, (length-bb)/length)
    base_axis = np.asarray(packet['base_axis'], float)
    scalars = np.column_stack([np.full(len(a), np.log1p(length)), np.log1p(width), width/length,
        np.full(len(a), closed), closed&(np.abs(width-length) <= TOL), chord < .1,
        np.full(len(a), candidate.get('circle') is not None), np.full(len(a), bool(candidate.get('STEP'))),
        np.full(len(a), bool(packet.get('pose', {}).get('passed'))), np.full(len(a), source_parts), count,
        clearance/width, np.minimum(end1, end2), np.maximum(end1, end2), chord/width,
        np.abs(delta@base_axis)/width, (tangent[a]*tangent[b]).sum(1), turning/width, turning,
        pm, ps, extrema(p, a, b), extrema(p, a, b, True), p[a], p[b-1],
        outside_left, outside_right, left_ok, right_ok,
        extrema(q, a+1, b, True), internal_sum, internal_sum/np.maximum(1, inside_n)])
    result = np.column_stack([mean, std, event_context[a], event_context[b],
        np.broadcast_to(context[lo], (len(a), 6)), scalars]).astype(np.float32)
    assert result.shape == (len(indexes), len(NAMES)) and np.isfinite(result).all()
    return result, indexes, np.column_stack([aa, bb])


def weighted_intervals(ids, bounds, weights):
    """Same conflict objective on a certified linear interval component."""
    order = sorted(ids, key=lambda i:(bounds[i][1], bounds[i][0], i))
    ends = np.array([bounds[i][1] for i in order]); best = np.zeros(len(order)+1)
    count = np.zeros(len(order)+1, int); take = np.zeros(len(order), bool); previous = []
    for k, i in enumerate(order):
        j = int(np.searchsorted(ends[:k], bounds[i][0]+provenance.MIN_OVERLAP_MM, side='right'))
        previous.append(j); score = best[j]+weights[i]; number = count[j]+1
        if score > best[k]+1e-12 or abs(score-best[k]) <= 1e-12 and number < count[k]:
            best[k+1], count[k+1], take[k] = score, number, True
        else: best[k+1], count[k+1] = best[k], count[k]
    kept, k = [], len(order)
    while k:
        if take[k-1]: kept.append(order[k-1]); k = previous[k-1]
        else: k -= 1
    return float(best[-1]), tuple(sorted(kept))


def maximum_compatible(weights, adjacency, linear_solver=None):
    """Exact graph objective with fast dense-clique and component cases."""
    sys.setrecursionlimit(max(sys.getrecursionlimit(), len(weights)+200))
    calls = [0]
    @lru_cache(None)
    def solve(mask):
        calls[0] += 1
        if not mask: return 0., ()
        bit = mask & -mask; component, pending = bit, bit
        while pending:
            v = pending & -pending; pending ^= v; i = v.bit_length()-1
            new = adjacency[i]&mask&~component; component |= new; pending |= new
        if component != mask:
            a, aa = solve(component); b, bb = solve(mask^component)
            return a+b, tuple(sorted(aa+bb))
        ids, rest = [], mask
        while rest:
            v = rest & -rest; rest ^= v; ids.append(v.bit_length()-1)
        if all((adjacency[i]&mask).bit_count() == len(ids)-1 for i in ids):
            i = max(ids, key=lambda i:(weights[i], -i)); return weights[i], (i,)
        if linear_solver is not None:
            answer = linear_solver(ids, mask, weights, adjacency)
            if answer is not None: return answer
        vertex = max(ids, key=lambda i:((adjacency[i]&mask).bit_count(), weights[i], -i))
        bit = 1 << vertex; a, aa = solve(mask^bit)
        b, bb = solve(mask & ~bit & ~adjacency[vertex]); b += weights[vertex]; bb = tuple(sorted(bb+(vertex,)))
        if b > a+1e-12 or abs(b-a) <= 1e-12 and (len(bb), bb) < (len(aa), aa): return b, bb
        return a, aa
    value, ids = solve((1 << len(weights))-1)
    return list(ids), dict(objective=value, objective_unit='complete_interval_validity_log_odds',
        exact_to_float_arithmetic=True, arithmetic_tie=1e-12, subproblems=calls[0],
        no_fixed_record_count_or_length_bonus=True, objective_is_decoding_surrogate_not_TP=True)


def linear_components(packet, lengths, records, footprints):
    """A path embedding is used only after its whole graph matches exactly."""
    cache = {}
    def mapping(ci, offset=0., sign=1., seen=()):
        assert ci not in seen
        path = packet['candidates'][ci].get('source_material_path')
        if not path: return [(ci, offset, sign)]
        out = []
        for p in path:
            child = p['curve']; origin = p['end_mm'] if p['reversed'] else p['start_mm']
            out += mapping(child, offset+sign*origin, sign*(-1 if p['reversed'] else 1), seen+(ci,))
        return out
    def solve(ids, mask, weights, adjacency):
        primitives = {ci for i in ids for ci, a, b in footprints[i]}
        candidates = sorted({records[i]['candidate'] for i in ids}, key=lambda ci:-lengths[ci])
        for ci in candidates:
            if ci not in cache:
                entries = mapping(ci); keys = [p for p, offset, sign in entries]
                cache[ci] = {p:(offset, sign) for p, offset, sign in entries} if len(keys) == len(set(keys)) else {}
            table = cache[ci]
            if not primitives <= table.keys(): continue
            bounds = {}
            for i in ids:
                q = [table[p][0]+table[p][1]*x for p, a, b in footprints[i] for x in (a, b)]
                if not q: break
                bounds[i] = (min(q), max(q))
            if len(bounds) != len(ids) or any(b-a <= provenance.MIN_OVERLAP_MM for a, b in bounds.values()): continue
            expected = {i:0 for i in ids}; active = []
            for a, b, i in sorted((a, b, i) for i, (a, b) in bounds.items()):
                active = [(end, j, start) for end, j, start in active if end-a > provenance.MIN_OVERLAP_MM]
                for end, j, start in active:
                    if min(b, end)-max(a, start) > provenance.MIN_OVERLAP_MM:
                        expected[i] |= 1 << j; expected[j] |= 1 << i
                active.append((b, i, a))
            if all(expected[i] == adjacency[i]&mask for i in ids):
                return weighted_intervals(ids, bounds, weights)
        return None
    return solve


def select(proposals, packet):
    lengths = [float(base.prior.arclength(c['points'])[-1]) for c in packet['candidates']]
    unique, records, footprints = {}, [], []
    for item in proposals:
        ci, a, b = item['candidate'], item['a'], item['b']
        parts = provenance.primitive_intervals(ci, a, b, packet, lengths)
        # Literal source parameter equivalence only; no3mm geometry clustering.
        key = tuple(sorted((int(i), round(x/TOL), round(y/TOL)) for i, x, y in parts))
        if not key: key = ('UNMAPPED', ci, a, b)
        if key in unique:
            index = unique[key]
            if item['confidence'] > records[index]['confidence']: records[index] = item; footprints[index] = parts
        else:
            unique[key] = len(records); records.append(item); footprints.append(parts)
    adjacency = [0]*len(records); by_primitive = {}; edges = 0
    for i, parts in enumerate(footprints):
        for primitive, a, b in parts: by_primitive.setdefault(primitive, []).append((a, b, i))
    for spans in by_primitive.values():
        active = []
        for a, b, i in sorted(spans):
            active = [(end, j, begin) for end, j, begin in active if end-a > provenance.MIN_OVERLAP_MM]
            for end, j, begin in active:
                if i != j and min(end, b)-max(a, begin) > provenance.MIN_OVERLAP_MM:
                    if not adjacency[i]&(1 << j): edges += 1
                    adjacency[i] |= 1 << j; adjacency[j] |= 1 << i
            active.append((b, i, a))
    weights = [math.log(min(1-1e-12, r['confidence'])/max(1e-12, 1-r['confidence'])) for r in records]
    kept, optimization = maximum_compatible(weights, adjacency, linear_components(packet, lengths, records, footprints))
    selected = []
    for i in kept:
        r = records[i]
        selected.append(dict(candidate=r['candidate'], confidence=r['confidence'], weld_type='horizontal',
            interval=dict(start_mm=r['a'], end_mm=r['b'], index=r['proposal'], predicted_count=len(kept)),
            points=base.competition.original.native_slice(packet['candidates'][r['candidate']]['points'], r['a'], r['b']).tolist(),
            whole_event_range_model=True))
    selected, normalization = base.primitive.normalize(selected, packet)
    return selected, dict(generated_above_threshold=len(proposals), unique_native_ranges=len(records),
        conflicts=edges, kept=len(kept), optimization=optimization, normalization=normalization,
        source_identity_only_not_new_physical_joint_certificate=True)


class WholeIntervalModel:
    def __init__(self, profile_model, classifier):
        self.profile_model, self.classifier = profile_model, classifier
        self.last_timing = self.last_diagnostics = None
        self.last_probabilities = None
        self.progress = None

    def predict(self, observed, packet):
        started = time.perf_counter(); data = packet['event_interval_graph_v304']
        x = np.asarray(packet['event_interval_observations_v304'], np.float32)
        nodes = np.asarray(packet['event_node_observations_v308'], np.float32)
        p, q, context = fields(self.profile_model, x, nodes, data, self.profile_model.backbone.last_states)
        proposals, summaries, total, probabilities = [], [], 0, []
        for ci in range(len(packet['candidates'])):
            f, indexes, ranges = features(packet, data, x, p, q, context, ci)
            probability = self.classifier.predict_proba(f)[:, list(self.classifier.classes_).index(1)]
            probabilities.append(probability.astype(np.float32))
            for i in np.flatnonzero(probability > .5):
                proposals.append(dict(candidate=ci, a=float(ranges[i, 0]), b=float(ranges[i, 1]),
                    confidence=float(probability[i]), proposal=total+int(i)))
            summaries.append(dict(candidate=ci, ranges=len(f), above_threshold=int(sum(probability > .5))))
            total += len(f)
            if self.progress: self.progress('CANDIDATE_RANGES', ci, len(f), summaries[-1]['above_threshold'])
        if self.progress: self.progress('GLOBAL_NATIVE_SELECTION_BEGIN', len(proposals))
        result, selection = select(proposals, packet)
        if self.progress: self.progress('GLOBAL_NATIVE_SELECTION_END', len(result))
        self.last_probabilities = probabilities
        self.last_diagnostics = dict(generated_ranges=total, candidates=summaries, selection=selection,
            no_local_foreground_or_old_stopping_gate=True, fixed_range_count=False,
            new_should_weld_and_unit_boundaries_are_model_predictions_not_physical_certification=True)
        self.last_timing = dict(cpu_seconds=time.perf_counter()-started, features=len(NAMES),
            whole_native_event_pair_generation=True, generated_ranges=total, kept=len(result))
        return result


def decode(features, packet, model):
    old = base.publication._fallback; base.publication._fallback = base.corner.fallback
    try: result = base.exporter.decode(features, packet, model)
    finally: base.publication._fallback = old
    result['whole_event_intervals_v316'] = model.last_diagnostics
    return result


def input_curves(record, model): return profile.input_curves(record, model.profile_model)
