"""Finite-source feasibility followed by exact compatible interval selection.

Continuation is not endpoint confidence. Use the actually executed start/end
pointer support as well; do not add a length reward or invent should-weld facts.
Material witnesses, unknown side and process status remain separate evidence.
"""
import math
import time
from functools import lru_cache
import numpy as np
from . import curveonly_01a1ac7c_source_selection_v56 as provenance
from . import curveonly_01a1ac7c_native_reference_v154 as reference
from .curveonly_01a1ac7c_interval_decode_v16 import native_slice

TOL = reference.PROJECTION_NUMERIC_MM


def executed_support(states):
    """Associate every original code with its real autoregressive execution."""
    records = {}
    for state in states:
        items = []
        for direction in ('forward_steps', 'reverse_steps'):
            for step in state[direction]:
                if not step['continue_selected']:
                    continue
                full = step['termination'] == 'FULL_CYCLE'
                if not full and step['predicted_end_mm'] <= step['predicted_start_mm']:
                    continue
                c = float(step['continue_probability'])
                if full:
                    support = float(step['full_cycle_probability'])
                    fields = dict(full_cycle_probability=support)
                else:
                    a, b = float(step['start_peak_probability']), float(step['end_peak_probability'])
                    support = math.sqrt(a*b)
                    fields = dict(start_pointer_probability=a, end_pointer_probability=b)
                items.append(dict(direction=direction, step=step['step'],
                    continuation_probability=c, endpoint_or_cycle_support=support,
                    utility=c*support, calibrated_geometry_probability=False, **fields))
        assert len(items) == len(state['codes']), (state['candidate'], len(items), len(state['codes']))
        for index, (code, item) in enumerate(zip(state['codes'], items)):
            assert code['confidence'] == item['continuation_probability']
            records[state['candidate'], index] = item
    return records


def pair_identity(proposal):
    """Only explicitly qualified input evidence establishes a joint pair."""
    pair = proposal.get('joint_material_pair')
    if (proposal.get('joint_material_pair_status') == 'CONFIRMED_INPUT_GEOMETRY'
            and isinstance(pair, (list, tuple)) and len(pair) == 2):
        return tuple(sorted(map(int, pair)))
    return None


def feasibility(proposal, packet, lengths):
    ci = proposal['candidate']
    if not isinstance(ci, int) or not 0 <= ci < len(lengths):
        return dict(feasible=False, reason='INVALID_SOURCE_INDEX'), []
    c = packet['candidates'][ci]
    points = np.asarray(c['points'], float)
    a, b = [float(proposal['interval'][k]) for k in ('start_mm', 'end_mm')]
    closed = np.linalg.norm(points[0]-points[-1]) <= 1e-6
    length = lengths[ci]
    finite = math.isfinite(a) and math.isfinite(b) and b > a
    bounded = finite and a >= -TOL and b-a <= length+TOL and (
        a < length+TOL if closed else b <= length+TOL)
    if not bounded:
        return dict(feasible=False, reason='OUTSIDE_FINITE_NATIVE_RANGE_OR_REPEATED_TRAVERSAL'), []
    expected = native_slice(points, a, b)
    actual = np.asarray(proposal['points'], float)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        return dict(feasible=False, reason='INVALID_NATIVE_GEOMETRY'), []
    delta = float(np.linalg.norm(actual-expected, axis=1).max())
    if delta > TOL:
        return dict(feasible=False, reason='PROPOSAL_DEPARTS_FROM_CERTIFIED_NATIVE_SLICE',
                    native_slice_max_error_mm=delta), []
    spans = provenance.primitive_intervals(ci, a, b, packet, lengths)
    owners = sorted({int(packet['candidates'][p].get('material', -1)) for p, _, _ in spans})
    # Existing local support is not a full joint or process certificate. In
    # particular the carrier-union other_distance metadata contains flags.
    witnesses = []
    for connection in c.get('source_connections', []):
        relation = connection.get('relation', {})
        witnesses.append(dict(kind=relation.get('kind'),
            source_materials=relation.get('source_materials'),
            finite_face_materials=sorted({f['material'] for f in relation.get('faces', [])}),
            radius_mm=relation.get('evidence_radius_mm'),
            should_weld_and_process_continuity=False))
    return dict(feasible=True, reason='EXACT_FINITE_SOURCE_PARAMETER_RANGE',
        native_slice_max_error_mm=delta, original_owner_materials=owners,
        confirmed_joint_pair=pair_identity(proposal),
        joint_pair_status='CONFIRMED_INPUT_GEOMETRY' if pair_identity(proposal) is not None else 'UNKNOWN',
        weld_side=proposal.get('weld_side'),
        weld_side_status='KNOWN' if proposal.get('weld_side') is not None else 'UNKNOWN',
        local_finite_witnesses=witnesses, true_gap_or_process_constraints='UNKNOWN',
        finite_support_not_should_weld=True, native_end_not_automatic_weld_end=True), spans


def overlap(left, right):
    result = []
    for ci, a, b in left:
        for cj, c, d in right:
            if ci == cj:
                lo, hi = max(a, c), min(b, d)
                if hi-lo > provenance.MIN_OVERLAP_MM:
                    result.append(dict(primitive=ci, start_mm=lo, end_mm=hi, overlap_mm=hi-lo))
    return result


def maximum_compatible(weights, adjacency):
    """Exact maximum weight independent set; disconnected parts are separable."""
    n = len(weights)
    assert len(adjacency) == n and all(math.isfinite(w) and w >= 0 for w in weights)
    calls = [0]

    def members(mask):
        while mask:
            bit = mask & -mask
            yield bit.bit_length()-1
            mask ^= bit

    @lru_cache(None)
    def solve(mask):
        calls[0] += 1
        if not mask:
            return 0., ()
        first = mask & -mask
        component, pending = first, first
        while pending:
            bit = pending & -pending
            pending ^= bit
            index = bit.bit_length()-1
            new = adjacency[index] & mask & ~component
            component |= new
            pending |= new
        if component != mask:
            a, aa = solve(component)
            b, bb = solve(mask ^ component)
            return a+b, tuple(sorted(aa+bb))
        ids = list(members(mask))
        vertex = max(ids, key=lambda i: ((adjacency[i] & mask).bit_count(), weights[i], -i))
        bit = 1 << vertex
        without, left = solve(mask ^ bit)
        with_value, right = solve(mask & ~bit & ~adjacency[vertex])
        with_value += weights[vertex]
        right = tuple(sorted(right+(vertex,)))
        if with_value > without or (with_value == without and right < left):
            return with_value, right
        return without, left

    value, selected = solve((1 << n)-1)
    assert all(not (adjacency[i] & sum(1 << j for j in selected)) for i in selected)
    return list(selected), dict(objective=value, exact=True,
        search_subproblems=calls[0], memoized_subproblems=solve.cache_info().currsize,
        no_length_reward=True, no_fixed_interval_count=True)


def select(proposals, packet, states):
    start = time.perf_counter()
    lengths = [float(provenance.cad.arclength(c['points'])[-1]) for c in packet['candidates']]
    support = executed_support(states)
    records, spans, eligible = [], [], []
    for index, proposal in enumerate(proposals):
        feasible, parts = feasibility(proposal, packet, lengths)
        original = proposal.get('finite_representation_provenance', {})
        source = original.get('original_candidate', proposal['candidate'])
        interval = original.get('original_interval', proposal['interval'])
        score = support[source, interval['index']]
        assert score['continuation_probability'] == proposal['confidence']
        records.append(dict(proposal=index, candidate=proposal['candidate'], interval=proposal['interval'],
            feasibility=feasible, model_support=score, original_source_candidate=source,
            original_code_index=interval['index']))
        spans.append(parts)
        if feasible['feasible']:
            eligible.append(index)
    adjacency = [0]*len(eligible)
    edges = []
    for i, old_i in enumerate(eligible):
        a = proposals[old_i]
        for j in range(i):
            old_j = eligible[j]
            b = proposals[old_j]
            if a['weld_type'] != b['weld_type'] or a.get('weld_side') != b.get('weld_side'):
                continue
            pa, pb = pair_identity(a), pair_identity(b)
            if pa is not None and pb is not None and pa != pb:
                continue
            shared = overlap(spans[old_i], spans[old_j])
            if shared:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
                edges.append(dict(proposals=[old_j, old_i], shared_finite_intervals=shared,
                    confirmed_joint_pair=pa if pa == pb else None,
                    joint_identity_status='CONFIRMED_INPUT_GEOMETRY' if pa is not None and pa == pb else 'UNKNOWN_SAME_SOURCE_REPRESENTATION_CONFLICT'))
    weights = [records[i]['model_support']['utility'] for i in eligible]
    chosen, optimization = maximum_compatible(weights, adjacency)
    kept = {eligible[i] for i in chosen}
    order = sorted(kept, key=lambda i: (-proposals[i]['confidence'], i))
    rejected = [dict(records[i], reason='INFEASIBLE_NATIVE_RANGE' if not records[i]['feasibility']['feasible']
                     else 'GLOBAL_COMPATIBLE_INTERVAL_OBJECTIVE') for i in range(len(proposals)) if i not in kept]
    return [proposals[i] for i in order], dict(input_proposals=len(proposals), kept=len(kept),
        kept_proposals=order, rejected=rejected, proposals=records, conflicts=edges,
        optimization=optimization, minimum_overlap_mm=provenance.MIN_OVERLAP_MM,
        source='INPUT_FINITE_PROVENANCE_AND_EXECUTED_ENDPOINT_SUPPORT_GLOBAL_COMPETITION_V289',
        original_points_ranges_confidences_unchanged=True,
        no_candidate_extension_or_length_reward=True, no_NEW_should_weld_or_gap_or_side_certification=True,
        no_labels_names_coordinates_or_models_fitted=True, seconds=time.perf_counter()-start)


class RuleSelection:
    def __init__(self, base):
        self.base, self.selection, self.last_timing = base, None, None

    def predict(self, features, packet):
        proposals = self.base.predict(features, packet)
        result, self.selection = select(proposals, packet, self.base.last_states)
        self.last_timing = dict(self.base.last_timing, finite_source_selection=self.selection)
        return result
