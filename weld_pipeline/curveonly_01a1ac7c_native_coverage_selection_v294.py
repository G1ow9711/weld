"""Integrate model support over finite original arclength, not record count.

The V289 per-record objective could prefer two incomplete halves to an explicit
whole-cycle prediction. Equal-density subdivisions now have equal measure.
This is a decoding surrogate; acceptance still counts complete weld units.
"""
import math
import time
from functools import lru_cache
from . import curveonly_01a1ac7c_rule_selection_v289 as original

feasibility, pair_identity = original.feasibility, original.pair_identity
provenance = original.provenance


def measure(spans):
    groups = {}
    for ci, a, b in spans:
        groups.setdefault(ci, []).append((a, b))
    value = 0.
    for parts in groups.values():
        end = -math.inf
        for a, b in sorted(parts):
            value += max(0., b-max(a, end))
            end = max(end, b)
    return value


def maximum_compatible(weights, adjacency):
    """Exact objective; equally supported geometry prefers fewer representations."""
    assert len(weights) == len(adjacency) and all(math.isfinite(w) and w >= 0 for w in weights)
    calls = [0]
    @lru_cache(None)
    def solve(mask):
        calls[0] += 1
        if not mask:
            return 0., ()
        first = mask & -mask
        component, pending = first, first
        while pending:
            bit = pending & -pending; pending ^= bit
            i = bit.bit_length()-1
            new = adjacency[i] & mask & ~component
            component |= new; pending |= new
        if component != mask:
            a, aa = solve(component); b, bb = solve(mask ^ component)
            return a+b, tuple(sorted(aa+bb))
        ids = []; rest = mask
        while rest:
            bit = rest & -rest; rest ^= bit; ids.append(bit.bit_length()-1)
        vertex = max(ids, key=lambda i: ((adjacency[i] & mask).bit_count(), weights[i], -i))
        bit = 1 << vertex
        a, aa = solve(mask ^ bit)
        b, bb = solve(mask & ~bit & ~adjacency[vertex]); b += weights[vertex]
        bb = tuple(sorted(bb+(vertex,)))
        if b > a or (b == a and (len(bb), bb) < (len(aa), aa)):
            return b, bb
        return a, aa
    value, kept = solve((1 << len(weights))-1)
    chosen = sum(1 << i for i in kept)
    assert all(not (adjacency[i] & chosen) for i in kept)
    return list(kept), dict(objective=value, objective_unit='model_support_times_unique_native_mm',
        exact=True, search_subproblems=calls[0], no_record_count_reward=True,
        no_fixed_interval_count=True, equal_objective_tie='FEWER_REPRESENTATIONS_THEN_INPUT_ORDER',
        no_always_longest=True, new_acceptance_counting_unit=False)


def select(proposals, packet, states):
    start = time.perf_counter()
    # Reuse the exact source feasibility, model-execution association and
    # conflict witnesses. Its old choice is deliberately discarded.
    _, receipt = original.select(proposals, packet, states)
    records = receipt['proposals']
    lengths = [float(provenance.cad.arclength(c['points'])[-1]) for c in packet['candidates']]
    eligible = []
    for i, (proposal, record) in enumerate(zip(proposals, records)):
        q = proposal['interval']
        spans = provenance.primitive_intervals(proposal['candidate'], q['start_mm'], q['end_mm'], packet, lengths)
        native_mm = measure(spans)
        density = record['model_support']['utility']
        record['model_support'] = dict(record['model_support'], per_native_mm_support=density,
            unique_native_support_mm=native_mm, utility=density*native_mm,
            utility_definition='CONTINUATION_TIMES_ENDPOINT_SUPPORT_INTEGRATED_OVER_UNIQUE_NATIVE_ARCLENGTH')
        if record['feasibility']['feasible']:
            eligible.append(i)
    position = {old:i for i, old in enumerate(eligible)}
    adjacency = [0]*len(eligible)
    for edge in receipt['conflicts']:
        a, b = edge['proposals']
        if a in position and b in position:
            i, j = position[a], position[b]
            adjacency[i] |= 1 << j; adjacency[j] |= 1 << i
    chosen, optimization = maximum_compatible([records[i]['model_support']['utility'] for i in eligible], adjacency)
    kept = {eligible[i] for i in chosen}
    order = sorted(kept, key=lambda i: (-proposals[i]['confidence'], i))
    receipt.update(kept=len(kept), kept_proposals=order, optimization=optimization,
        rejected=[dict(records[i], reason='INFEASIBLE_NATIVE_RANGE' if not records[i]['feasibility']['feasible']
            else 'GLOBAL_NATIVE_COVERAGE_INTEGRAL_OBJECTIVE') for i in range(len(proposals)) if i not in kept],
        source='FINITE_SOURCE_NATIVE_COVERAGE_INTEGRAL_V294',
        native_geometric_measure_not_record_count=True,
        new_acceptance_counting_unit=False, seconds=time.perf_counter()-start)
    receipt.pop('no_candidate_extension_or_length_reward', None)
    receipt['no_candidate_extension'] = True
    return [proposals[i] for i in order], receipt


class RuleSelection(original.RuleSelection):
    def predict(self, features, packet):
        proposals = self.base.predict(features, packet)
        result, self.selection = select(proposals, packet, self.base.last_states)
        self.last_timing = dict(self.base.last_timing, finite_source_selection=self.selection)
        return result
