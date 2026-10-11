"""Offline full-parent D3 compilation with conservative endpoint rejection.

Every prediction row, reference column and original assignment cost survives.
For all-line pairs, a positive D3 count requires at least one child whose two
endpoints pass the original 3 mm test in some allowed whole correspondence.
The broad phase accepts ANY child pairing and either orientation, including
pairings the final scorer forbids. This preserves partial-parent matches.
Matching arc kinds and closed contours go through the original scorer without
an endpoint rejection, so resampled reversal and arbitrary circle anchors are
never approximated. Valid frozen parents with no possible child kind/type
match retain their complete false-positive counts and unavailable cost rows;
their sampled geometry need not be materialized for label-side matching.
This is label-side training computation, not inference.
"""
import numpy as np

from .server_offline_parent_set_reference import compile_reference_table, evaluate_reference_subset
from .sjg_native_d3_adapter_1056 import native_targets
from .world_set_evaluation_285 import reference_segment


def endpoint_candidates(decoded, truth, references):
    """Only reject pairs with a proven zero count under the existing D3 gate."""
    parents = decoded['parents']
    structural = np.array([p['child_count'] == truth['original_child_count']
                           and p['is_closed'] == truth['is_closed'] for p in parents], bool)
    if not structural.any() or truth['is_closed']:
        return structural
    keys = {(c['original_kind'], c['weld_type']) for c in truth['children']}
    keep = np.zeros(len(parents), bool)
    starts, ends, owners, types = [], [], [], []
    for i in np.flatnonzero(structural):
        parent = parents[int(i)]
        if any(s['kind'] != 'line' and (s['kind'], s['weld_type']) in keys
               for s in parent['segments']):
            keep[i] = True
            continue
        for segment in parent['segments']:
            if segment['kind'] != 'line' or (segment['kind'], segment['weld_type']) not in keys:
                continue
            points = np.asarray(segment['points'], np.float64)
            starts.append(points[0]); ends.append(points[-1])
            owners.append(i); types.append(segment['weld_type'])
    if not owners:
        return keep
    starts, ends = np.asarray(starts), np.asarray(ends)
    owners, types = np.asarray(owners, np.int64), np.asarray(types)
    for child, ref in zip(truth['children'], references):
        if child['original_kind'] != 'line':
            continue
        points = np.asarray(ref['points'], np.float64)
        relevant = np.flatnonzero(types == child['weld_type'])
        if not len(relevant):
            continue
        a, b = starts[relevant], ends[relevant]
        # Identical float64 subtraction, sum-of-squares and sqrt to child_error.
        # No spatial radius, rounding tolerance or alternate acceptance gate.
        forward = ((np.linalg.norm(a - points[0], axis=1) <= 3.)
                   & (np.linalg.norm(b - points[-1], axis=1) <= 3.))
        reverse = ((np.linalg.norm(b - points[0], axis=1) <= 3.)
                   & (np.linalg.norm(a - points[-1], axis=1) <= 3.))
        keep[owners[relevant[forward | reverse]]] = True
    return keep


def kind_candidates(parents, truth):
    """A structural/type mismatch cannot contribute any original D3 child TP."""
    keys = {(c['original_kind'], c['weld_type']) for c in truth['children']}
    return np.array([p['child_count'] == truth['original_child_count']
                     and p['is_closed'] == truth['is_closed']
                     and any((s['kind'], s['weld_type']) in keys for s in p['segments'])
                     for p in parents], bool)


def compile_exact_endpoint_reference(parents, target, decode_subset, batch_size=1024,
                                     direction_semantics=None, progress=None):
    """Compile all original costs, then perform ONE full global assignment."""
    if batch_size < 1:
        raise ValueError('Positive full-parent chunk size required')
    truth = native_targets(target)
    if sum(len(p['children']) for p in truth['parents']) != target['original_children']:
        raise ValueError('All original reference children are required')
    directions = [False] * len(truth['parents']) if direction_semantics is None else list(direction_semantics)
    if len(directions) != len(truth['parents']) or any(v not in (None, False, True) for v in directions):
        raise ValueError('An explicit policy per reference parent is required')
    references = [[reference_segment(c) for c in t['children']] for t in truth['parents']]
    rows, columns = len(parents), len(truth['parents'])
    counts = np.asarray([p['child_count'] for p in parents], np.int64)
    if len(counts) and (counts <= 0).any():
        raise ValueError('Positive complete parent counts required')
    if len(counts) and 9 * counts.max() >= 500_000:
        raise ValueError('Parent too large to recover exact D3 count from cost')
    if any(len(p['segments']) != p['child_count'] for p in parents):
        raise ValueError('Frozen complete parent ownership is malformed')
    costs = np.full((rows, columns), 1e12, np.float64)
    tp = np.zeros((rows, columns), np.int64)
    checked_pairs, scored_pairs, sampled_parents = 0, 0, 0
    for start in range(0, rows, batch_size):
        stop = min(start + batch_size, rows)
        chunk = parents[start:stop]
        possible = np.zeros(stop-start, bool)
        for t in truth['parents']:
            possible |= kind_candidates(chunk, t)
        pending = np.flatnonzero(possible)
        checked_pairs += (stop-start)*columns
        if not len(pending):
            if progress:
                progress(dict(completed_parents=stop,total_parents=rows,checked_pairs=checked_pairs,
                              original_scored_pairs=scored_pairs,sampled_parents=sampled_parents))
            continue
        decoded = decode_subset(parents, start+pending)
        sampled_parents += len(pending)
        chunk_counts = np.asarray([p['child_count'] for p in decoded['parents']], np.int64)
        if (decoded['invalid_predictions'] or len(chunk_counts) != len(pending)
                or not np.array_equal(chunk_counts, counts[start+pending])
                or decoded['total_prediction_children'] != int(chunk_counts.sum())
                or any(len(p['segments']) != p['child_count'] for p in decoded['parents'])):
            raise ValueError('Decoder changed complete prediction ownership')
        for ti, (t, refs) in enumerate(zip(truth['parents'], references)):
            indices = np.flatnonzero(endpoint_candidates(decoded, t, refs))
            scored_pairs += len(indices)
            if not len(indices):
                continue
            selected = [decoded['parents'][int(i)] for i in indices]
            part = dict(decoded, parents=selected,
                        total_prediction_children=sum(p['child_count'] for p in selected))
            single_target = dict(target, parents=[target['parents'][ti]],
                                 original_children=len(t['children']))
            table, _ = compile_reference_table(part, single_target, [directions[ti]])
            if int(table['invalid_children']) or not np.array_equal(table['child_counts'], chunk_counts[indices]):
                raise ValueError('Original scorer changed prediction ownership')
            costs[start + pending[indices], ti] = table['costs'][:, 0]
            tp[start + pending[indices], ti] = table['tp'][:, 0]
        if progress:
            progress(dict(completed_parents=stop, total_parents=rows,
                          checked_pairs=checked_pairs, original_scored_pairs=scored_pairs,
                          sampled_parents=sampled_parents))
    result = dict(costs=costs, tp=tp, child_counts=counts, invalid_children=np.asarray(0))
    score = evaluate_reference_subset(result, np.arange(rows))
    overall = dict(G=target['original_children'], **{k: score[k] for k in ('predicted', 'TP', 'FP')})
    return result, dict(categories=dict(overall=overall), costs_from_original_D3=True,
        global_assignment_not_sum_of_chunk_TP=True, all_prediction_and_reference_rows_retained=True,
        checked_pairs=checked_pairs, original_scored_pairs=scored_pairs,
        proven_unavailable_pairs=checked_pairs-scored_pairs,
        sampled_parents=sampled_parents,unsampled_unavailable_parent_rows=rows-sampled_parents,
        arcs_and_closed_contours_use_original_scorer=True,
        model_accuracy_or_full_contract_not_claimed=True)
