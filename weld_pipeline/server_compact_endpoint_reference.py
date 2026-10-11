"""Exact offline D3 costs using original compact line endpoints before sampling.

All rows, reference columns, false-positive counts and global matching remain.
Only provably unavailable label-side pairs avoid geometry materialization.
The frozen runtime line sampler sets its first/last points to these float64
endpoints exactly. Matching arcs and closed parents use the original scorer.
This module does not filter inference candidates or change model inputs.
"""
import numpy as np

from .server_offline_parent_set_reference import compile_reference_table, evaluate_reference_subset
from .server_predicted_parent_pool import decoded_subset as original_decoder
from .sjg_native_d3_adapter_1056 import native_targets
from .world_set_evaluation_285 import reference_segment


class CompactEndpointIndex:
    """Build each chunk's measured endpoints once, shared by all references."""
    def __init__(self, parents, decoder):
        self.counts = np.asarray([p['child_count'] for p in parents], np.int64)
        self.closed = np.asarray([p['is_closed'] for p in parents], bool)
        lines, other = {}, {}
        self.unsafe = []
        for owner, parent in enumerate(parents):
            for segment in parent['segments']:
                key = (parent['child_count'], parent['is_closed'], segment['weld_type'])
                if segment['kind'] != 'line':
                    other.setdefault((*key, segment['kind']), []).append(owner)
                    continue
                if decoder is original_decoder:
                    # The original sampler ignores any stored points and
                    # rebuilds line samples from the declared endpoints.
                    ends = np.asarray(segment['endpoints_mm'], np.float64)
                    if ends.shape != (2, 3):
                        raise ValueError('Malformed compact line endpoints')
                elif 'points' in segment:
                    points = np.asarray(segment['points'], np.float64)
                    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
                        raise ValueError('Malformed sampled line endpoints')
                    ends = points[[0, -1]]
                else:
                    raise ValueError('Compact endpoint rejection requires the original frozen runtime decoder')
                values = lines.setdefault(key, [[], []])
                values[0].append(ends); values[1].append(owner)
        self.lines = {}
        for key, (ends, owners) in lines.items():
            ends, owners = np.asarray(ends), np.asarray(owners, np.int64)
            # Do not bypass the original decoder's invalid-line or sample-
            # capacity errors. Such rows still undergo original sampling.
            lengths = np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1)
            unsafe = (~np.isfinite(ends).all(axis=(1, 2)) | (lengths <= 0)
                      | (np.ceil(lengths) + 1 > 1_000_001))
            self.unsafe.extend(owners[unsafe].tolist())
            self.lines[key] = (ends, owners)
        self.other = {key: np.asarray(owners, np.int64) for key, owners in other.items()}

    def candidates(self, truth, references):
        count, closed = truth['original_child_count'], truth['is_closed']
        structural = (self.counts == count) & (self.closed == closed)
        if closed or not structural.any():
            return structural
        keep = np.zeros(len(self.counts), bool)
        if self.unsafe:
            keep[np.asarray(self.unsafe, np.int64)] = True
            keep &= structural
        for child, reference in zip(truth['children'], references):
            key = (count, closed, child['weld_type'])
            kind = child['original_kind']
            if kind != 'line':
                owners = self.other.get((*key, kind))
                if owners is not None:
                    keep[owners] = True
                continue
            values = self.lines.get(key)
            if values is None:
                continue
            ends, owners = values
            points = np.asarray(reference['points'], np.float64)
            a, b = ends[:, 0], ends[:, 1]
            forward = ((np.linalg.norm(a - points[0], axis=1) <= 3.)
                       & (np.linalg.norm(b - points[-1], axis=1) <= 3.))
            reverse = ((np.linalg.norm(b - points[0], axis=1) <= 3.)
                       & (np.linalg.norm(a - points[-1], axis=1) <= 3.))
            # ANY child pairing and either orientation is a conservative
            # prerequisite; only original D3 decides legal correspondence.
            keep[owners[forward | reverse]] = True
        return keep


def compile_compact_endpoint_reference(parents, target, decode_subset, batch_size=1024,
                                      direction_semantics=None, progress=None):
    if batch_size < 1:
        raise ValueError('Positive full-parent chunk size required')
    truth = native_targets(target)
    if sum(len(p['children']) for p in truth['parents']) != target['original_children']:
        raise ValueError('All original reference children are required')
    directions = [False] * len(truth['parents']) if direction_semantics is None else list(direction_semantics)
    if len(directions) != len(truth['parents']) or any(v not in (None, False, True) for v in directions):
        raise ValueError('An explicit policy per reference parent is required')
    rows, columns = len(parents), len(truth['parents'])
    counts = np.asarray([p['child_count'] for p in parents], np.int64)
    if len(counts) and ((counts <= 0).any() or 9 * counts.max() >= 500_000):
        raise ValueError('Complete parent counts exceed the original D3 count contract')
    if any(len(p['segments']) != p['child_count'] for p in parents):
        raise ValueError('Frozen complete parent ownership is malformed')
    refs = [[reference_segment(c) for c in t['children']] for t in truth['parents']]
    costs = np.full((rows, columns), 1e12, np.float64)
    tp = np.zeros((rows, columns), np.int64)
    checked_pairs, scored_pairs, sampled_parents = 0, 0, 0
    for start in range(0, rows, batch_size):
        stop = min(start + batch_size, rows)
        index = CompactEndpointIndex(parents[start:stop], decode_subset)
        possible = [index.candidates(t, r) for t, r in zip(truth['parents'], refs)]
        union = np.zeros(stop-start, bool)
        for mask in possible:
            union |= mask
        pending = np.flatnonzero(union)
        checked_pairs += (stop-start)*columns
        if len(pending):
            decoded = decode_subset(parents, start+pending)
            sampled_parents += len(pending)
            decoded_counts = np.asarray([p['child_count'] for p in decoded['parents']], np.int64)
            if (decoded['invalid_predictions'] or len(decoded_counts) != len(pending)
                    or not np.array_equal(decoded_counts, counts[start+pending])
                    or decoded['total_prediction_children'] != int(decoded_counts.sum())
                    or any(len(p['segments']) != p['child_count'] for p in decoded['parents'])):
                raise ValueError('Decoder changed complete prediction ownership')
            for ti, mask in enumerate(possible):
                selected_ids = np.flatnonzero(mask[pending])
                if not len(selected_ids):
                    continue
                selected = [decoded['parents'][int(i)] for i in selected_ids]
                part = dict(decoded, parents=selected,
                            total_prediction_children=sum(p['child_count'] for p in selected))
                single_target = dict(target, parents=[target['parents'][ti]],
                                     original_children=len(truth['parents'][ti]['children']))
                table, _ = compile_reference_table(part, single_target, [directions[ti]])
                if (int(table['invalid_children'])
                        or not np.array_equal(table['child_counts'], decoded_counts[selected_ids])):
                    raise ValueError('Original scorer changed prediction ownership')
                costs[start+pending[selected_ids], ti] = table['costs'][:, 0]
                tp[start+pending[selected_ids], ti] = table['tp'][:, 0]
                scored_pairs += len(selected_ids)
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
        proven_unavailable_pairs=checked_pairs-scored_pairs, sampled_parents=sampled_parents,
        unsampled_unavailable_parent_rows=rows-sampled_parents,
        arcs_and_closed_contours_use_original_scorer=True,
        compact_endpoints_checked_before_sampling=True, model_accuracy_or_full_contract_not_claimed=True)
