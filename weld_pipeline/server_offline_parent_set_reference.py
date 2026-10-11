"""TRAIN-only exact reference utility for subsets of a frozen whole-parent pool.

No runtime module may consume this table. It retains the ORIGINAL D3 assignment
costs (including unavailable-pair costs and geometric tie terms), rather than
replacing matching by summed per-parent TP or a different matching objective.
"""
from types import FunctionType
import numpy as np
from scipy.optimize import linear_sum_assignment
from .contract_geometry_d3_409 import score_source
from .sjg_native_d3_adapter_1056 import native_targets


def compile_reference_table(decoded, target, direction_semantics=None):
    truth = native_targets(target)
    if direction_semantics is None:
        direction_semantics = [False] * len(truth['parents'])
    counts = np.asarray([p['child_count'] for p in decoded['parents']], dtype=np.int64)
    if len(counts) and (counts <= 0).any():
        raise ValueError('Positive child counts required')
    # A successful D3 child has <= 3 mm per endpoint plus <= 3 mm Hausdorff.
    # This bound makes its integer TP term recoverable without changing costs.
    if len(counts) and 9 * counts.max() >= 500_000:
        raise ValueError('Parent too large to recover exact D3 count from cost')
    parents = [dict(p, segments=[dict(s, points=np.asarray(s['points'],np.float64))
                                for s in p['segments']]) for p in decoded['parents']]
    cached = dict(decoded, parents=parents)
    environment = dict(score_source.__globals__)
    original_reference = environment['reference_segment']
    refs = {}
    def reference(child):
        key = id(child)
        if key not in refs:
            r = original_reference(child)
            refs[key] = dict(r, points=np.asarray(r['points'],np.float64))
        return refs[key]
    captured = {}
    def assignment(cost):
        captured['cost'] = np.array(cost,copy=True)
        return linear_sum_assignment(cost)
    environment['reference_segment'] = reference
    environment['linear_sum_assignment'] = assignment
    scorer = FunctionType(score_source.__code__, environment, score_source.__name__,
                          score_source.__defaults__, score_source.__closure__)
    score = scorer(cached, truth, direction_semantics)
    costs = captured.get('cost', np.full((len(parents),len(truth['parents'])),1e12))
    available = costs < 0
    tp = np.zeros(costs.shape,dtype=np.int64)
    tp[available] = np.rint(-costs[available]/1e6).astype(np.int64)
    if np.any(tp > counts[:,None]) or np.any(tp < 0):
        raise ValueError('D3 cost/count contract violated')
    error = costs[available] + tp[available]*1e6
    if np.any(error < -1e-7) or np.any(error > 9*tp[available]+1e-7):
        raise ValueError('D3 geometry error term outside proven count bound')
    invalid = sum(v['counted_false_children'] for v in decoded['invalid_predictions'])
    table = dict(costs=costs,tp=tp,child_counts=counts,invalid_children=np.asarray(invalid))
    check = evaluate_reference_subset(table,np.arange(len(parents)))
    expected = score['categories']['overall']
    if check['TP']!=expected['TP'] or check['FP']!=expected['FP']:
        raise AssertionError('Compiled utility differs from original full-pool D3')
    return table,score


def evaluate_reference_subset(table, selected):
    ids = np.asarray(selected,dtype=np.int64)
    if ids.ndim!=1 or len(set(ids.tolist()))!=len(ids) or np.any(ids<0) or np.any(ids>=len(table['child_counts'])):
        raise ValueError('Unique valid whole-parent row indices required')
    # Preserve caller output order because D3 can break assignment ties by row.
    costs = table['costs'][ids]
    matched = 0
    if costs.size:
        rows,columns = linear_sum_assignment(costs)
        matched = int(table['tp'][ids[rows],columns].sum())
    predicted = int(table['child_counts'][ids].sum()) + int(table['invalid_children'])
    return dict(TP=matched,FP=predicted-matched,predicted=predicted,
                reference_utility=2*matched-predicted)


def compile_compact_reference_table(parents, target, decode_subset, batch_size=1024):
    """Keep every original assignment cost while sampling bounded parent chunks.

    D3 costs are pairwise before its one global assignment. Original chunk
    scoring captures each complete cost row; no local chunk TP is summed into
    a global result. The full cost matrix and global assignment remain intact.
    This offline helper reports overall reference capacity only.
    """
    if batch_size < 1:
        raise ValueError('Positive exact reference compilation chunk size required')
    rows, columns = len(parents), len(target['parents'])
    costs = np.empty((rows, columns), np.float64)
    tp = np.empty((rows, columns), np.int64)
    counts = np.asarray([p['child_count'] for p in parents], np.int64)
    for start in range(0, rows, batch_size):
        stop = min(start+batch_size, rows)
        decoded = decode_subset(parents, np.arange(start, stop))
        table, _ = compile_reference_table(decoded, target)
        if int(table['invalid_children']) or not np.array_equal(table['child_counts'], counts[start:stop]):
            raise ValueError('Compact parent decoder changed complete prediction ownership')
        costs[start:stop], tp[start:stop] = table['costs'], table['tp']
    result = dict(costs=costs, tp=tp, child_counts=counts, invalid_children=np.asarray(0))
    score = evaluate_reference_subset(result, np.arange(rows))
    overall = dict(G=target['original_children'], **{k: score[k] for k in ('predicted', 'TP', 'FP')})
    return result, dict(categories=dict(overall=overall),
                       costs_from_original_D3=True, global_assignment_not_sum_of_chunk_TP=True)
