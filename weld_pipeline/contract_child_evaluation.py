"""New-contract child geometry diagnostics; never imported by inference.

Missing side/attribute truth blocks full acceptance. Geometry scores remain
available for development without changing the frozen parent scorer.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from weld_pipeline.evaluation import welding_type
from weld_pipeline.curve_geometry_metrics import polyline_hausdorff_bounds

VERSION = 'contract_child_geometry_v1'


def child_units(welds):
    out = []
    for parent_index, weld in enumerate(welds):
        for index, segment in enumerate(weld['segments']):
            points = np.asarray(segment['points'], dtype=float)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
                raise ValueError('Invalid child geometry')
            out.append({'parent_index': parent_index, 'parent_id': weld.get('id'),
                'segment_index': index, 'segment_count': len(weld['segments']),
                'type': welding_type(weld['type']), 'kind': segment['kind'],
                'points': points.tolist(), 'weld_side': weld.get('weld_side'),
                'is_closed': bool(np.linalg.norm(points[0] - points[-1]) < 1e-6),
                'id': str(weld.get('id', parent_index)) + ':' + str(index)})
    return out


def _error(pred, truth, limit=3.):
    a, b = np.asarray(pred['points']), np.asarray(truth['points'])
    direct = np.linalg.norm(a[[0, -1]] - b[[0, -1]], axis=1)
    reverse_error = np.linalg.norm(a[[-1, 0]] - b[[0, -1]], axis=1)
    reverse = (reverse_error.max(), reverse_error.sum()) < (direct.max(), direct.sum())
    error = reverse_error if reverse else direct
    # Side agreement cannot be inferred from an absent field.
    side_known = pred['weld_side'] is not None and truth['weld_side'] is not None
    side_ok = not side_known or pred['weld_side'] == truth['weld_side']
    basic = pred['type'] == truth['type'] and pred['kind'] == truth['kind']
    basic = basic and pred['is_closed'] == truth['is_closed'] and side_ok and error.max() <= limit
    # A repeated traverse can have zero Hausdorff error yet violates D4.
    direction = a[-1]-a[0]
    if pred['kind']=='line' and np.linalg.norm(direction)>1e-9:
        projection = a @ (direction/np.linalg.norm(direction))
        basic = basic and bool(np.all(np.diff(projection)>=-1e-6))
    vertices = np.round(a[:-1] if pred['is_closed'] else a,6)
    basic = basic and len(np.unique(vertices,axis=0))==len(vertices)
    metric = None
    if basic:
        metric = polyline_hausdorff_bounds(a, b, step_mm=1.)
        # D4 defines sampled maximum with sample step <= 1 mm. Report the
        # continuous upper bound too, but do not add a stricter contract gate.
        if metric['lower_mm'] <= limit < metric['upper_mm']:
            metric = polyline_hausdorff_bounds(a, b, step_mm=.05)
        basic = metric['lower_mm'] <= limit
    return {'accepted': bool(basic), 'start_mm': float(error[0]), 'end_mm': float(error[1]),
        'reversed': bool(reverse), 'whole_line': metric, 'side_truth_and_prediction_present': side_known}


def score_children(predictions, truths):
    pred, truth = child_units(predictions), child_units(truths)
    n, m = len(pred), len(truth)
    valid = np.zeros((n, m), dtype=bool)
    cost = np.full((n, m), 1e9)
    errors = {}
    for i, p in enumerate(pred):
        for j, t in enumerate(truth):
            if p['type'] != t['type'] or p['kind'] != t['kind']:
                continue
            # Cheap endpoint gate precedes full polyline sampling.
            a, b = np.asarray(p['points'])[[0, -1]], np.asarray(t['points'])[[0, -1]]
            if min(np.linalg.norm(a-b, axis=1).max(), np.linalg.norm(a[::-1]-b, axis=1).max()) > 3.:
                continue
            e = _error(p, t)
            errors[i, j] = e
            if e['accepted']:
                valid[i, j] = True
                cost[i, j] = e['start_mm'] + e['end_mm'] + e['whole_line']['lower_mm']
    matches = []
    if n and m:
        ii, jj = linear_sum_assignment(cost)
        matches = [(int(i), int(j)) for i, j in zip(ii, jj) if valid[i, j]]
    # Ignore IDs for pairing. Enforce the actual stored sequence order in any
    # multi-child parent pair; full reversal is allowed for undirected geometry.
    groups = {}
    for i, j in matches:
        groups.setdefault((pred[i]['parent_index'], truth[j]['parent_index']), []).append((i, j))
    invalid_order = set()
    for pairs in groups.values():
        ordered = sorted(pairs, key=lambda ij: pred[ij[0]]['segment_index'])
        indices = np.array([truth[j]['segment_index'] for i, j in ordered])
        if len(indices) > 2 and not (np.all(np.diff(indices) > 0) or np.all(np.diff(indices) < 0)):
            invalid_order.update(ordered)
    matches = [ij for ij in matches if ij not in invalid_order]
    used_pred, used_truth = {i for i,j in matches}, {j for i,j in matches}
    rows = [{'prediction_index': i, 'truth_index': j, 'prediction_id': pred[i]['id'],
        'truth_id': truth[j]['id'], **errors[i,j]} for i,j in matches]
    categories = {}
    for name, key, value in [('horizontal','type','horizontal'), ('vertical','type','vertical'),
            ('line','kind','line'), ('curve','kind','arc')]:
        ts = {j for j,t in enumerate(truth) if t[key] == value}
        ps = {i for i,p in enumerate(pred) if p[key] == value}
        tp = sum(j in ts for i,j in matches)
        fp = len(ps - used_pred)
        categories[name] = {'truth': len(ts), 'predictions': len(ps), 'valid': tp, 'false': fp,
            'rate': tp/len(ts) if ts else None,
            'status': 'PASS' if ts and tp/len(ts) > .95 and fp/len(ts) <= .05 else ('FAIL' if ts else 'NOT_RUN_NO_TRUTH')}
    tp = len(matches)
    core_pass = (tp/m > .95 and (n-tp)/m <= .05 and
        all(c['status'] == 'PASS' for c in categories.values() if c['truth'])) if m else n == 0
    side_present = all(t['weld_side'] is not None for t in truth) and all(p['weld_side'] is not None for p in pred)
    return {'evaluation_version': VERSION, 'truth_count': m, 'prediction_count': n,
        'valid_children': tp, 'false_predictions': n-tp, 'missed_children': m-tp,
        'valid_extraction_rate': tp/m if m else None, 'false_rate_per_truth': (n-tp)/m if m else None,
        'geometry_core_gates_pass': core_pass, 'categories': categories,
        'side_gate': 'AVAILABLE' if side_present else 'NOT_RUN_MISSING_SIDE_SCHEMA',
        'full_contract_pass': False, 'full_contract_status': 'NOT_ACCEPTED_ATTRIBUTES_AND_INDEPENDENT_TEST_REQUIRED',
        'matches': rows, 'unmatched_prediction_indices': sorted(set(range(n))-used_pred),
        'unmatched_truth_indices': sorted(set(range(m))-used_truth),
        'order_rejected_pairs': [list(ij) for ij in sorted(invalid_order)]}


def aggregate(scores):
    keys = ['truth_count','prediction_count','valid_children','false_predictions','missed_children']
    totals = {key: sum(s[key] for s in scores.values()) for key in keys}
    g, p, tp = totals['truth_count'], totals['prediction_count'], totals['valid_children']
    totals.update(valid_extraction_rate=tp/g if g else None, false_rate_per_truth=(p-tp)/g if g else None)
    totals['categories'] = {}
    for name in ['horizontal','vertical','line','curve']:
        c = {k: sum(s['categories'][name][k] for s in scores.values()) for k in ['truth','predictions','valid','false']}
        c.update(rate=c['valid']/c['truth'] if c['truth'] else None,
            status='PASS' if c['truth'] and c['valid']/c['truth'] > .95 and c['false']/c['truth'] <= .05 else ('FAIL' if c['truth'] else 'NOT_RUN_NO_TRUTH'))
        totals['categories'][name] = c
    totals['geometry_core_gates_pass'] = bool(g and tp/g > .95 and (p-tp)/g <= .05 and
        all(c['status']=='PASS' for c in totals['categories'].values() if c['truth']))
    totals['full_contract_pass'] = False
    return totals
