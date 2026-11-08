"""Independent replay of frozen exports, retaining every polyline corner.

This is a scoring-version correction, not a model improvement. It does not
claim welding-side, direction or E-field acceptance where those are unknown.
"""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse, json, math, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from tools import curveonly_01a1ac7c_experiment as e

VERSION = 'CURVEONLY_EXPORT_GEOMETRY_CORNER_ARC_V2'
ROOTREC = e.REC


def sampled(points, spacing=1.):
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError('Invalid finite 3D curve points')
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    take = np.r_[True, d > 1e-12]
    p = p[take]
    if len(p) < 2:
        raise ValueError('Zero-length published curve')
    count = np.maximum(1, np.ceil(np.linalg.norm(np.diff(p, axis=0), axis=1)/spacing).astype(int))
    out = [p[i] + np.arange(n)[:, None]/n*(p[i+1]-p[i]) for i, n in enumerate(count)]
    return np.vstack([*out, p[-1:]])


def distances(query, points):
    """Exact point-to-continuous-polyline distance using a safe KD search."""
    q = np.asarray(query, float); p = np.asarray(points, float)
    a = p[:-1]; v = np.diff(p, axis=0); den = (v*v).sum(1)
    good = den > 1e-24; a = a[good]; v = v[good]; den = den[good]
    if not len(a):
        raise ValueError('Empty polyline segments')
    centers = a + v/2; half = np.sqrt(den).max()/2
    tree = cKDTree(centers)
    result = []
    for start in range(0, len(q), 2048):
        x = q[start:start+2048]
        d, _ = tree.query(x, workers=1)
        # Closest center is a point of its segment, hence d is an upper bound.
        # Any closer segment must have center within d + max_half_length.
        neighbours = tree.query_ball_point(x, d+half+1e-9, workers=1)
        counts = np.fromiter(map(len, neighbours), int, count=len(x))
        indices = np.concatenate(neighbours).astype(int)
        qi = np.repeat(np.arange(len(x)), counts)
        delta = x[qi] - a[indices]
        t = np.clip((delta*v[indices]).sum(1)/den[indices], 0, 1)
        square = ((delta-t[:, None]*v[indices])**2).sum(1)
        result.extend(np.sqrt(np.minimum.reduceat(square, np.r_[0, np.cumsum(counts)[:-1]])))
    return np.asarray(result)


def arc_points(parameters):
    a = parameters
    c = np.array(a['center_mm'], float); u = np.array(a['start_direction'], float)
    n = np.array(a['plane_normal'], float); r = float(a['radius_mm']); sweep = float(a['sweep_rad'])
    if not np.isfinite(np.r_[c, u, n, r, sweep]).all() or r <= 0:
        raise ValueError('Invalid analytic circle')
    if max(abs(np.linalg.norm(u)-1), abs(np.linalg.norm(n)-1), abs(u@n)) > 1e-6:
        raise ValueError('Invalid directed circle basis')
    if abs(sweep) < 1e-10 or abs(sweep) > 2*math.pi+1e-10:
        raise ValueError('Zero turn or repeated circular traversal')
    grid = np.linspace(0, sweep, max(3, int(math.ceil(abs(sweep)*r)), int(math.ceil(abs(sweep)/.02)))+1)
    return c+r*(np.cos(grid)[:, None]*u + np.sin(grid)[:, None]*np.cross(n, u))


def exported(parent):
    result = dict(parent); parts = []; turns = []; problems = []
    try:
        for i, s in enumerate(parent['segments']):
            if s['segment_index'] != i:
                problems.append('SEGMENT_ORDER')
            p = sampled(s['points'])
            if 'arc_parameters' in s:
                actual = arc_points(s['arc_parameters'])
                if max(np.linalg.norm(p[0]-actual[0]), np.linalg.norm(p[-1]-actual[-1])) > 1e-6:
                    problems.append('ANALYTIC_ENDPOINT_DISAGREEMENT')
                if max(distances(p, actual).max(), distances(actual, p).max()) > .1:
                    problems.append('ANALYTIC_POINTS_DISAGREEMENT')
                length = abs(s['arc_parameters']['sweep_rad'])*s['arc_parameters']['radius_mm']
                turns.append(abs(s['arc_parameters']['sweep_rad']))
                p = actual
            else:
                length = e.core.arclength(np.asarray(s['points']))[-1]
                fit = e.core.circle_fit(np.asarray(s['points']))
                turns.append(abs(fit['sweep']) if fit else None)
            if abs(float(s['length_mm'])-length) > 1e-6*max(1, length):
                problems.append('LENGTH_DISAGREEMENT')
            if parts and np.linalg.norm(parts[-1][-1]-p[0]) > 1e-6:
                problems.append('SEGMENT_GAP_OR_ORDER')
            parts.append(p)
        if not parts:
            raise ValueError('No published segments')
        actual = np.vstack([parts[0], *[p[1:] for p in parts[1:]]])
        summary = sampled(parent['points'])
        if max(distances(actual, summary).max(), distances(summary, actual).max()) > .1:
            problems.append('PARENT_SUMMARY_DISAGREEMENT')
        result['points'] = actual.tolist()
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        problems.append('MALFORMED_EXPORT:'+str(exc))
    result['_valid_export'] = not problems
    result['_problems'] = problems
    result['_segment_turns'] = turns
    return result


def geometry_error(pred, truth):
    if not pred.get('_valid_export', True) or pred['weld_type'] != truth['weld_type']:
        return None
    a = np.asarray(pred['points'], float); b = np.asarray(truth['points'], float)
    pc = np.linalg.norm(a[0]-a[-1]) < .1; tc = np.linalg.norm(b[0]-b[-1]) < .1
    if pc != tc:
        return None
    end = 0. if pc else min(np.linalg.norm(a[[0, -1]]-b[[0, -1]], axis=1).max(),
                              np.linalg.norm(a[[0, -1]]-b[[-1, 0]], axis=1).max())
    if end > 3. or np.any(a.max(0) < b.min(0)-3) or np.any(b.max(0) < a.min(0)-3):
        return None
    error = float(distances(sampled(a), b).max())
    if error > 3.:
        return None
    error = max(error, float(distances(sampled(b), a).max()))
    if error > 3.:
        return None
    return dict(endpoint_mm=float(end), whole_curve_mm=error)


def process_error(pred, truth, all_truth):
    if not pred['_valid_export']:
        return 'INVALID_EXPORT'
    if truth.get('circle'):
        for turn in pred['_segment_turns']:
            if turn is None:
                return 'CIRCULAR_TURN_UNCERTIFIED'
            if turn > math.pi/6+1e-12:
                return 'ARC_TURN_ABOVE_30_DEGREES'
    else:
        expected = [np.asarray(all_truth['children'][i]['points'])[-1] for i in truth['children'][:-1]]
        actual = [np.asarray(s['points'])[-1] for s in pred['segments'][:-1]]
        if not all(any(np.linalg.norm(a-b) <= 3. for b in actual) for a in expected):
            return 'REQUIRED_GENERAL_CURVE_CUT_ABSENT'
    return None


def matching(pred, truth, process_truth=None):
    cost = np.full((len(pred), len(truth)), 1e6); edges = {}
    for i, p in enumerate(pred):
        for j, t in enumerate(truth):
            error = geometry_error(p, t)
            if error is not None and (process_truth is None or process_error(p, t, process_truth) is None):
                edges[i, j] = error; cost[i, j] = sum(error.values())
    found = []
    if cost.size:
        pi, ti = linear_sum_assignment(cost)
        found = [dict(prediction=int(i), truth=int(j), **edges[i, j]) for i, j in zip(pi, ti) if (i, j) in edges]
    tp = len(found)
    return dict(G=len(truth), P=len(pred), TP=tp, FP=len(pred)-tp, FN=len(truth)-tp, matches=found)


def score(round_name):
    start = time.monotonic(); src = ROOTREC/round_name
    if (src/'run/OUTPUTS_FROZEN.json').exists():
        src = src/'run'
    out = ROOTREC/'evaluation_v2'/round_name
    frozen = e.read(src/'OUTPUTS_FROZEN.json')
    index = {r['source_id']: r for r in e.read(ROOTREC/'PRIMARY129_TRUTH_INDEX.json')['cases']}
    cases = []
    for r in frozen['cases']:
        assert e.sha(r['prediction']['path']) == r['prediction']['sha256']
        value = e.read(r['prediction']['path']); t = e.read(index[r['source_id']]['truth']['path'])
        pred = [exported(p) for p in value['parents']]
        whole = matching(pred, t['runs']); process = matching(pred, t['runs'], t)
        row = dict(source_id=r['source_id'], split=r['split'], whole_curve=whole, process=process,
            export_problems=[dict(prediction=i, problems=p['_problems']) for i, p in enumerate(pred) if p['_problems']],
            process_failures=[dict(**m, reason=process_error(pred[m['prediction']], t['runs'][m['truth']], t)) for m in whole['matches'] if process_error(pred[m['prediction']], t['runs'][m['truth']], t)],
            input_failure=r.get('input_failure', False))
        cases.append(row); e.write(out/'cases'/r['source_id']/'SCORE.json', row)
        print('AUDIT', round_name, r['source_id'], whole['TP'], whole['G'], 'FP', whole['FP'], 'process', process['TP'], flush=True)
    totals = {}
    for split in ('TRAIN', 'VAL'):
        totals[split] = {}
        for metric in ('whole_curve', 'process'):
            v = {k: sum(r[metric][k] for r in cases if r['split'] == split) for k in ('G', 'P', 'TP', 'FP', 'FN')}
            v.update(extraction=v['TP']/v['G'] if v['G'] else None,
                     FP_per_G=v['FP']/v['G'] if v['G'] else None)
            totals[split][metric] = v
    result = dict(version=VERSION, source=e.bind(__file__), cases=cases, totals=totals,
        frozen_outputs=e.bind(src/'OUTPUTS_FROZEN.json'), primary_truth=e.bind(ROOTREC/'PRIMARY129_TRUTH_INDEX.json'),
        all_original_corners_preserved=True, continuous_polyline_distance=True, analytic_parameters_authoritative=True,
        unknown_circular_turn_is_failure=True, side_direction_and_E_attributes_accepted=False, full_contract_pass=False,
        no_model_or_export_change=True, elapsed_seconds=time.monotonic()-start)
    e.write(out/'RESULT.json', result)
    print('AUDIT_TOTALS', round_name, json.dumps(totals), flush=True)


def controls():
    line = np.array([[0., 0, 0], [10., 0, 0]])
    assert distances(np.array([[5., 2, 0], [12., 0, 0]]), line).tolist() == [2., 2.]
    rng = np.random.default_rng(20261106)
    p = rng.normal(size=(37, 3)); q = rng.normal(size=(53, 3))
    a = p[:-1]; v = np.diff(p, axis=0); d = q[:, None]-a
    t = np.clip((d*v).sum(2)/(v*v).sum(1), 0, 1)
    brute = np.linalg.norm(d-t[:, :, None]*v, axis=2).min(1)
    assert np.max(abs(distances(q, p)-brute)) < 1e-12
    zig = np.array([[0., 0, 0], [.01, 3.01, 0], [10., 0, 0]])
    assert any(np.array_equal(x, zig[1]) for x in sampled(zig))
    base = dict(weld_type='horizontal', points=line.tolist())
    truth = [base]
    assert matching([base, base], truth)['FP'] == 1
    assert matching([base], [base, base])['FN'] == 1
    shifted = dict(base, points=(line+[0, 3.01, 0]).tolist())
    assert matching([shifted], truth)['TP'] == 0
    shifted = dict(base, points=(line+[0, 2.99, 0]).tolist())
    assert matching([shifted], truth)['TP'] == 1
    target = dict(circle=True)
    for turn, expected in ((None, 'CIRCULAR_TURN_UNCERTIFIED'), (math.pi/6, None), (math.pi/6+.00001, 'ARC_TURN_ABOVE_30_DEGREES')):
        assert process_error(dict(_valid_export=True, _segment_turns=[turn]), target, {}) == expected
    e.write(ROOTREC/'evaluation_v2/CONTROLS.json', dict(passed=True, exact_distance_vs_bruteforce=True,
        corner_retention=True, one_to_one_counts=True, tolerance_3mm=True, unknown_turn_rejected=True,
        arc30_boundary=True, source=e.bind(__file__), version=VERSION))
    print('AUDIT_CONTROLS_PASSED', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['controls', 'score']); ap.add_argument('--round', default='round05')
    args = ap.parse_args()
    controls() if args.action == 'controls' else score(args.round)
