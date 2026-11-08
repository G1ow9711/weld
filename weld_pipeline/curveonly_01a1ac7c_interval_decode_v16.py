"""Pure NumPy interval decoder, numerically copied from the R9 control.

No Torch dependency or fitted model is needed for curve publication.
"""
import time,math
import numpy as np
from scipy.spatial import cKDTree
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_arc_publication_v6 as publication

def native_slice(points, start, end):
    p = np.asarray(points, float); length = float(cad.arclength(p)[-1])
    if end <= length+1e-8:
        return cad.slice_curve(p, start, min(end, length))
    if np.linalg.norm(p[0]-p[-1]) > 1e-6 or end-start > length+1e-6:
        raise ValueError('Unproved or repeated native wrap')
    a = cad.slice_curve(p, start, length); b = cad.slice_curve(p, 0., end-length)
    return np.vstack([a, b[1:]])

def interval_votes(c, predictions, threshold):
    p = np.asarray(c['points']); length = float(cad.arclength(p)[-1])
    s = np.linspace(0., length, len(predictions)); prob = predictions[:, :3]
    cls = np.argmax(prob, axis=1); selected = (cls>0) & (np.max(prob, axis=1)>=threshold)
    intervals = []; closed_native = np.linalg.norm(p[0]-p[-1]) <= 1e-6
    for kind in (1, 2):
        rows = np.flatnonzero(selected & (cls==kind))
        if not len(rows):
            continue
        a = s[rows]-predictions[rows, 3]; b = s[rows]+predictions[rows, 4]
        span = b-a; closed_vote = closed_native & (predictions[rows, 5]>=.5)
        if closed_native:
            a = np.mod(a, length); b = a+span
            a[closed_vote] = 0.; b[closed_vote] = length
            keep = (span > 1e-6) & ((span <= length+1e-7) | closed_vote)
        else:
            keep = (a>=-3.) & (b<=length+3.) & (b>a+1e-6)
            a = np.maximum(a, 0.); b = np.minimum(b, length)
            keep &= b > a+1e-6
        rows = rows[keep]; a = a[keep]; b = b[keep]
        if not len(rows):
            continue
        votes = np.column_stack([a,b]); w = prob[rows, kind]
        keys = np.round(votes/4.).astype(np.int64); unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        mass = np.bincount(inverse, weights=w); order = np.argsort(-mass, kind='stable')
        used = np.zeros(len(votes), bool); accepted = []
        for bin_index in order:
            center = unique[bin_index]*4.
            take = (~used) & (abs(votes-center).max(1)<=4.)
            if not take.any():
                continue
            aa,bb = np.average(votes[take], axis=0, weights=w[take]); nn = int(take.sum())
            support = float(w[take].sum()); expected = max(1., (bb-aa)/2.)
            if nn < min(2, len(rows)) or support < .2*expected:
                continue
            used[take] = True
            # One consensus may straddle adjacent4mm bins; retain one proposal.
            if any(max(abs(aa-x['start']), abs(bb-x['end']))<=4. for x in accepted):
                continue
            accepted.append(dict(start=float(aa), end=float(bb), cls=kind,
                confidence=float(w[take].mean()), support=nn, expected_stations=expected))
        intervals.extend(accepted)
    return intervals

def decode(features, packet, model, threshold=.5):
    started = time.perf_counter(); predictions = model.predict(features, packet); proposals = []
    for ci, c in enumerate(packet['candidates']):
        a,b = packet['offsets'][ci:ci+2]
        for vote in interval_votes(c, predictions[a:b], threshold):
            p = native_slice(c['points'], vote['start'], vote['end'])
            proposals.append(dict(points=p.tolist(), source_candidate=ci, vote=vote,
                weld_type='horizontal' if vote['cls']==1 else 'vertical', confidence=vote['confidence']))
    kept = []
    for proposal in sorted(proposals, key=lambda r:-r['confidence']):
        p = cad.resample(proposal['points'],1.); duplicate = False
        for old in kept:
            q = np.asarray(old['points'])
            if proposal['weld_type'] != old['weld_type'] or np.linalg.norm(p.mean(0)-q.mean(0))>5.:
                continue
            if max(cKDTree(p).query(q,workers=1)[0].max(), cKDTree(q).query(p,workers=1)[0].max())<.5:
                duplicate=True; break
        if not duplicate:
            kept.append(proposal)
    parents = []
    for i, r in enumerate(kept):
        p = np.asarray(r['points']); fit = cad.circle_fit(p)
        segment = dict(segment_index=0, points=p.tolist(), length_mm=float(cad.arclength(p)[-1]),
            geometry_type='arc' if fit else 'bspline_polyline', reason='COMPLETE_INTERVAL_ENDPOINT_VOTES',
            status='NOMINAL_GEOMETRY_OBSERVATION_RECORDED')
        parents.append(dict(parent_id=f'curve_{i:05d}', weld_type=r['weld_type'], weld_side=None,
            is_closed=bool(np.linalg.norm(p[0]-p[-1])<.1), segments=[segment], points=p.tolist(),
            confidence=r['confidence'], source_candidate=r['source_candidate'], circle=fit,
            interval_vote=r['vote'], actual_geometry_status='NOT_INDEPENDENTLY_ACCEPTED'))
    decoded = dict(parents=parents, geometry_units=len(parents), published_segments=len(parents), units='mm',
        feature_columns=34, same_model_and_configuration=True, interval_model=model.last_timing,
        model_stage_seconds=time.perf_counter()-started)
    return publication.publish(decoded)
