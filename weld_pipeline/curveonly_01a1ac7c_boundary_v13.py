"""Compact all-population boundary head on the unchanged34 curve features."""
import math
import numpy as np
from scipy.spatial import cKDTree
from . import curveonly_01a1ac7c_runtime_v12 as previous
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_arc_publication_v6 as publication


class CurveHeads:
    def __init__(self, occupancy, boundary):
        self.occupancy = occupancy; self.boundary = boundary
        self.classes_ = occupancy.classes_; self.n_features_in_ = occupancy.n_features_in_

    def predict_proba(self, x):
        return self.occupancy.predict_proba(x)


def coordinate(point, curve):
    p = np.asarray(curve, float); s = cad.arclength(p)
    _, nearest = cKDTree(p).query(np.asarray(point), workers=1)
    indices = np.unique(np.clip([nearest-1, nearest], 0, len(p)-2))
    a = p[indices]; v = p[indices+1]-a
    t = np.clip(((np.asarray(point)-a)*v).sum(1)/np.maximum((v*v).sum(1), 1e-24), 0, 1)
    distance = np.linalg.norm(np.asarray(point)-a-t[:, None]*v, axis=1)
    j = int(np.argmin(distance))
    return float(s[indices[j]]+t[j]*np.linalg.norm(v[j]))


def interval(parent, candidate):
    q = np.asarray(parent['points'], float); p = np.asarray(candidate['points'], float)
    length = float(cad.arclength(q)[-1]); full = float(cad.arclength(p)[-1])
    # A complete closed carrier has the same spatial endpoint at s=0 and s=L.
    # Nearest station lookup alone incorrectly maps both endpoints to s=0.
    if parent['is_closed'] and abs(length-full) <= .001*max(full, 1.):
        return 0., full
    return sorted([coordinate(q[0], p), coordinate(q[-1], p)])


def piece(parent, points, index):
    length = float(cad.arclength(points)[-1]); circle = cad.circle_fit(points)
    count = 1 if not circle else max(1, int(math.ceil(abs(circle['sweep'])/(math.pi/6))))
    segments = []
    for i in range(count):
        p = cad.slice_curve(points, length*i/count, length*(i+1)/count)
        segments.append(dict(segment_index=i, points=p.tolist(), length_mm=float(cad.arclength(p)[-1]),
            geometry_type='arc' if circle else 'bspline_polyline', reason='LEARNED_TASK_INTERVAL',
            status='NOMINAL_GEOMETRY_OBSERVATION_RECORDED'))
    return dict(parent, parent_id=f'curve_{index:05d}', points=points.tolist(), circle=circle,
        segments=segments, is_closed=bool(np.linalg.norm(points[0]-points[-1]) < .1),
        boundary_source='ALL_CANDIDATE_BACKGROUND_TRAIN_HEAD')


def decode(features, packet, model, threshold=.5):
    decoded = previous.decode(features, packet, model.occupancy, threshold)
    events = model.boundary.predict_proba(features)[:, 1] if len(features) else np.empty(0)
    parents = []
    for parent in decoded['parents']:
        ci = parent['source_candidate']; c = packet['candidates'][ci]
        a, b = packet['offsets'][ci:ci+2]; s = np.asarray(c['station_s_mm']); prob = events[a:b]
        lo, hi = interval(parent, c); q = np.asarray(parent['points']); length = float(cad.arclength(q)[-1])
        cuts = []; start = 0
        while start < len(prob):
            if prob[start] < threshold:
                start += 1; continue
            end = start+1
            while end < len(prob) and prob[end] >= threshold:
                end += 1
            if s[start] > lo+1. and s[end-1] < hi-1.:
                peak = np.flatnonzero(prob[start:end] >= prob[start:end].max()-1e-10)+start
                where = (float((s[peak[0]]+s[peak[-1]])/2)-lo)*length/max(hi-lo, 1e-12)
                if 1e-6 < where < length-1e-6:
                    cuts.append(where)
            start = end
        bounds = [0., *sorted(set(cuts)), length]
        for begin, end in zip(bounds[:-1], bounds[1:]):
            parents.append(piece(parent, cad.slice_curve(q, begin, end), len(parents)))
    result = dict(decoded, parents=parents, geometry_units=len(parents),
        published_segments=sum(len(p['segments']) for p in parents), boundary_model=True)
    return publication.publish(result)
