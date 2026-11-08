"""Bounded XYZ-only cylindrical-wall trace proposals in the workpiece frame.

Unlike the earlier global triples on a dense planar layer, local circles are
estimated on projected nonplanar observations. Every extent is observation
bounded. These are material candidates, never an automatic should-weld rule.
"""
from __future__ import annotations
import time, math
import numpy as np
from scipy.spatial import cKDTree


def fit2(q):
    origin = q.mean(0); x = q-origin
    cov = x.T@x/len(x)
    eigen = np.linalg.eigvalsh(cov)
    if eigen[0] < 1e-8 or eigen[0]/max(eigen[-1], 1e-12) > .12:
        return None
    try:
        c = np.linalg.solve(2*cov, (x*(x*x).sum(1)[:, None]).mean(0))
    except np.linalg.LinAlgError:
        return None
    radius = float(np.sqrt(np.mean(((x-c)**2).sum(1))))
    residual = abs(np.linalg.norm(x-c, axis=1)-radius)
    return origin+c, radius, residual


def proposals(xyz):
    started = time.monotonic(); xyz = np.asarray(xyz, float)
    # Z is the agreed workpiece base normal, not a weld-derived transform.
    levels, counts = np.unique(np.round(xyz[:, 2]*4).astype(np.int64), return_counts=True)
    selected = np.argsort(counts)[::-1]
    selected = [j for j in selected if counts[j] >= max(200, .005*len(xyz))][:8]
    planes = np.array([np.median(xyz[np.round(xyz[:, 2]*4).astype(np.int64) == levels[j], 2]) for j in selected])
    if not len(planes):
        return [], dict(reason='NO_DOMINANT_LEVELS', seconds=time.monotonic()-started)
    wall = xyz[np.min(abs(xyz[:, 2, None]-planes[None]), axis=1) > .3]
    if len(wall) < 24:
        return [], dict(reason='NO_NONPLANAR_OBSERVATIONS', seconds=time.monotonic()-started)
    keys = np.round(wall[:, :2]*2).astype(np.int64)
    _, chosen, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    points = wall[chosen, :2]
    zmin = np.full(len(chosen), np.inf); zmax = np.full(len(chosen), -np.inf)
    np.minimum.at(zmin, inverse, wall[:, 2]); np.maximum.at(zmax, inverse, wall[:, 2])
    tree = cKDTree(points)
    _, seeds = np.unique(np.floor(points/6).astype(np.int64), axis=0, return_index=True)
    if len(seeds) > 12000:
        rng = np.random.default_rng(20261106)
        seeds = np.sort(rng.choice(seeds, 12000, replace=False))
    dist, idx = tree.query(points[seeds], k=min(256, len(points)), distance_upper_bound=64., workers=1)
    raw = []; local_fits = 0
    for i in range(len(seeds)):
        for window in (8., 16., 32., 64.):
            ids = idx[i, dist[i] <= window]
            if len(ids) < 12:
                continue
            q = points[ids]; result = fit2(q); local_fits += 1
            if result is None:
                continue
            c, radius, residual = result
            if not 3. <= radius <= 20*window:
                continue
            if np.quantile(residual, .9) > .15 or np.median(residual) > .03:
                continue
            # A material straight trace has no measurable sag and is excluded.
            ang = np.unwrap(np.arctan2(q[:, 1]-c[1], q[:, 0]-c[0]))
            span = np.ptp(ang)
            if span > math.pi:
                ang = np.angle(np.exp(1j*(ang-ang[0]))); span = np.ptp(ang)
            if radius*(1-math.cos(min(span, math.pi)/2)) < .15:
                continue
            raw.append((float(np.quantile(residual, .9)), -len(q), c, radius))
    circles = []
    for quality, negsupport, center, radius in sorted(raw, key=lambda x:(x[0], x[1])):
        if any(np.linalg.norm(center-c) < 1. and abs(radius-r) < 1. for c, r in circles):
            continue
        circles.append((center, radius))
        if len(circles) >= 256:
            break
    out = []; accepted_circles = []; projected_queries = 0
    for center, radius in circles:
        # The nearest observations to a sampled circle provide bounded support;
        # a full-area/global point-circle RANSAC is never run.
        for iteration in range(2):
            n = min(16000, max(24, int(math.ceil(2*math.pi*radius/.5))))
            theta = np.arange(n)*2*math.pi/n
            ring = center+radius*np.column_stack([np.cos(theta), np.sin(theta)])
            d, ii = tree.query(ring, k=min(4, len(points)), distance_upper_bound=.8, workers=1)
            projected_queries += len(ring)
            ids = np.unique(ii[np.isfinite(d)])
            ids = ids[abs(np.linalg.norm(points[ids]-center, axis=1)-radius) <= .08]
            if len(ids) < 16:
                break
            fit = fit2(points[ids])
            if fit is not None and np.linalg.norm(fit[0]-center) < .5 and abs(fit[1]-radius) < .5:
                center, radius = fit[:2]
        else:
            angle = np.mod(np.arctan2(points[ids, 1]-center[1], points[ids, 0]-center[0]), 2*math.pi)
            order = np.argsort(angle); ids = ids[order]; angle = angle[order]
            delta = np.diff(np.r_[angle, angle[0]+2*math.pi])*radius
            gaps = np.flatnonzero(delta > 4.)
            if not len(gaps):
                groups = [np.r_[np.arange(len(ids)), 0]]
            else:
                pivot = (gaps[np.argmax(delta[gaps])]+1) % len(ids)
                ids = np.roll(ids, -pivot); angle = np.unwrap(np.roll(angle, -pivot))
                gaps = np.flatnonzero(np.diff(angle)*radius > 4.)+1
                groups = np.split(np.arange(len(ids)), gaps)
            count_before = len(out)
            for group in groups:
                if len(group) < 12:
                    continue
                aa = np.unwrap(angle[group]); arc = (aa[-1]-aa[0])*radius
                if arc < 5.:
                    continue
                support = ids[group]
                for level in planes:
                    near = (zmin[support] <= level+.8) & (zmax[support] >= level-.8)
                    if near.mean() < .7:
                        continue
                    lo, hi = aa[np.flatnonzero(near)[[0, -1]]]
                    theta = np.linspace(lo, hi, max(3, int(math.ceil((hi-lo)*radius))+1, int(math.ceil((hi-lo)/.02))+1))
                    q = np.column_stack([center[0]+radius*np.cos(theta), center[1]+radius*np.sin(theta), np.full(len(theta), level)])
                    out.append(dict(points=q.tolist(), material=-1, other_distance=[1e4]*3,
                        STEP=False, pose=False, source='XYZ_PROJECTED_NONPLANAR_TRACE',
                        observation=dict(projected_residual_mm=.08, height_bounded=True, support_points=len(support),
                            circle_center_xy=center.tolist(), radius_mm=float(radius))))
            if len(out) > count_before:
                accepted_circles.append(dict(center=center.tolist(), radius=float(radius), published=len(out)-count_before))
    return out, dict(seconds=time.monotonic()-started, raw_points=len(xyz), nonplanar_points=len(wall),
        projected_points=len(points), seeds=len(seeds), local_fits=local_fits, hypotheses=len(raw),
        unique_hypotheses=len(circles), projected_queries=projected_queries, candidates=len(out),
        levels=planes.tolist(), circles=accepted_circles, labels_or_reference_curves_read=False,
        all_values_global_not_case_tuned=True, should_weld_judgment=False)


def point_curves(xyz):
    curves, _ = proposals(xyz)
    return curves
