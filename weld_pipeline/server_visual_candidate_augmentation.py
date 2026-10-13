"""Shared input-only visual arc augmentation for training and inference.

Preserves all original profiles and their IDs. New visual profiles have their
own raw/visual feature branch; no fabricated legacy feature rows are supplied.
Current observed arc recovery is XY parallel. It does not certify weld validity.
"""
import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits
from .sjg_curve_queries_907 import CurveCarrier
from .server_tiled_multiview_visual_features import visual_features, SCHEMA

def circle3(p):
    matrix = 2 * (p[1:] - p[0])
    if abs(np.linalg.det(matrix)) < 1.:
        return None
    center = np.linalg.solve(matrix, np.sum(p[1:] ** 2, axis=1) - np.sum(p[0] ** 2))
    return center, float(np.linalg.norm(p[0] - center))

def propose(xyz, scores, seed=1031):
    chosen = np.flatnonzero(scores >= .5)
    if len(chosen) < 100:
        return [], dict(selected_points=len(chosen), reason='Insufficient visual point support')
    p = xyz[chosen]; bins, counts = np.unique(np.round(p[:, 2] * 2), return_counts=True)
    levels = bins[np.argsort(-counts, kind='stable')[:3]] / 2
    rng = np.random.default_rng(seed); output = []
    max_radius = float(np.linalg.norm(np.ptp(xyz[:, :2], axis=0)) * 10)
    for level in levels:
        ids = chosen[np.abs(p[:, 2] - level) <= .25]
        z = float(np.median(xyz[ids, 2]))
        remaining = xyz[ids, :2].copy()
        for iteration in range(8):
            if len(remaining) < 100:
                break
            subset = remaining[rng.choice(len(remaining), min(6000, len(remaining)), replace=False)]
            best = None; best_count = 0
            for _ in range(384):
                candidate = circle3(subset[rng.choice(len(subset), 3, replace=False)])
                if candidate is None:
                    continue
                center, radius = candidate
                if not 20 <= radius <= max_radius:
                    continue
                distance = np.abs(np.linalg.norm(subset - center, axis=1) - radius)
                n = int((distance <= 1.).sum())
                if n > best_count:
                    best = (center, radius); best_count = n
            if best is None or best_count < 50:
                break
            center, radius = best
            for _ in range(3):
                mask = np.abs(np.linalg.norm(remaining - center, axis=1) - radius) <= 1.
                support = remaining[mask]
                if len(support) < 100:
                    break
                fit = least_squares(lambda v: np.linalg.norm(support - v[:2], axis=1) - v[2],
                                    np.r_[center, radius], max_nfev=30, loss='soft_l1', f_scale=.5)
                center, radius = fit.x[:2], float(fit.x[2])
            mask = np.abs(np.linalg.norm(remaining - center, axis=1) - radius) <= 1.
            support = remaining[mask]
            if len(support) < 100 or not 20 <= radius <= max_radius:
                break
            remaining = remaining[~mask]
            angles = np.sort(np.mod(np.arctan2(support[:, 1] - center[1], support[:, 0] - center[0]), 2 * np.pi))
            # Rotate at the largest observed gap and retain other finite gaps.
            wrap_gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
            cut = (int(np.argmax(wrap_gaps)) + 1) % len(angles)
            unwrapped = np.r_[angles[cut:], angles[:cut] + 2 * np.pi]
            groups = np.split(unwrapped, np.flatnonzero(np.diff(unwrapped) * radius > 10.) + 1)
            for group in groups:
                if len(group) < 30 or (group[-1] - group[0]) * radius < 30:
                    continue
                representation = dict(kind='arc', arc_parameters=dict(center_mm=[*center.tolist(), z],
                    radius_mm=radius, plane_normal=[0., 0., 1.],
                    start_direction=[float(np.cos(group[0])), float(np.sin(group[0])), 0.],
                    sweep_rad=float(group[-1] - group[0]), is_closed=False, units='mm', angle_units='rad'))
                CurveCarrier(representation)  # Reject malformed finite analytic geometry.
                output.append(dict(representation=representation, support_points=len(group),
                    radial_support_tolerance_mm=1., maximum_observed_support_gap_mm=10.,
                    observed_plane_z_mm=z, input_selection_depth_mm=float(level),
                    maximum_input_plane_distance_mm=float(np.abs(xyz[ids, 2] - z).max()),
                    plane_band_halfwidth_mm=.25, input_generated=True))
    return output, dict(selected_points=len(chosen), input_depth_levels_mm=levels.tolist(),
                        orientation_scope='XY parallel arcs only', raw_points=len(xyz))


def predict_point_scores(bundle, xyz):
    if bundle['schema'] != SCHEMA or bundle['raw_units'] != 'mm':
        raise ValueError('Visual model input contract changed')
    features, audit = visual_features(xyz)
    with threadpool_limits(limits=2):
        scores = np.concatenate([bundle['estimator'].predict_proba(features[b:b+65536])[:,1]
            for b in range(0,len(xyz),65536)]).astype(np.float32)
    if scores.shape != (len(xyz),) or not np.isfinite(scores).all():
        raise ValueError('Full raw point predictions required')
    return scores, audit


def augmented_profiles(original_profiles, xyz, scores):
    xyz = np.asarray(xyz, np.float64)
    scores = np.asarray(scores)
    if scores.shape != (len(xyz),) or not np.isfinite(scores).all():
        raise ValueError('Scores must bind every original input point')
    proposals, audit = propose(xyz, scores)
    selected = np.flatnonzero(scores >= .5)
    extra = []
    for proposal in proposals:
        curve = CurveCarrier(proposal['representation'])
        ids = selected[np.abs(xyz[selected,2]-proposal['input_selection_depth_mm']) <= .25]
        _, distance = curve.project(xyz[ids])
        ids = ids[distance <= 1.25]
        extra.append(dict(representation=proposal['representation'], length_mm=curve.length,
            support_normal=curve.arc.plane_normal.tolist(), support_intervals_mm=[],
            raw_support_point_ids=ids.tolist(), source='VISUAL_OBSERVED_FINITE_ARC_V1',
            legacy_feature_row_available=False, feature_route='RAW_XYZ_VISUAL_CONTEXT',
            support_is_certified_weld_contact=False, visual_support_audit=proposal))
    combined = list(original_profiles) + extra
    if any(a is not b for a,b in zip(combined,original_profiles)):
        raise ValueError('Original profile identity changed')
    return combined, dict(audit, original_profiles=len(original_profiles), additional_profiles=len(extra),
        original_profile_ids_preserved=True, fabricated_legacy_features=False,
        GT_geometry_reads=0, full_weld_extraction=False)
