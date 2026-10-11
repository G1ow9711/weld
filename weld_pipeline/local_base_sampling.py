"""Estimate local 2D sampling gaps without using reference labels or CAD."""
import numpy as np
from scipy.spatial import cKDTree


def sampling_fill_radius(tree, point, inward, lateral):
    ids = tree.query_ball_point(np.asarray(point)[:2], 48.)
    if len(ids) < 6:
        return 6., {'state': 'INSUFFICIENT_LOCAL_SAMPLING_STATISTICS', 'radius_mm': 6.}
    points = tree.data[ids]
    centre = points.mean(0)
    covariance = (points - centre).T @ (points - centre) / len(points)
    if np.linalg.eigvalsh(covariance)[0] < 1.:
        return 6., {'state': 'LOCAL_POINTS_ONLY_SUPPORT_A_LINE', 'radius_mm': 6.}
    distances, neighbours = cKDTree(points).query(points, k=min(24, len(points)))
    displacement = points[neighbours] - points[:, None, :]
    spacings = []
    for axis in (inward, lateral):
        projected = np.abs(displacement @ np.asarray(axis))
        cone = (projected >= .75 * distances) & (distances > 1e-5)
        minimum = np.where(cone, distances, np.inf).min(1)
        finite = minimum[np.isfinite(minimum)]
        if len(finite) < 3:
            return 6., {'state': 'TWO_DIMENSIONAL_SAMPLING_GAPS_UNRESOLVED', 'radius_mm': 6.}
        spacings.append(float(np.median(finite)))
    radius = max(6., .5 * float(np.hypot(*spacings)))
    return radius, {'state': 'LOCAL_2D_SAMPLING_GAPS_MEASURED',
        'inward_spacing_mm': spacings[0], 'lateral_spacing_mm': spacings[1],
        'radius_mm': radius, 'points_used': len(points),
        'material_identity_or_hole_continuity_implied': False}
