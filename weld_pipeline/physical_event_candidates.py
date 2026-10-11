"""Unclassified boundary locations from point-derived finite planes and line extents.

This module accepts geometry only. Proposals are not welds or measured attributes.
"""
import numpy as np

VERSION = 'pcd_physical_event_candidates_v1'
PADDING_MM = 20.
PLANE_RANGE_TOLERANCE_MM = 2.
PLANE_Z_TOLERANCE_MM = .05


def candidate_positions(sequence, planes, finite=True):
    if sequence['type'] != 'horizontal' or sequence.get('curve_kind') == 'arc':
        raise ValueError('This proposal audit covers horizontal lines only')
    q = np.asarray(sequence['xyz'], float)
    u = np.asarray(sequence['direction'], float)
    if q.ndim != 2 or q.shape[1] != 3 or len(q) < 2 or not np.isfinite(q[[0, -1]]).all():
        raise ValueError('Finite line endpoints required')
    if u.shape != (3,) or not np.isfinite(u).all() or abs(np.linalg.norm(u)-1.) > 1e-6:
        raise ValueError('Unit tangent required')
    origin = q[0]; span = float((q[-1]-origin)@u)
    if span <= 2*PADDING_MM or abs(u[2]) > 1e-6:
        raise ValueError('Expected padded horizontal line from the fixed generator')
    positions = [PADDING_MM, span-PADDING_MM]
    counts = dict(inner_endpoints=2, crossing_planes=0, parallel_planes=0,
                  outside_sequence=0, outside_finite_plane=0)
    for plane in planes:
        normal = np.asarray(plane['normal_xy'], float)
        along = np.asarray(plane['direction_xy'], float)
        offset = float(plane['offset_mm'])
        if (normal.shape != (2,) or along.shape != (2,)
                or not np.isfinite(np.r_[normal, along, offset]).all()
                or abs(np.linalg.norm(normal)-1.) > 1e-6
                or abs(np.linalg.norm(along)-1.) > 1e-6):
            raise ValueError('Finite unit plane basis required')
        denominator = float(normal@u[:2])
        if abs(denominator) < 1e-8:
            counts['parallel_planes'] += 1
            continue
        t = float((offset-normal@origin[:2])/denominator)
        if t < 0. or t > span:
            counts['outside_sequence'] += 1
            continue
        point = origin+t*u
        if finite:
            lo, hi = map(float, plane['range_mm'])
            zl, zh = map(float, plane['z_range_mm'])
            along_position = float(point[:2]@along)
            if (not lo-PLANE_RANGE_TOLERANCE_MM <= along_position <= hi+PLANE_RANGE_TOLERANCE_MM
                    or not zl-PLANE_Z_TOLERANCE_MM <= point[2] <= zh+PLANE_Z_TOLERANCE_MM):
                counts['outside_finite_plane'] += 1
                continue
        positions.append(t)
        counts['crossing_planes'] += 1
    # Equality alone removes duplicates; no separation or minimum-length rule.
    result = np.unique(np.asarray(positions, float))
    counts['unique_positions'] = len(result)
    return result, counts


def closest_event(sequence, positions, point):
    origin = np.asarray(sequence['xyz'][0], float)
    u = np.asarray(sequence['direction'], float)
    positions = np.asarray(positions, float)
    target = float((np.asarray(point, float)-origin)@u)
    insertion = int(np.searchsorted(positions, target))
    indices = {max(0, min(len(positions)-1, insertion)),
               max(0, min(len(positions)-1, insertion-1))}
    choices = [(float(np.linalg.norm(origin+positions[i]*u-point)), i) for i in indices]
    error, index = min(choices)
    return dict(error_mm=error, covered_3mm=error <= 3.,
                candidate_position_mm=float(positions[index]),
                point_mm=(origin+positions[index]*u).tolist())


def physical_checks():
    seq = dict(xyz=np.array([[-20., 0., 0.], [120., 0., 0.]]),
               direction=np.array([1., 0., 0.]), type='horizontal')
    def plane(offset, yrange=(-10., 10.), zrange=(-1., 50.)):
        return dict(normal_xy=[1., 0.], direction_xy=[0., 1.], offset_mm=offset,
                    range_mm=list(yrange), z_range_mm=list(zrange))
    planes = [plane(13.25), plane(6.2680269), plane(6.274),
              plane(30., (10., 20.)), plane(40., zrange=(10., 20.)), plane(200.),
              dict(normal_xy=[0., 1.], direction_xy=[-1., 0.], offset_mm=1.,
                   range_mm=[-100., 100.], z_range_mm=[-1., 50.])]
    finite, _ = candidate_positions(seq, planes)
    unlimited, _ = candidate_positions(seq, planes, finite=False)
    expected = np.array([20., 26.2680269, 26.274, 33.25, 120.])
    if not np.allclose(finite, expected, atol=1e-12, rtol=0):
        raise ValueError('Finite/parallel/extent/close-boundary physical check failed')
    if not np.allclose(unlimited, np.r_[expected[:-1], 50., 60., expected[-1]], atol=1e-12, rtol=0):
        raise ValueError('Unlimited control changed the finite-plane diagnosis')
    reverse = dict(seq, xyz=seq['xyz'][::-1], direction=-seq['direction'])
    reversed_positions, _ = candidate_positions(reverse, planes)
    if not np.allclose(reversed_positions, np.sort(140.-finite), atol=1e-12, rtol=0):
        raise ValueError('Reversing traversal changed physical event points')
    flips = [dict(p, normal_xy=(-np.asarray(p['normal_xy'])).tolist(), offset_mm=-p['offset_mm'],
                  direction_xy=(-np.asarray(p['direction_xy'])).tolist(),
                  range_mm=[-p['range_mm'][1], -p['range_mm'][0]]) for p in planes]
    flipped, _ = candidate_positions(seq, flips)
    if not np.array_equal(finite, flipped):
        raise ValueError('Plane normal/direction signs changed proposals')
    angle = .731; c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    translation = np.array([1034.5, -842.25, 87.])
    transformed = dict(seq, xyz=seq['xyz']@rotation.T+translation,
                       direction=rotation@seq['direction'])
    moved = []
    for p in planes:
        n = rotation[:2, :2]@p['normal_xy']; d = rotation[:2, :2]@p['direction_xy']
        moved.append(dict(p, normal_xy=n.tolist(), direction_xy=d.tolist(),
            offset_mm=p['offset_mm']+float(n@translation[:2]),
            range_mm=(np.asarray(p['range_mm'])+d@translation[:2]).tolist(),
            z_range_mm=(np.asarray(p['z_range_mm'])+translation[2]).tolist()))
    moved_positions, _ = candidate_positions(transformed, moved)
    error = float(np.abs(moved_positions-finite).max())
    if error > 1e-9:
        raise ValueError('Translation/yaw changed event locations')
    return dict(finite_extent_and_parallel_and_subsample_pass=True,
                close_distinct_boundaries_preserved=True, short_6_268mm_preserved=True,
                traversal_reverse_pass=True, plane_sign_pass=True,
                translation_yaw_max_position_error_mm=error,
                proposals_are_not_welds_or_acceptance=True)
