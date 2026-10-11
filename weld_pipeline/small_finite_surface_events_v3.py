"""Small finite vertical-face evidence from raw XYZ, independent of labels.

These are additional event hypotheses. They neither declare welds nor constrain
the existing continuous decoder. Nominal gravity and point-derived directions
are the same as the straight-candidate generator.
"""
import numpy as np
from scipy.spatial import cKDTree

from weld_pipeline.contact_component_event_candidates import component_positions
from weld_pipeline.physical_event_candidates import candidate_positions

VERSION = 'raw_multiscale_small_finite_faces_v3'
NEIGHBORS = (8, 16, 32)
PLANE_TOLERANCE_MM = .12
OFFSET_VOTE_MM = .02


def direction_basis(geometry):
    directions = []
    for value in [*geometry['xy_frame'],
                  *(p['normal_xy'] for p in geometry['vertical_planes'])]:
        normal = np.asarray(value, float).copy()
        normal /= np.linalg.norm(normal)
        if normal[np.argmax(np.abs(normal))] < 0:
            normal *= -1
        if not any(abs(normal@old) > .99999 for old in directions):
            directions.append(normal)
    return np.asarray(directions)


def small_surface_planes(xyz, geometry):
    """Recover small face hypotheses using multiscale 2D surface support.

    A width/height test here verifies a sampled surface, not minimum weld length.
    No point, reference weld, process event, or old proposal is removed.
    """
    xyz = np.asarray(xyz, float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Finite raw millimeter XYZ required')
    points = np.unique(xyz, axis=0)
    normals = direction_basis(geometry)
    if len(points) < 8 or not len(normals):
        return [], dict(unique_raw_points=len(points), accepted_by_scale={}, planes=0)
    # Preserve actual sampling multiplicities for neighbourhood selection.
    # Evidence and finite support counts still use distinct coordinates.
    neighbourhood_points = xyz
    tree = cKDTree(neighbourhood_points)
    selected = [[] for _ in normals]
    counts = {k:0 for k in NEIGHBORS}
    for begin in range(0, len(neighbourhood_points), 10000):
        q = neighbourhood_points[begin:begin+10000]
        _, indices = tree.query(q, k=min(max(NEIGHBORS), len(neighbourhood_points)), workers=2)
        for k in NEIGHBORS:
            if k > indices.shape[1]:
                continue
            local = neighbourhood_points[indices[:, :k]]-q[:, None]
            local -= local.mean(1, keepdims=True)
            values, vectors = np.linalg.eigh(np.einsum('nki,nkj->nij', local, local)/k)
            local_normal = vectors[:, :, 0]
            alignment = np.abs(local_normal[:, :2]@normals.T)
            axis = alignment.argmax(1)
            keep = ((values[:, 0] < .05*np.maximum(values[:, 1], 1e-8))
                    & (values[:, 1] > 1e-4) & (np.abs(local_normal[:, 2]) < .15)
                    & (alignment[np.arange(len(q)), axis] > .995))
            counts[k] += int(keep.sum())
            for ni in np.unique(axis[keep]):
                selected[int(ni)].append(q[keep & (axis == ni)])
    origin = points[:, :2].mean(0)
    planes = []; seen = set()
    for ni, batches in enumerate(selected):
        if not batches:
            continue
        evidence = np.unique(np.concatenate(batches), axis=0)
        nominal = normals[ni]
        projected = (evidence[:, :2]-origin)@nominal
        keys, inverse, support = np.unique(np.round(projected/OFFSET_VOTE_MM).astype(np.int64),
                                          return_inverse=True, return_counts=True)
        for group in np.flatnonzero(support >= 2):
            seed = evidence[inverse == group]
            centre = seed[:, :2].mean(0)
            if np.linalg.norm(np.ptp(seed[:, :2], axis=0)) < .5:
                # A vertical column can carry a genuine planar neighbourhood.
                # Its local PCA already established 2D support and direction;
                # retain that point-derived direction rather than fitting a
                # rank-zero XY subset. The complete face is still checked below.
                if np.ptp(seed[:, 2]) < .5:
                    continue
                normal = nominal.copy()
            else:
                _, _, vectors = np.linalg.svd(seed[:, :2]-centre, full_matrices=False)
                normal = vectors[-1]
            if normal@nominal < 0:
                normal *= -1
            if normal@nominal < .99995:
                continue
            offset = float(normal@centre)
            residual = np.abs((seed[:, :2]-centre)@normal)
            if np.quantile(residual, .9) > .10:
                continue
            direction = np.array([-normal[1], normal[0]])
            if direction[np.argmax(np.abs(direction))] < 0:
                direction *= -1
            face = points[np.abs(points[:, :2]@normal-offset) <= PLANE_TOLERANCE_MM]
            along = face[:, :2]@direction
            if len(face) < 6 or np.ptp(along) < .5 or np.ptp(face[:, 2]) < .5:
                continue
            # Only identical numerical hypotheses are removed. This does not
            # merge neighbouring event positions or impose a weld length.
            identity = (*normal.round(9), round(offset, 6))
            if identity in seen:
                continue
            seen.add(identity)
            planes.append(dict(normal_xy=normal.tolist(), direction_xy=direction.tolist(),
                offset_mm=offset, range_mm=[float(along.min()), float(along.max())],
                z_range_mm=[float(face[:, 2].min()), float(face[:, 2].max())],
                support_count=len(face), local_3d_evidence_count=len(seed),
                source='raw_multiscale_small_finite_face'))
    return planes, dict(unique_raw_points=len(points), direction_count=len(normals),
        accepted_by_scale=counts, planes=len(planes),
        faces_under_10mm_height=sum(p['z_range_mm'][1]-p['z_range_mm'][0] < 10. for p in planes),
        sampled_surface_is_not_reviewed_material_identity=True)


def small_face_positions(sequence, geometry, planes):
    original, old_stats = component_positions(sequence, geometry)
    added, new_stats = candidate_positions(sequence, planes, finite=True)
    result = np.unique(np.r_[original, added])
    return result, dict(original_positions=len(original), new_face_positions=len(added),
        total_positions=len(result), original_stats=old_stats, face_stats=new_stats)


def physical_checks():
    geometry = dict(xy_frame=[[1., 0.], [0., 1.]], vertical_planes=[], horizontal_lines=[])
    y, z = np.meshgrid(np.linspace(-6., 6., 13), [0., 1.5, 3.])
    face = np.column_stack((np.full(y.size, 13.25), y.ravel(), z.ravel()))
    planes, stats = small_surface_planes(face, geometry)
    if not planes or min(abs(p['offset_mm']-13.25) for p in planes if abs(p['normal_xy'][0]) > .99) > 1e-6:
        raise ValueError('A sampled 3mm face was dropped')
    sequence = dict(type='horizontal', xyz=np.array([[-20., 0., 3.], [120., 0., 3.]]), direction=np.array([1., 0., 0.]))
    pool, _ = small_face_positions(sequence, geometry, planes)
    if np.abs(pool-33.25).min() > 1e-6:
        raise ValueError('A sub-grid small-face crossing was lost')
    x, y = np.meshgrid(np.arange(-20., 21., 2.), np.arange(-20., 21., 2.))
    flat = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    if small_surface_planes(flat, geometry)[0]:
        raise ValueError('A horizontal sampling grid became vertical faces')
    angle = .731; c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    translation = np.array([1034.5, -842.25, 87.])
    moved_geometry = dict(geometry, xy_frame=(np.asarray(geometry['xy_frame'])@rotation[:2, :2].T).tolist())
    moved_planes, _ = small_surface_planes(face@rotation.T+translation, moved_geometry)
    moved_seq = dict(sequence, xyz=sequence['xyz']@rotation.T+translation, direction=rotation@sequence['direction'])
    moved, _ = small_face_positions(moved_seq, moved_geometry, moved_planes)
    distances = np.abs(moved[:, None]-pool[None, :])
    error = float(max(distances.min(0).max(), distances.min(1).max()))
    if error > 1e-6:
        raise ValueError('Small-face events changed under translation/yaw')
    reverse = dict(sequence, xyz=sequence['xyz'][::-1], direction=-sequence['direction'])
    backwards, _ = small_face_positions(reverse, geometry, planes)
    if not np.allclose(backwards, np.sort(140.-pool), atol=1e-6, rtol=0.):
        raise ValueError('Small-face events changed under reversed traversal')
    narrow = [dict(planes[0], offset_mm=value) for value in (6.2680269, 6.274)]
    close, _ = small_face_positions(sequence, geometry, narrow)
    if not all(np.abs(close-(20.+value)).min() < 1e-6 for value in (6.2680269, 6.274)):
        raise ValueError('Close distinct event hypotheses or a short tail were merged')
    return dict(small_3mm_face_supported=True, horizontal_grid_rejected=True,
        subgrid_crossing_supported=True, traversal_reverse=True, close_events_and_short_tail_retained=True,
        translation_yaw_max_error_mm=error, decoder_output_not_limited_to_pool=True,
        pure_geometry_only=True, full_contract_pass=False)
