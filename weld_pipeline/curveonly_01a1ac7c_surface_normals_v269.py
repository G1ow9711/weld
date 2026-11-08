"""Unsigned finite-mesh normal envelopes; input observations, never weld rules.

The .025mm surface band is the existing triangulation deflection, not a weld
clearance or matching tolerance. All compatible triangle normals are retained.
No orientation or contract-normal accuracy is certified by these descriptors.
"""
import time
import numpy as np
from . import curveonly_01a1ac7c_material_direction_v160 as direction
from .curveonly_01a1ac7c_material_direction_v157 import DIRECTION_EPS_MM

original, indexed, shadow, reference = direction.original, direction.indexed, direction.shadow, direction.reference
DEFLECTION_MM = direction.DEFLECTION_MM
SUPPORT_BAND_MM = DEFLECTION_MM
TIE_MM, LAYER_TIE_MM = 1e-8, direction.LAYER_TIE_MM
NORMAL_NAMES = ('available', 'abs_tangent_min', 'abs_tangent_max',
                'abs_base_min', 'abs_base_max', 'abs_radial_min', 'abs_radial_max')
NAMES = ('geometry_available', 'owner_known', 'frame_available', 'owner_surface_distance_log_mm')+tuple(
    f'{layer}_{name}' for layer in ('owner', 'nearest_layer1', 'nearest_layer2') for name in NORMAL_NAMES)
assert len(NAMES) == 25


def closest_envelope(points, mesh, frames):
    """Exact distances to finite triangles and normals within the mesh band."""
    q, tri, frames = np.asarray(points, float), np.asarray(mesh, float), np.asarray(frames, float)
    assert frames.shape == (len(q), 3, 3) and tri.ndim == 3 and tri.shape[1:] == (3, 3) and len(tri)
    a = tri[:, 0]
    v, w = tri[:, 1]-a, tri[:, 2]-a
    normal = np.cross(v, w)
    square_normal = (normal*normal).sum(1)
    valid_normal = square_normal > 1e-24
    unit = normal/np.maximum(np.sqrt(square_normal)[:, None], 1e-30)
    vv, vw, ww = (v*v).sum(1), (v*w).sum(1), (w*w).sum(1)
    det = vv*ww-vw*vw
    out = dict(distance_mm=np.empty(len(q)), normal_available=np.zeros(len(q), bool),
               normal_min=np.zeros((len(q), 3)), normal_max=np.zeros((len(q), 3)),
               support_triangles=np.zeros(len(q), np.int64))
    batch = max(1, min(64, 100000//len(tri)))
    for lo in range(0, len(q), batch):
        hi = min(len(q), lo+batch)
        x = q[lo:hi]
        delta = x[:, None]-a
        signed = (delta*normal).sum(2)/np.maximum(square_normal, 1e-30)
        plane = x[:, None]-signed[:, :, None]*normal
        av, aw = ((plane-a)*v).sum(2), ((plane-a)*w).sum(2)
        u = (av*ww-aw*vw)/np.maximum(det, 1e-30)
        t = (aw*vv-av*vw)/np.maximum(det, 1e-30)
        inside = valid_normal & (u >= 0) & (t >= 0) & (u+t <= 1)
        best = np.where(inside, ((x[:, None]-plane)**2).sum(2), np.inf)
        for edge in range(3):
            start = tri[:, edge]
            vector = tri[:, (edge+1) % 3]-start
            t = np.clip(((x[:, None]-start)*vector).sum(2)/np.maximum((vector*vector).sum(1), 1e-30), 0, 1)
            candidate = start+t[:, :, None]*vector
            best = np.minimum(best, ((x[:, None]-candidate)**2).sum(2))
        each = np.sqrt(best)
        minimum = each.min(1)
        support = (each <= minimum[:, None]+SUPPORT_BAND_MM+TIE_MM) & valid_normal
        available = support.any(1)
        projection = np.clip(np.abs(np.einsum('nac,tc->nta', frames[lo:hi], unit)), 0, 1)
        low = np.where(support[:, :, None], projection, np.inf).min(1)
        high = np.where(support[:, :, None], projection, -np.inf).max(1)
        out['distance_mm'][lo:hi] = minimum
        out['normal_available'][lo:hi] = available
        out['normal_min'][lo:hi] = np.where(available[:, None], low, 0)
        out['normal_max'][lo:hi] = np.where(available[:, None], high, 0)
        out['support_triangles'][lo:hi] = support.sum(1)
    assert all(np.isfinite(value).all() for value in out.values())
    return out


def closest(points, mesh, frames, cache=None):
    points, mesh, frames = np.asarray(points, float), np.asarray(mesh, float), np.asarray(frames, float)
    cache = {} if cache is None else cache
    entry = cache.get(id(mesh))
    if entry is None or entry['mesh'] is not mesh:
        entry = indexed.index(mesh, False)
        cache[id(mesh)] = entry
    _, seed = entry['tree'].query(points, workers=1)
    seed = np.asarray(seed).reshape(-1)
    upper = np.empty(len(points))
    for ti in np.unique(seed):
        rows = np.flatnonzero(seed == ti)
        upper[rows] = original.closest(points[rows], mesh[ti:ti+1])['distance_mm']
    selected = [{int(ti)} for ti in seed]
    for group in entry['groups']:
        radius = upper+group['radius']+SUPPORT_BAND_MM+TIE_MM+indexed.PRUNE_GUARD_MM
        near = group['tree'].query_ball_point(points, radius, workers=1)
        for values, local in zip(selected, near):
            values.update(map(int, group['ids'][local]))
    batches = {}
    for qi, ids in enumerate(selected):
        batches.setdefault(tuple(sorted(ids)), []).append(qi)
    out = dict(distance_mm=np.empty(len(points)), normal_available=np.zeros(len(points), bool),
               normal_min=np.zeros((len(points), 3)), normal_max=np.zeros((len(points), 3)),
               support_triangles=np.zeros(len(points), np.int64))
    for ids, rows in batches.items():
        rows = np.asarray(rows, int)
        part = closest_envelope(points[rows], mesh[np.asarray(ids, int)], frames[rows])
        for name in out:
            out[name][rows] = part[name]
    return out


def observe(points, owners, frames, frame_available, meshes, cache=None):
    """Owner and two distinct nonowner distance layers, with tie envelopes."""
    points, frames = np.asarray(points, float), np.asarray(frames, float)
    owners, frame_available = np.asarray(owners, int), np.asarray(frame_available, bool)
    known = (owners >= 0) & (owners < len(meshes))
    values = np.zeros((len(points), len(NAMES)), np.float32)
    values[:, :3] = np.column_stack([np.ones(len(points)), known, frame_available])
    cache = {} if cache is None else cache
    boxes = np.asarray([np.r_[m.min(axis=(0, 1)), m.max(axis=(0, 1))] for m in meshes])
    lower = np.linalg.norm(np.maximum(np.maximum(boxes[:, :3][None]-points[:, None],
                                                points[:, None]-boxes[:, 3:][None]), 0), axis=2)
    distance = np.full((len(points), len(meshes)), np.inf)
    available = np.zeros(distance.shape, bool)
    low, high = np.zeros((*distance.shape, 3)), np.zeros((*distance.shape, 3))
    second = np.full(len(points), np.inf)
    queries = 0
    for owner in np.unique(owners[known]):
        indices = np.flatnonzero(owners == owner)
        part = closest(points[indices], meshes[owner], frames[indices], cache)
        queries += len(indices)
        valid = part['normal_available'] & frame_available[indices]
        values[indices, 3] = np.log1p(part['distance_mm'])
        values[indices, 4] = valid
        for axis in range(3):
            values[indices, 5+2*axis] = np.where(valid, part['normal_min'][:, axis], 0)
            values[indices, 6+2*axis] = np.where(valid, part['normal_max'][:, axis], 0)
    for mi in np.argsort(np.median(lower, axis=0), kind='stable'):
        mi = int(mi)
        take = np.flatnonzero(known & (owners != mi) & (lower[:, mi] <= second+LAYER_TIE_MM+indexed.PRUNE_GUARD_MM))
        if not len(take):
            continue
        part = closest(points[take], meshes[mi], frames[take], cache)
        queries += len(take)
        distance[take, mi], available[take, mi] = part['distance_mm'], part['normal_available']
        low[take, mi], high[take, mi] = part['normal_min'], part['normal_max']
        first = distance.min(1)
        second = np.where(distance > first[:, None]+LAYER_TIE_MM, distance, np.inf).min(1)
    first = distance.min(1)
    for li, target in enumerate((first, second)):
        finite = np.isfinite(distance) & np.isfinite(target)[:, None]
        difference = np.full(distance.shape, np.inf)
        np.subtract(distance, target[:, None], out=difference, where=finite)
        tied = finite & (np.abs(difference) <= LAYER_TIE_MM)
        valid = tied & available & frame_available[:, None]
        has = valid.any(1)
        offset = 11+li*7
        values[:, offset] = has
        for axis in range(3):
            a = np.where(valid, low[:, :, axis], np.inf).min(1)
            b = np.where(valid, high[:, :, axis], -np.inf).max(1)
            values[:, offset+1+2*axis] = np.where(has, a, 0)
            values[:, offset+2+2*axis] = np.where(has, b, 0)
    assert np.isfinite(values).all()
    return values, dict(owner_known=known.tolist(), exact_triangle_queries=queries,
        nearest_distinct_layer_distances_mm=[np.where(np.isfinite(x), x, -1).tolist() for x in (first, second)],
        unavailable_distance_minus1_is_receipt_only_not_model_input=True,
        mesh_normal_band_mm=SUPPORT_BAND_MM, no_weld_or_process_decision=True)


def prepare_meshes(packet, solids):
    transform = np.asarray(packet.get('pose', {}).get('transform_STEP_to_XYZ', []))
    assert packet.get('pose', {}).get('passed') and transform.shape == (4, 4) and solids
    basis = shadow.basis(packet['base_axis'])
    origin = transform[:3, 3]
    meshes = []
    for solid in solids:
        # Only OCC's in-process triangulation cache; analytic material is unchanged.
        original.cad.BRepTools.Clean_s(solid)
        triangles = shadow.triangles(solid, DEFLECTION_MM)
        meshes.append(((triangles@transform[:3, :3].T+transform[:3, 3])-origin)@basis)
    return meshes, basis, origin


def build(packet, solids):
    start = time.perf_counter()
    values = np.zeros((packet['offsets'][-1], len(NAMES)), np.float32)
    pose = packet.get('pose', {})
    if not (solids and pose.get('passed') and np.asarray(pose.get('transform_STEP_to_XYZ', [])).shape == (4, 4)):
        return values, dict(available=False, reason='NO_QUALIFIED_FINITE_STEP', geometry_and_GT_retained=True,
                            seconds=time.perf_counter()-start, names=list(NAMES))
    meshes, basis, origin = prepare_meshes(packet, solids)
    owner_at = indexed.owner_function(packet)
    cache, details = {}, []
    for ci, candidate in enumerate(packet['candidates']):
        lo, hi = packet['offsets'][ci:ci+2]
        points = (np.asarray(candidate['stations'])-origin)@basis
        native, receipt = reference.parameters(candidate)
        owners = np.asarray([owner_at(ci, float(s), packet) for s in native])
        closed = np.linalg.norm(np.asarray(candidate['points'])[0]-candidate['points'][-1]) <= 1e-6
        frames, frame_available = direction.station_frames(points, closed)
        part, observed = observe(points, owners, frames, frame_available, meshes, cache)
        values[lo:hi] = part
        details.append(dict(candidate=ci, native_reference=receipt, observation=observed, geometry_retained=True))
    return values, dict(available=True, names=list(NAMES), candidates=details, materials=len(solids),
        mesh_triangles=sum(len(m) for m in meshes), mesh_deflection_mm=DEFLECTION_MM,
        unsigned_normals_not_oriented_contract_attributes=True, all_compatible_triangle_normals_enveloped=True,
        mesh_band_not_weld_clearance_or_matching_tolerance=True, no_labels_models_or_process_decision=True,
        seconds=time.perf_counter()-start)
