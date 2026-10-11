"""Input-only boundary events for line and circular joint carriers.

Plane crossings are proposals, not evidence that a physical plate or weld ends.
Observed finite-face support and nearest-point distance remain explicit. No
labels, CAD, source IDs as features, or guessed process setback distances.
"""
import argparse
import gc
import math
import time
import traceback
import numpy as np
from scipy.spatial import cKDTree, QhullError
from weld_pipeline.sjg_local_runtime_904 import OUT, read, write, digest
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
from weld_pipeline.sjg_finite_face_domains_1235 import face_halfspaces, domains
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_relative_geometry_1135 import transformations


def plane_crossings(curve, normal, offset):
    """Return all isolated intersections, not arbitrary points on coplanar curves."""
    n = np.asarray(normal, float)
    if curve.kind == 'line':
        rate = float(n @ curve.direction)
        if abs(rate) < 1e-10:
            return []
        q = float((offset - n @ curve.ends[0]) / rate)
        return [q] if 0 <= q <= curve.length else []
    arc = curve.arc
    u = arc.start_direction
    v = np.sign(arc.sweep_rad) * np.cross(arc.plane_normal, u)
    a, b = arc.radius_mm * float(n @ u), arc.radius_mm * float(n @ v)
    c = float(n @ arc.center_mm - offset)
    radius = math.hypot(a, b)
    if radius < 1e-10 or abs(c) > radius:
        return []
    phi, alpha = math.atan2(b, a), math.acos(np.clip(-c/radius, -1, 1))
    span = abs(arc.sweep_rad)
    roots = []
    for base in (phi-alpha, phi+alpha):
        for k in range(-2, 3):
            angle = base + k*2*math.pi
            if 0 <= angle <= span:
                roots.append(float(angle*arc.radius_mm))
    return sorted(set(roots))


def controls(dest):
    from weld_pipeline.contract_curve_geometry import CircularArc
    line = CurveCarrier(dict(kind='line', endpoints_mm=[[-10, 0, 0], [10, 0, 0]]))
    n = np.array([1., 0, 0]); offset = 2.
    assert plane_crossings(line, n, offset) == [12.]
    assert plane_crossings(line, [0, 1, 0], 0) == []
    errors = []
    for name, R, shift in transformations():
        moved = CurveCarrier(dict(kind='line', endpoints_mm=(line.ends@R.T+shift).tolist()))
        nn = n@R.T
        root = plane_crossings(moved, nn, offset+nn@shift)
        errors.append(abs(root[0]-12.))
    arc = CircularArc([0, 0, 0], 10, [0, 0, 1], [1, 0, 0], 2*np.pi, True)
    circle = CurveCarrier(dict(kind='arc', arc_parameters=arc.to_parameters()))
    roots = plane_crossings(circle, n, 0)
    assert np.allclose(roots, [5*np.pi, 15*np.pi], atol=1e-10)
    assert max(errors) < 1e-10
    write(dest/'controls.json', dict(line_exact=True, rigid_max_error_mm=max(errors),
        circular_two_intersections=True, coplanar_curve_has_no_fabricated_root=True))


def prepare(key, dest, sequence_run='run390'):
    start = time.perf_counter()
    case = load_case(key); graph = case['graph']; case['x']._mmap.close()
    map_path = OUT/f'{sequence_run}/source_{key}/sequence_map.npz'
    source_receipt = read(OUT/f'{sequence_run}/source_{key}/receipt.json')
    assert digest(map_path) == source_receipt['map_sha256']
    with np.load(map_path) as z:
        pids, rows = z['profile_ids'].copy(), z['admitted_rows'].copy()
    curves = [graph.curve(int(pid)) for pid in pids]
    cohort = next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key'] == key)
    assert digest(OUT/cohort['input_file']) == cohort['input_sha256']
    xyz = np.load(OUT/cohort['input_file'], mmap_mode='r')
    face_hulls = {}; unresolved = []; events = []
    # kind 0: raw carrier limits; 1: joint-face outer hull limits; 2: other plane.
    # Event association IDs are bookkeeping only, not model feature channels.
    for j, curve in enumerate(curves):
        events.extend([(j, 0., -1, 0, 0., 0., 0., 0., False),
                       (j, curve.length, -1, 0, 0., 0., 0., 0., False)])
    with np.load(OUT/f'run185/source_{key}/surface_support.npz') as support:
        for fid, face in enumerate(case['surfaces']):
            sk = face.get('point_ids_key')
            n = np.asarray(face.get('normal', [0, 0, 0]), float)
            norm = float(np.linalg.norm(n))
            if norm < 1e-10 or sk is None or sk not in support:
                unresolved.append(dict(face=fid, reason='missing plane or observed support'))
                continue
            n = n/norm
            offset = float(face['offset'])/norm
            points = np.asarray(xyz[support[sk]], float)
            if len(points) < 3:
                unresolved.append(dict(face=fid, reason='fewer than three observed points'))
                continue
            try:
                A, b, diag = face_halfspaces(points, n)
                face_hulls[fid] = (A, b)
            except (ValueError, QhullError):
                unresolved.append(dict(face=fid, reason='degenerate observed hull'))
                continue
            roots, owners = [], []
            for j, curve in enumerate(curves):
                if fid in graph.profiles[int(pids[j])].get('face_indices', [])[:2]:
                    continue
                for q in plane_crossings(curve, n, offset):
                    roots.append(q); owners.append(j)
            if roots:
                positions = np.array([curves[j].at(q) for j, q in zip(owners, roots)])
                # Chunk the halfspace matrix, retaining exact signed excess.
                excess = np.concatenate([np.max(positions[i:i+512]@A.T+b, axis=1)
                                         for i in range(0, len(positions), 512)])
                tree = cKDTree(points)
                nearest = tree.query(positions, workers=2)[0]
                residual = float(np.max(np.abs(points@n-offset)))
                for j, q, e, distance in zip(owners, roots, excess, nearest):
                    incidence = abs(float(curves[j].tangent(q)@n))
                    events.append((j, q, fid, 2, float(e), float(distance), residual, incidence, bool(e <= 1e-8)))
                del tree
            if fid % 64 == 0:
                print('BOUNDARY_EVENTS', key, 'FACE', fid+1, '/', len(case['surfaces']), 'EVENTS', len(events), flush=True)
    for j, curve in enumerate(curves):
        fids = graph.profiles[int(pids[j])].get('face_indices', [])[:2]
        if len(fids) != 2 or any(fid not in face_hulls for fid in fids):
            continue
        A = np.concatenate([face_hulls[fid][0] for fid in fids])
        b = np.concatenate([face_hulls[fid][1] for fid in fids])
        for lo, hi in domains(curve, A, b):
            events.extend([(j, lo, -1, 1, 0., 0., 0., 0., True),
                           (j, hi, -1, 1, 0., 0., 0., 0., True)])
    events.sort(key=lambda v: (v[0], v[1], v[3], v[2]))
    values = np.array(events, float)
    index = values[:, 0].astype(np.int64)
    offsets = np.r_[0, np.cumsum(np.bincount(index, minlength=len(pids)))]
    folder = dest/f'source_{key}'; folder.mkdir(exist_ok=False)
    np.savez(folder/'boundary_events.npz', profile_ids=pids, admitted_rows=rows, offsets=offsets,
             q_mm=values[:, 1], surface_ids=values[:, 2].astype(np.int32), kinds=values[:, 3].astype(np.uint8),
             hull_signed_excess_mm=values[:, 4], nearest_observed_point_mm=values[:, 5],
             plane_residual_max_mm=values[:, 6], incidence=values[:, 7], finite_support=values[:, 8].astype(bool))
    assert np.isfinite(values).all()
    write(folder/'receipt.json', dict(status='INPUT_BOUNDARY_EVENTS_FROZEN', source=key,
        input_sha256=cohort['input_sha256'], input_map_sha256=source_receipt['map_sha256'],
        events_sha256=digest(folder/'boundary_events.npz'), profiles=len(pids), events=len(events),
        finite_supported_events=int(values[:, 8].sum()), unresolved_faces=unresolved,
        GT_reads=0, STEP_reads=0, fits=0, hull_support_is_not_material_truth=True,
        coplanar_plate_membership_unknown=True, seconds=time.perf_counter()-start))
    xyz._mmap.close()
    print('BOUNDARY_EVENTS_READY', key, len(events), 'FINITE_SUPPORTED', int(values[:, 8].sum()),
          'SECONDS', round(time.perf_counter()-start, 2), flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run', required=True);ap.add_argument('--sources', nargs='+', required=True)
    args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    write(dest/'registered_protocol.json', dict(source_sha256=digest(__file__), sources=args.sources,
        GT_reads=0, STEP_reads=0, fits=0, purpose='geometric endpoint event coverage before model fitting',
        baseline379_preserved=True, external_consultation_paused=True))
    controls(dest)
    try:
        for key in args.sources:
            prepare(key, dest);gc.collect()
    except BaseException:
        write(dest/'failure.json', dict(error=traceback.format_exc()));raise


if __name__ == '__main__':main()
