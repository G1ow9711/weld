"""Offline CAD/PCD diagnosis of every missed physical proposal endpoint.

No boundary or welding label is changed or used by an inference entrypoint.
"""
import json
import time
from collections import Counter
from pathlib import Path

import gmsh
import numpy as np
from scipy.spatial import cKDTree

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.finite_contact_plate_profiles import FiniteContactPlates

BASE = Path('weld_pipeline/reports/contact_component_candidates_163_20261002')
OUT = Path('weld_pipeline/reports/missing_contact_boundaries_164_20261002')
SPLIT = Path('weld_pipeline/data/full_available_fit_v1/split.json')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def closest_bounded_edge(tag, point):
    coordinate, _ = gmsh.model.getClosestPoint(1, int(tag), np.asarray(point, float).tolist())
    # The installed native kernel can return an uninitialized curve parameter.
    # Recompute it from the valid closest coordinate on the bounded curve.
    parameter = gmsh.model.getParametrization(1, int(tag), np.asarray(coordinate).tolist())
    lower, upper = gmsh.model.getParametrizationBounds(1, int(tag))
    parameter = np.clip(np.asarray(parameter, float), np.asarray(lower), np.asarray(upper))
    closest = np.asarray(gmsh.model.getValue(1, int(tag), parameter.tolist()), float)
    return closest, float(np.linalg.norm(closest-point))


def checked_face_projection(tag, point):
    coordinate, uv = gmsh.model.getClosestPoint(2, int(tag), np.asarray(point, float).tolist())
    coordinate = np.asarray(coordinate, float); uv = np.asarray(uv, float)
    value = np.asarray(gmsh.model.getValue(2, int(tag), uv.tolist()), float)
    if not np.isfinite(uv).all() or np.linalg.norm(value-coordinate) > 1e-6:
        uv = np.asarray(gmsh.model.getParametrization(2, int(tag), coordinate.tolist()), float)
        value = np.asarray(gmsh.model.getValue(2, int(tag), uv.tolist()), float)
        if not np.isfinite(uv).all() or np.linalg.norm(value-coordinate) > 1e-6:
            return coordinate, None
    return coordinate, uv


def contains(face, point):
    coordinate, uv = checked_face_projection(face['face'], point)
    if uv is None or np.linalg.norm(coordinate-point) > .2:
        return False
    if gmsh.model.isInside(2, int(face['face']), uv.tolist(), parametric=True):
        return True
    delta = np.asarray(face['center_mm'])-coordinate; n = np.asarray(face['parametric_normal'])
    delta -= n*(delta@n)
    if np.linalg.norm(delta) < 1e-8:
        return False
    coordinate, uv = checked_face_projection(face['face'], coordinate+.1*delta/np.linalg.norm(delta))
    return bool(uv is not None and gmsh.model.isInside(2, int(face['face']), uv.tolist(), parametric=True))


def finite_edge_preflight():
    gmsh.clear(); a = gmsh.model.occ.addPoint(0., 0., 0.); b = gmsh.model.occ.addPoint(10., 0., 0.)
    line = gmsh.model.occ.addLine(a, b); gmsh.model.occ.synchronize()
    cases = ((np.array([-5., 2., 0.]), np.array([0., 0., 0.])),
             (np.array([15., 2., 0.]), np.array([10., 0., 0.])),
             (np.array([3.25, 2., 0.]), np.array([3.25, 0., 0.])))
    for point, expected in cases:
        closest, _ = closest_bounded_edge(line, point)
        if not np.allclose(closest, expected, atol=1e-6, rtol=0.):
            raise ValueError('Native bounded-edge projection failed; infinite edge prohibited')
    face = gmsh.model.occ.addRectangle(0., 0., 0., 10., 8.); gmsh.model.occ.synchronize()
    descriptor = dict(face=face, center_mm=[5., 4., 0.], parametric_normal=[0., 0., 1.])
    if not contains(descriptor, np.array([3.25, 2.5, 0.])) or contains(descriptor, np.array([15., 2.5, 0.])):
        raise ValueError('Finite surface interior/exterior check failed')
    return dict(boundary_parameter_clamping_verified=True, subsample_projection_verified=True,
                parameter_recomputed_from_curve_coordinate=True, finite_surface_domain_check_verified=True,
                native_projection_numeric_tolerance_mm=1e-6)


def main():
    if OUT.exists():
        raise ValueError('Preserve full missing-boundary diagnostic')
    done = json.loads((BASE/'complete.json').read_text('utf8'))
    config = json.loads((BASE/'config.json').read_text('utf8'))
    split = json.loads(SPLIT.read_text('utf8')); started = time.perf_counter()
    gmsh.initialize([], readConfigFiles=False)
    gmsh.option.setNumber('General.Terminal', 0); gmsh.option.setString('Geometry.OCCTargetUnit', 'MM')
    totals = Counter(); pieces = {}; all_rows = []
    try:
        preflight = finite_edge_preflight(); OUT.mkdir(parents=True)
        protocol = dict(train=config['train'], complete_base_sha256=sha256(BASE/'complete.json'),
            expected_missed_endpoints=989, all_child_denominator=1221, all_endpoint_denominator=2442,
            CAD_face_tolerance_mm=.2, unchanged_face_boundary_nudge_mm=.1,
            CAD_normal_unsigned_match_deg=3., local_PCD_radius_mm=48., membership_probe_mm=[.5, 5.],
            sources={p:sha256(p) for p in [__file__, 'weld_pipeline/audit_step_weld_contacts.py',
                'weld_pipeline/finite_contact_plate_profiles.py']}, native_edge_preflight=preflight,
            STEP_offline_only=True, CAD_or_face_IDs_not_runtime_features=True,
            original_labels_unchanged=True, observed_or_new_held_read=False,
            diagnostic_only=True, reviewed_customer_truth=False, full_contract_pass=False)
        dump(OUT/'config.json', protocol)
        for name in config['train']:
            source_path = BASE/(name+'.json')
            if sha256(source_path) != done['piece_receipts'][name]:
                raise ValueError('Frozen full candidate coverage changed')
            source = json.loads(source_path.read_text('utf8'))
            missed = [r for r in source['endpoints'] if not r['component_covered_3mm']]
            mf = Path(split['piece_roots'][name])/'manifest.json'; manifest = json.loads(mf.read_text('utf8'))
            if sha256(mf) != source['source_manifest_sha256']:
                raise ValueError('Original full child reference changed')
            pcd = Path(manifest['record']['input'])
            if sha256(pcd) != source['input_sha256']:
                raise ValueError('Original point input changed')
            xyz = read_xyz_pcd(pcd)*1000.; tree = cKDTree(xyz); profiles = FiniteContactPlates(xyz)
            rows = []; counts = Counter(); faces = []; edge_cache = {}
            ref_path = Path('weld_pipeline/reports/step_contact_continuity_candidates_v1')/(name+'_attribute_candidates.json')
            cad_path = Path('weld_pipeline/reports/step_geometry_20260930')/(name+'.json')
            reference = None; rotation = translation = None
            if ref_path.exists() and cad_path.exists():
                reference = json.loads(ref_path.read_text('utf8')); cad = json.loads(cad_path.read_text('utf8'))
                step = Path(reference['step'])
                if (not reference['pose_verified_for_reference'] or sha256(step) != reference['step_sha256']
                        or sha256(step) != cad['sha256'] or reference['pcd_sha256'] != source['input_sha256']):
                    raise ValueError('Original STEP/PCD or fixed pose identity changed')
                transform = np.asarray(reference['alignment']['transform'], float)
                rotation, translation = transform[:3, :3], transform[:3, 3]
                if not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-8) or np.linalg.det(rotation) < .99:
                    raise ValueError('A rigid proper registered reference pose is required')
                gmsh.clear(); gmsh.model.occ.importShapes(str(step.resolve())); gmsh.model.occ.synchronize()
                entities = {int(tag) for _, tag in gmsh.model.getEntities(2)}
                for volume in cad['volumes']:
                    for face in volume['planar_faces']:
                        if int(face['face']) not in entities:
                            raise ValueError('Imported STEP face identities changed')
                        faces.append(dict(face, volume=volume['volume'], board_name=volume['name']))
                normals = np.asarray([f['parametric_normal'] for f in faces], float)
                centres = np.asarray([f['center_mm'] for f in faces], float)
            entries = manifest['sequences']; xyz_cache = np.load(mf.parent/'xyz.npy', mmap_mode='r', allow_pickle=False)
            features = np.load(mf.parent/'features.npy', mmap_mode='r', allow_pickle=False)
            line_map = {(row['parent_id'], row['segment']):row['sequence'] for row in
                json.loads((Path('weld_pipeline/reports/physical_event_candidates_161_20261002')/(name+'.json')).read_text('utf8'))['children']}
            welds = {w['id']:w for w in manifest['truth']}
            for original in missed:
                q = np.asarray(original['point_mm'], float); si = line_map[(original['parent_id'], original['segment'])]
                entry = entries[si]; mid = (entry['start']+entry['stop'])//2
                axes = features[mid, 60:66].astype(float).reshape(2, 3)
                axes /= np.linalg.norm(axes, axis=1, keepdims=True)
                segment = welds[original['parent_id']]['segments'][original['segment']]
                ends = np.asarray(segment['points'], float)[[0, -1]]; tangent = ends[-1]-ends[0]
                tangent /= np.linalg.norm(tangent)
                inward = tangent if original['end'] == 0 else -tangent
                plane = profiles.plane(q, axes[0]); raw = dict(nearest_raw_sample_mm=float(tree.query(q)[0]))
                if plane is not None and len(plane['points']):
                    distances = np.linalg.norm(plane['points']-q, axis=1)
                    ids = np.flatnonzero(distances <= 48.); local = plane['points'][ids]
                    raw.update(contact_plane_points=len(plane['points']), nearest_contact_sample_mm=float(distances.min()),
                        local_contact_samples=len(ids), nearest_contact_samples_mm=plane['points'][np.argsort(distances)[:8]].tolist())
                    if len(local):
                        raw['local_offsets_along_inward_and_height_mm'] = np.column_stack(((local-q)@inward, local[:, 2]-q[2]))[
                            np.argsort(np.linalg.norm(local-q, axis=1))[:16]].tolist()
                found = []
                if reference is not None:
                    native = (q-translation)@rotation
                    normal_distance = np.abs(np.einsum('ij,ij->i', centres-native, normals))
                    for fi in np.flatnonzero(normal_distance <= .2):
                        face = faces[int(fi)]
                        if not contains(face, native):
                            continue
                        n = rotation@normals[int(fi)]; matched = np.abs(axes@n) >= np.cos(np.deg2rad(3.))
                        if face['face'] not in edge_cache:
                            edge_cache[face['face']] = [int(tag) for dim, tag in gmsh.model.getBoundary(
                                [(2, int(face['face']))], combined=False, oriented=False, recursive=False) if dim == 1]
                        boundaries = []
                        for tag in edge_cache[face['face']]:
                            position, distance = closest_bounded_edge(tag, native)
                            boundaries.append(dict(edge=tag, type=gmsh.model.getType(1, tag), distance_mm=distance,
                                                   point_world_mm=(rotation@position+translation).tolist()))
                        boundaries.sort(key=lambda row:row['distance_mm'])
                        probes = {}
                        for step_mm in (.5, 5.):
                            probes[str(step_mm)] = dict(inward=contains(face, (q+step_mm*inward-translation)@rotation),
                                                        outward=contains(face, (q-step_mm*inward-translation)@rotation))
                        found.append(dict(face=int(face['face']), volume=int(face['volume']), normal_world=n.tolist(),
                            standing_contact_normal_match=bool(matched[0]), base_contact_normal_match=bool(matched[1]),
                            normal_dot_tangent=float(n@tangent), normal_Z=float(n[2]),
                            closest_boundaries=boundaries[:2], probes=probes))
                standing = [f for f in found if f['standing_contact_normal_match']]
                base = [f for f in found if f['base_contact_normal_match']]
                at_boundary = any(f['closest_boundaries'] and f['closest_boundaries'][0]['distance_mm'] <= .2
                                  for f in standing+base)
                cross = [f for f in found if abs(f['normal_dot_tangent']) > .1]
                state = ('NO_STEP_REFERENCE_AVAILABLE' if reference is None else
                         'FINITE_CONTACT_FACE_BOUNDARY' if at_boundary else
                         'THIRD_FINITE_FACE_CROSSING' if cross else
                         'CONTACT_FACE_INTERIOR_WITHOUT_LOCAL_CAD_EVENT' if standing and base else
                         'FINITE_CAD_CONTACT_UNRESOLVED')
                row = dict(original, diagnostic_state=state, pure_PCD_sampling=raw,
                    finite_CAD_faces=found, standing_faces=len(standing), base_faces=len(base), crossing_faces=len(cross),
                    unsigned_normal_matching_is_not_E4_acceptance=True, source_truth_status='NOT_REVIEWED')
                rows.append(row); counts[state] += 1; totals[state] += 1; all_rows.append(row)
            result = dict(piece=name, audited_missed_endpoints=len(rows), states=dict(counts), rows=rows,
                input_sha256=source['input_sha256'], original_source_manifest_sha256=sha256(mf),
                coverage_receipt_sha256=sha256(source_path), STEP_available=reference is not None,
                step_sha256=reference['step_sha256'] if reference else None,
                pose_reference_sha256=sha256(ref_path) if reference else None,
                cad_description_sha256=sha256(cad_path) if reference else None,
                original_labels_unchanged=True, diagnostic_only=True, full_contract_pass=False)
            dump(OUT/(name+'.json'), result); pieces[name] = {k:v for k, v in result.items() if k != 'rows'}
            print('MISSING_CONTACT_BOUNDARIES', name, 'ENDPOINTS', len(rows), dict(counts), flush=True)
            del xyz, tree, profiles, xyz_cache, features
        if len(all_rows) != 989:
            raise ValueError('All missed endpoints must remain represented')
        groups = {}
        for key, predicate in [('unmarked_workrange', lambda r:r['parent_workrange_any'] is False),
                               ('internal', lambda r:r['role'] == 'internal')]:
            selected = [r for r in all_rows if predicate(r)]
            groups[key] = dict(total=len(selected), states=dict(Counter(r['diagnostic_state'] for r in selected)))
        receipt = dict(config_sha256=sha256(OUT/'config.json'), total_missed_endpoints=len(all_rows),
            states=dict(totals), groups=groups, pieces=pieces,
            piece_receipts={n:sha256(OUT/(n+'.json')) for n in pieces}, elapsed_seconds=time.perf_counter()-started,
            STEP_offline_only=True, no_customer_truth_promotion=True, original_labels_unchanged=True,
            observed_or_new_held_read=False, full_contract_pass=False)
        dump(OUT/'complete.json', receipt); print('MISSING_CONTACT_BOUNDARIES_COMPLETE', dict(totals), groups, flush=True)
    finally:
        gmsh.finalize()


if __name__ == '__main__':
    main()
