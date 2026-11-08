"""Input finite-face events and arbitrary interval cells on native curve paths.

Analytic carrier roots locate possible events. Exact distance to the ORIGINAL
trimmed face distinguishes a finite witness from a hole or unbounded-plane
extension. Neither observation declares should-weld, a gap, or torch clearance.
"""
import math
import time
from collections import Counter
import numpy as np
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_native_reference_v154 as reference
from . import curveonly_01a1ac7c_exact_mesh_index_v110 as ownership
from tools import probe_DEF_native_NURBS_finite_surface_20261019 as native
from OCP.BRepCheck import BRepCheck_Analyzer

NUMERIC_MM = reference.PROJECTION_NUMERIC_MM
SURFACE_CONVERSION_MM = native.CONVERSION_MM


def plane_roots(points, station, normal, offset):
    points, station, normal = np.asarray(points), np.asarray(station), np.asarray(normal)
    norm = np.linalg.norm(normal)
    y = (points@normal-offset)/norm
    if np.max(np.abs(y)) <= NUMERIC_MM:
        return [], 'COINCIDENT_CARRIER_NOT_A_DISCRETE_CUT'
    roots = []
    for i in range(len(points)-1):
        a, b = float(y[i]), float(y[i+1])
        if abs(a) <= NUMERIC_MM:
            roots.append(float(station[i]))
        if a*b < 0:
            roots.append(float(station[i]+(station[i+1]-station[i])*a/(a-b)))
    if abs(y[-1]) <= NUMERIC_MM:
        roots.append(float(station[-1]))
    return roots, 'POLYLINE_ANALYTIC_PLANE_ROOTS'


def cylinder_roots(points, station, axis, origin, radius):
    points, station = np.asarray(points), np.asarray(station)
    axis = np.asarray(axis, float); axis = axis/np.linalg.norm(axis)
    delta = points-origin
    radial = delta-(delta@axis)[:, None]*axis
    residual = np.linalg.norm(radial, axis=1)-radius
    if np.max(np.abs(residual)) <= SURFACE_CONVERSION_MM+NUMERIC_MM:
        return [], 'COINCIDENT_CYLINDER_CARRIER_NOT_A_DISCRETE_CUT'
    roots = []
    for i, (start, vector) in enumerate(zip(radial[:-1], np.diff(radial, axis=0))):
        a = float(vector@vector); b = 2*float(start@vector); c = float(start@start-radius*radius)
        if a <= np.finfo(float).tiny:
            continue
        disc = b*b-4*a*c
        numerical = 64*np.finfo(float).eps*max(b*b, abs(4*a*c), 1.)
        if disc < -numerical:
            continue
        disc = max(0., disc)
        for t in ((-b-math.sqrt(disc))/(2*a), (-b+math.sqrt(disc))/(2*a)):
            if 0 <= t <= 1:
                roots.append(float(station[i]+t*(station[i+1]-station[i])))
    return roots, 'POLYLINE_ANALYTIC_CYLINDER_ROOTS'


def nodes(events, length):
    values = sorted([0., float(length)]+[min(length, max(0., float(e['q_mm']))) for e in events])
    result = []
    for q in values:
        if not result or q-result[-1] > NUMERIC_MM:
            result.append(q)
    if length-result[-1] <= NUMERIC_MM:
        result[-1] = float(length)
    assert result[0] == 0. and result[-1] == length and np.all(np.diff(result) > 0)
    return result


def model_events(state, length, closed):
    events = []
    for direction in ('forward_steps', 'reverse_steps'):
        reverse = direction == 'reverse_steps'
        for step in state[direction]:
            if not step['continue_selected']:
                continue
            if step['termination'] == 'FULL_CYCLE':
                a, b = 0., length
            else:
                a, b = step['predicted_start_mm'], step['predicted_end_mm']
                if reverse:
                    a, b = length-b, length-a
            for role, q in [('start', a), ('end', b)]:
                if closed and not (abs(q) <= NUMERIC_MM or abs(q-length) <= NUMERIC_MM):
                    q %= length
                events.append(dict(q_mm=float(q), kind='LEARNED_RANGE_HYPOTHESIS_BOUNDARY',
                    role=role, direction=direction, autoregressive_step=step['step'],
                    confidence=step['continue_probability'], geometry_or_process_confirmation=False))
    return events


def build(packet, solids, neural_states=None, progress=lambda *args: None):
    started = time.perf_counter()
    pose = packet.get('pose', {}); transform = np.asarray(pose.get('transform_STEP_to_XYZ', []))
    if not (solids and pose.get('passed') and transform.shape == (4, 4)):
        return dict(available=False, reason='NO_QUALIFIED_FINITE_STEP', candidates=[],
                    no_geometry_or_truth_dropped=True, process_semantics='UNKNOWN')
    rotation, translation = transform[:3, :3], transform[:3, 3]
    assert np.allclose(rotation.T@rotation, np.eye(3), atol=1e-7, rtol=0)
    assert np.linalg.det(rotation) > 0
    table, face_objects, errors = [], [], []
    for mi, solid in enumerate(solids):
        for fi, item in enumerate(cad.items(solid, cad.TopAbs_FACE)):
            face = cad.TopoDS.Face_s(item)
            try:
                shape = native.canonical(face)
                valid = bool(BRepCheck_Analyzer(face, True).IsValid())
            except Exception as exc:
                shape, valid = dict(status='UNKNOWN_CANONICAL_SURFACE', reason=repr(exc)), False
            row = dict(material=mi, local_face=fi, face_valid=valid, **shape)
            table.append(row); face_objects.append(face)
        progress('MATERIAL_FACE_INVENTORY', mi+1, len(solids), len(table))
    owner_at = ownership.owner_function(packet)
    states = {s['candidate']:s for s in neural_states} if neural_states is not None else {}
    candidates, queries = [], 0
    for ci, candidate in enumerate(packet['candidates']):
        world = np.asarray(candidate['points'], float); station = cad.arclength(world)
        length = float(station[-1]); points = (world-translation)@rotation
        geometry_events = [dict(q_mm=0., kind='NATIVE_REPRESENTATION_START', process_semantics='UNKNOWN'),
                           dict(q_mm=length, kind='NATIVE_REPRESENTATION_END', process_semantics='UNKNOWN')]
        coincident = []
        for fi, row in enumerate(table):
            status = row['status']
            if status == 'CANONICAL_PLANE':
                roots, method = plane_roots(points, station, row['normal'], row['offset_mm'])
            elif status == 'CANONICAL_CYLINDER':
                roots, method = cylinder_roots(points, station, row['axis'], np.asarray(row['axis_point_mm']), row['radius_mm'])
            else:
                continue
            if method.startswith('COINCIDENT_'):
                coincident.append(dict(face=fi, material=row['material'], state=method,
                    finite_full_contact_or_process_state='UNKNOWN'))
            for q in sorted(set(roots)):
                point = np.asarray([np.interp(q, station, points[:, k]) for k in range(3)])
                try:
                    distance, closest = native.finite_distance(face_objects[fi], point); queries += 1
                except Exception as exc:
                    errors.append(dict(candidate=ci, face=fi, q_mm=q, error=repr(exc))); continue
                supported = distance <= SURFACE_CONVERSION_MM+NUMERIC_MM
                state = ('FINITE_ORIGINAL_FACE_ROOT_OBSERVED' if supported and row['face_valid'] else
                         'UNKNOWN_INVALID_FACE' if not row['face_valid'] else 'ROOT_OUTSIDE_ORIGINAL_FINITE_FACE')
                owner = int(owner_at(ci, q, packet))
                geometry_events.append(dict(q_mm=q, kind='FACE_CARRIER_ROOT', face=fi,
                    material=row['material'], input_owner_material=owner,
                    same_owner=owner == row['material'], analytic_surface=status, method=method,
                    original_finite_face_distance_mm=float(distance),
                    closest_delta_XYZ_mm=((closest-point)@rotation.T).tolist(),
                    finite_witness_state=state, face_valid=row['face_valid'],
                    process_semantics='UNKNOWN', torch35_applicability='UNKNOWN',
                    infinite_plane_or_cylinder_not_material_extent=True))
        for member in candidate.get('source_material_path', []):
            for name in ('start_mm', 'end_mm'):
                geometry_events.append(dict(q_mm=float(member[name]), kind='NATIVE_MATERIAL_PATH_REPRESENTATION_JOIN',
                    process_semantics='UNKNOWN_NOT_AUTOMATIC_GAP_OR_CUT'))
        learned = model_events(states[ci], length, np.linalg.norm(world[0]-world[-1]) <= 1e-6) if ci in states else []
        geo = nodes(geometry_events, length); allnodes = nodes(geometry_events+learned, length)
        candidates.append(dict(candidate=ci, length_mm=length, closed=bool(np.linalg.norm(world[0]-world[-1]) <= 1e-6),
            geometry_events=geometry_events, learned_events=learned, coincident_carriers=coincident,
            geometry_nodes_mm=geo, all_nodes_mm=allnodes,
            geometry_cells_mm=list(map(list, zip(geo[:-1], geo[1:]))),
            all_cells_mm=list(map(list, zip(allnodes[:-1], allnodes[1:]))),
            no_fixed_interval_or_gap_count=True, should_weld_and_process_state='MODEL_DECISION_REQUIRED',
            no_points_or_native_range_changed=True))
        progress('NATIVE_EVENT_CELLS', ci+1, len(packet['candidates']), len(geometry_events), len(allnodes)-1)
    return dict(available=True, source='ORIGINAL_FINITE_STEP_SURFACE_EVENTS_AND_NATIVE_INTERVAL_CELLS_V298',
        faces=table, candidates=candidates, finite_queries=queries, failed_queries=errors,
        face_status_counts=dict(Counter(row['status'] for row in table)),
        all_input_candidates_retained=True, no_label_or_name_or_coordinate_lookup=True,
        surface_conversion_mm=SURFACE_CONVERSION_MM, numeric_reference_mm=NUMERIC_MM,
        no_process_gap_or_weld_side_inferred_from_geometry_alone=True,
        no35mm_distance_or_gap_width_or_short_weld_filter=True,
        nominal_geometry_not_actual_acceptance=True, seconds=time.perf_counter()-started)
