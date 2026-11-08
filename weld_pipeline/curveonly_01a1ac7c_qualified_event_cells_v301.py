"""Qualify sampling-band contact runs without deleting original event evidence.

The existing native chord bound is representation uncertainty, never a physical
clearance or welding tolerance. Roots inside a co-carried band are UNKNOWN;
their first/last locations remain interval hypotheses, with every raw root kept.
"""
import time
from collections import Counter
import numpy as np
from . import curveonly_01a1ac7c_structural_event_cells_v298 as original
from . import curveonly_01a1ac7c_sampler_v5 as sampler


def sampling_band(packet):
    audit = packet.get('native_sampling', {})
    expected = audit.get('method') == 'positive_weight_Bezier_control_polygon_subdivision'
    expected &= audit.get('requested_chord_bound_mm') == sampler.TOLERANCE_MM
    expected &= audit.get('labels_or_learned_parameters') is False
    if not expected:
        return None, dict(available=False, reason='DECLARED_NATIVE_SAMPLER_NOT_VERIFIED',
                          no_root_withheld=True)
    band = float(audit['requested_chord_bound_mm'])+original.SURFACE_CONVERSION_MM+original.NUMERIC_MM
    return band, dict(available=True, chord_bound_mm=audit['requested_chord_bound_mm'],
        analytic_circle_sampling_source='Existing native circle uses angle<=.02rad and arc step<=1mm, global chord sagitta<=.0025mm; below declared.025mm.',
        spline_sampling_source='Existing positive-weight Bezier control bound; other native kernel deflection uses.025mm.',
        canonical_conversion_mm=original.SURFACE_CONVERSION_MM, numeric_reference_mm=original.NUMERIC_MM,
        band_mm=band, kernel_and_native_conversion_assumptions_retained=True,
        not_material_gap_or_welding_tolerance=True)


def band_components(points, station, face, band):
    """Entire segments must lie in the band; a cylinder chord through its axis fails."""
    points, station = np.asarray(points, float), np.asarray(station, float)
    kind = face['status']
    if kind == 'CANONICAL_PLANE':
        normal = np.asarray(face['normal']); norm = np.linalg.norm(normal)
        residual = (points@normal-face['offset_mm'])/norm
        keep = np.maximum(abs(residual[:-1]), abs(residual[1:])) <= band
    elif kind == 'CANONICAL_CYLINDER':
        axis = np.asarray(face['axis']); axis = axis/np.linalg.norm(axis)
        delta = points-face['axis_point_mm']; radial = delta-(delta@axis)[:,None]*axis
        vector = np.diff(radial, axis=0); square = (vector*vector).sum(1)
        t = np.clip(-(radial[:-1]*vector).sum(1)/np.maximum(square, 1e-300), 0., 1.)
        minimum = np.linalg.norm(radial[:-1]+t[:,None]*vector, axis=1)
        r = np.linalg.norm(radial, axis=1)
        maximum = np.maximum(r[:-1], r[1:])
        keep = np.maximum(abs(minimum-face['radius_mm']), abs(maximum-face['radius_mm'])) <= band
    else:
        return []
    changes = np.diff(np.r_[False, keep, False].astype(np.int8))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    return [dict(start_mm=float(station[a]), end_mm=float(station[b]),
        first_segment=int(a), last_segment_exclusive=int(b),
        status='WITHIN_NATIVE_SAMPLING_BAND_CO_CARRIED_OR_TANGENT_UNKNOWN') for a,b in zip(starts,ends)]


def qualification_for_roots(points, station, face, roots, band, protected=()):
    components = band_components(points, station, face, band)
    groups = {}; result = {}
    for index, q in roots:
        if any(a-original.NUMERIC_MM <= q <= b+original.NUMERIC_MM for a,b in protected):
            result[index] = dict(keep=True, status='NOMINAL_CLEARANCE_EVENT_RETAINED_UNKNOWN')
            continue
        group = next((i for i,c in enumerate(components)
            if c['start_mm']-original.NUMERIC_MM <= q <= c['end_mm']+original.NUMERIC_MM), None)
        if group is None:
            result[index] = dict(keep=True, status='ROOT_OUTSIDE_CO_CARRIED_BAND_COMPONENT')
        else:
            groups.setdefault(group, []).append((index, q))
    for group, values in groups.items():
        values.sort(key=lambda v:(v[1],v[0]))
        boundary = {values[0][0],values[-1][0]}
        for index, q in values:
            result[index] = dict(keep=index in boundary, component=group,
                status='CO_CARRIED_BAND_EXTREME_HYPOTHESIS_RETAINED' if index in boundary else
                       'SAMPLING_BAND_INTERIOR_ROOT_UNCERTAIN_NOT_A_DISCRETE_BOUNDARY',
                no_should_weld_or_process_confirmation=True)
    return result, components


def qualify(packet, inventory):
    started = time.perf_counter(); band, receipt = sampling_band(packet)
    assert len(packet['candidates']) == len(inventory['candidates'])
    transform = np.asarray(packet['pose']['transform_STEP_to_XYZ'])
    rotation, translation = transform[:3,:3], transform[:3,3]
    output = []; total = Counter()
    for candidate, data in zip(packet['candidates'], inventory['candidates']):
        world = np.asarray(candidate['points']); station = original.cad.arclength(world)
        points = (world-translation)@rotation
        qualifications = {}; runs = {}; by_face = {}
        for index,event in enumerate(data['geometry_events']):
            if event['kind'] == 'FACE_CARRIER_ROOT':
                by_face.setdefault(event['face'], []).append((index,float(event['q_mm'])))
            else:
                qualifications[index] = dict(keep=True,status='NATIVE_RANGE_OR_REPRESENTATION_EVENT_UNCHANGED')
        protected = [(float(v['start_mm']),float(v['end_mm'])) for v in candidate.get('nominal_clearance_intervals', [])
                     if float(v['end_mm']) > float(v['start_mm'])]
        for fi, roots in by_face.items():
            face = inventory['faces'][fi]
            if band is None or not face['face_valid'] or not candidate.get('STEP') or not candidate.get('pose'):
                for index,_ in roots:
                    qualifications[index] = dict(keep=True,status='UNCERTAINTY_QUALIFICATION_UNAVAILABLE_ROOT_RETAINED')
                continue
            local, components = qualification_for_roots(points, station, face, roots, band, protected)
            qualifications.update(local)
            if components: runs[str(fi)] = components
        assert set(qualifications) == set(range(len(data['geometry_events'])))
        selected = [dict(event,raw_event_index=i,qualification=qualifications[i])
            for i,event in enumerate(data['geometry_events']) if qualifications[i]['keep']]
        withheld = [i for i in qualifications if not qualifications[i]['keep']]
        q = original.nodes(selected,data['length_mm'])
        allq = original.nodes(selected+data['learned_events'],data['length_mm'])
        counts = Counter(v['status'] for v in qualifications.values()); total.update(counts)
        output.append(dict(data, qualified_geometry_events=selected,
            raw_event_qualification=[dict(raw_event_index=i,**qualifications[i]) for i in sorted(qualifications)],
            uncertain_interior_raw_event_indices=withheld, co_carried_band_components=runs,
            qualified_geometry_nodes_mm=q, qualified_all_nodes_mm=allq,
            qualified_geometry_cells_mm=list(map(list,zip(q[:-1],q[1:]))),
            qualified_all_cells_mm=list(map(list,zip(allq[:-1],allq[1:]))),
            no_original_root_point_or_model_hypothesis_deleted=True,
            no_confirmed_gap_or_short_weld_removed=True,
            qualified_boundaries_are_hypotheses_not_should_weld=True))
    return dict(inventory,candidates=output,event_qualification_v301=dict(
        native_sampler=receipt,counts=dict(total),seconds=time.perf_counter()-started,
        all_original_events_and_nodes_and_finite_queries_retained=True,
        no_actual_geometry_or_weld_gap_or_process_side_certification=True,
        unknown_contact_does_not_mean_nonweld=True))


def build(packet,solids,neural_states=None,progress=lambda *args:None):
    return qualify(packet,original.build(packet,solids,neural_states,progress))
