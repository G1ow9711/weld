"""Typed finite-input boundary candidates, without any weld or cut decision."""
import math
from collections import Counter
import numpy as np
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_sampler_v5 as sampler
from . import curveonly_01a1ac7c_native_reference_v154 as reference
from . import curveonly_01a1ac7c_material_sections_v124 as sections
from OCP.GeomAbs import GeomAbs_BSplineCurve

PROJECTION_MM = sampler.TOLERANCE_MM + 1e-6


def project(points, query):
    """Exact point-to-polyline projection, not nearest-vertex distances."""
    points = np.asarray(points, float)
    query = np.asarray(query, float).reshape(-1, 3)
    arc = cad.arclength(points)
    delta = np.diff(points, axis=0)
    den = (delta*delta).sum(1)
    valid = den > 0
    a, delta, den = points[:-1][valid], delta[valid], den[valid]
    start = arc[:-1][valid]
    coordinates, distances = [], []
    for q in query:
        t = np.clip(((q-a)*delta).sum(1)/den, 0, 1)
        d = np.linalg.norm(q-a-t[:, None]*delta, axis=1)
        i = int(np.argmin(d))
        coordinates.append(start[i]+t[i]*np.sqrt(den[i]))
        distances.append(d[i])
    return np.asarray(coordinates), np.asarray(distances)


def knots(edge):
    ad = cad.BRepAdaptor_Curve(edge)
    if ad.GetType() != GeomAbs_BSplineCurve:
        return []
    spline = ad.BSpline()
    a, b = float(ad.FirstParameter()), float(ad.LastParameter())
    result = []
    eps = 32*np.finfo(float).eps*max(1., abs(a), abs(b))
    for i in range(1, spline.NbKnots()+1):
        value = float(spline.Knot(i))
        if spline.IsPeriodic():
            period = float(spline.Period())
            values = [value+j*period for j in range(
                math.floor((a-value)/period)-1, math.ceil((b-value)/period)+2)]
        else:
            values = [value]
        for q in values:
            if a+eps < q < b-eps:
                result.append(dict(parameter=q, multiplicity=int(spline.Multiplicity(i)),
                                   point=cad.vec(ad.Value(q)).tolist()))
    return result


def native_edges(record, packet):
    """Repeat only the original material-edge sampler and dedup, without queries."""
    transform = np.asarray(packet.get('pose', {}).get('transform_STEP_to_XYZ', []))
    if not (record.get('STEP') and packet.get('pose', {}).get('passed')
            and transform.shape == (4, 4)):
        return [], dict(available=False, reason='NO_QUALIFIED_NATIVE_STEP_SOURCE',
                        all_input_geometry_retained=True)
    solids = cad.load_material(record['STEP']['path'])
    raw, seen = [], set()
    sampler.reset_audit()
    for material, solid in enumerate(solids):
        for value in cad.items(solid, cad.TopAbs_EDGE):
            edge = cad.TopoDS.Edge_s(value)
            points = sampler.curve_points(edge)
            if points is None or len(points) < 3 or cad.arclength(points)[-1] < 1e-4:
                continue
            world = points@transform[:3, :3].T+transform[:3, 3]
            samples = cad.resample(world, max(cad.arclength(world)[-1]/16., .1))
            signature = min(np.round(samples, 3).tobytes(),
                            np.round(samples[::-1], 3).tobytes())
            if signature in seen:
                continue
            seen.add(signature)
            original_knots = knots(edge)
            queries = [q['point'] for q in original_knots]
            coordinates, error = project(points, queries)
            assert error.max(initial=0) <= PROJECTION_MM
            typed = [dict(station_mm=float(s), kind='NATIVE_BSPLINE_KNOT',
                          primitive=len(raw), knot_parameter=q['parameter'],
                          knot_multiplicity=q['multiplicity'],
                          projection_error_mm=float(d))
                     for s, d, q in zip(coordinates, error, original_knots)]
            raw.append(dict(points=world, material=material, events=typed))
    expected = packet['carrier_union']['original_candidates']
    assert len(raw) == expected, ('Native primitive enumeration changed', len(raw), expected)
    return raw, dict(available=True, material_solids=len(solids), primitives=len(raw),
                     original_primitive_count_exact=True,
                     knots=sum(len(q['events']) for q in raw),
                     max_knot_projection_mm=max((e['projection_error_mm'] for q in raw
                                                 for e in q['events']), default=0.),
                     no_contact_or_point_neighborhood_queries=True,
                     invalid_material_does_not_veto_verified_curve_shape=True)


def source_events(packet, raw):
    candidates = packet['candidates']
    cache, checks = {}, []

    def one(ci, stack=()):
        if ci in cache:
            return cache[ci]
        assert ci not in stack
        candidate = candidates[ci]
        points = np.asarray(candidate['points'])
        length = float(cad.arclength(points)[-1])
        closed = np.linalg.norm(points[0]-points[-1]) <= 1e-6
        events = [dict(station_mm=0., kind='SOURCE_ORIGIN' if closed else 'SOURCE_START'),
                  dict(station_mm=length, kind='SOURCE_ORIGIN' if closed else 'SOURCE_END')]
        path = candidate.get('source_material_path')
        if path:
            for member in path:
                start, end = float(member['start_mm']), float(member['end_mm'])
                events.extend([dict(station_mm=start, kind='MATERIAL_PATH_START'),
                               dict(station_mm=end, kind='MATERIAL_PATH_END')])
                child = int(member['curve'])
                child_length = float(cad.arclength(candidates[child]['points'])[-1])
                assert abs(end-start-child_length) <= 1e-6
                for event in one(child, stack+(ci,)):
                    s = child_length-event['station_mm'] if member['reversed'] else event['station_mm']
                    events.append(dict(event, station_mm=start+s, child_source=child))
        elif candidate.get('original_native_members') and raw:
            cursor, last, parts = 0., None, []
            for member in candidate['original_native_members']:
                primitive = int(member['curve'])
                source = raw[primitive]
                p = source['points'][::-1] if member['reversed'] else source['points']
                if last is not None:
                    cursor += float(np.linalg.norm(last-p[0]))
                span = float(cad.arclength(p)[-1])
                events.extend([dict(station_mm=cursor, kind='NATIVE_EDGE_START', primitive=primitive),
                               dict(station_mm=cursor+span, kind='NATIVE_EDGE_END', primitive=primitive)])
                for event in source['events']:
                    s = span-event['station_mm'] if member['reversed'] else event['station_mm']
                    events.append(dict(event, station_mm=cursor+s))
                cursor += span
                last = p[-1]
                parts.append(p)
            replay = np.vstack(parts)
            assert replay.shape == points.shape and np.allclose(replay, points, atol=1e-9, rtol=0), ci
            assert abs(cursor-length) <= 1e-6, ci
            checks.append(dict(candidate=ci, original_primitive_points_replayed=True,
                               maximum_point_error_mm=float(np.abs(replay-points).max())))
        for event in events:
            assert -1e-7 <= event['station_mm'] <= length+1e-7, (ci, event, length)
            event['station_mm'] = float(np.clip(event['station_mm'], 0, length))
            event['weld_or_process_cut'] = 'UNKNOWN'
        cache[ci] = events
        return events

    return [one(ci) for ci in range(len(candidates))], checks


def observed_events(features, local_values, packet, ci, station):
    start, end = packet['offsets'][ci:ci+2]
    solid = np.asarray(features[start:end, 34:])
    local = local_values[start:end]
    result = []
    for index in np.flatnonzero(np.any(np.abs(np.diff(solid, axis=0)) > 1e-6, axis=1)):
        a, b = solid[index:index+2]
        changed = np.flatnonzero(np.abs(a-b) > 1e-6)
        # Retain unknown/side-sort transitions as observations, not physical cuts.
        result.append(dict(station_mm=float((station[index]+station[index+1])/2),
                           kind='OBSERVED_SOLID_STATE_TRANSITION',
                           channels=[sections.NAMES[j] for j in changed],
                           bracket_mm=[float(station[index]), float(station[index+1])],
                           weld_or_process_cut='UNKNOWN', unordered_side_sort_can_change=True))
    for column, known, threshold, kind in (
            (1, 0, np.log1p(sampler.TOLERANCE_MM), 'OBSERVED_PAIR_MESH_CONTACT_BAND'),
            (9, 6, np.log1p(35.), 'OBSERVED_THIRD_PROJECTED_BOUNDARY_35MM'),
            (8, 6, 0., 'OBSERVED_THIRD_CLOSEST_HEIGHT_ZERO'),
            (10, 6, 0., 'OBSERVED_THIRD_BOUNDARY_MIN_HEIGHT_ZERO'),
            (11, 6, 0., 'OBSERVED_THIRD_BOUNDARY_MAX_HEIGHT_ZERO')):
        v = local[:, column]-threshold
        crossing = (np.sign(v[:-1]) != np.sign(v[1:])) & (local[:-1, known] > .5) & (local[1:, known] > .5)
        for index in np.flatnonzero(crossing):
            fraction = -v[index]/(v[index+1]-v[index])
            s = station[index]+fraction*(station[index+1]-station[index])
            result.append(dict(station_mm=float(s), kind=kind,
                               bracket_mm=[float(station[index]), float(station[index+1])],
                               interpolation='LINEAR_BETWEEN_EXISTING_OBSERVATIONS',
                               weld_or_process_cut='UNKNOWN', confirmed_35mm_rule_admissible='UNKNOWN'))
    return result


def build(record, features, packet):
    raw, receipt = native_edges(record, packet)
    sources, checks = source_events(packet, raw)
    local_values = np.asarray(packet['local_finite_material_v94']['values'])
    result = []
    for ci, (candidate, events) in enumerate(zip(packet['candidates'], sources)):
        station, _ = reference.parameters(candidate)
        values = events+observed_events(features, local_values, packet, ci, station)
        values.sort(key=lambda q: (q['station_mm'], q['kind']))
        positions = np.unique([q['station_mm'] for q in values])
        geometry_positions = np.unique([q['station_mm'] for q in events])
        result.append(dict(candidate=ci, length_mm=float(cad.arclength(candidate['points'])[-1]),
                           closed=bool(np.linalg.norm(np.asarray(candidate['points'])[0]-candidate['points'][-1]) <= 1e-6),
                           events=values, distinct_positions=len(positions),
                           geometry_distinct_positions=len(geometry_positions),
                           potential_open_pairs=int(len(positions)*(len(positions)-1)//2),
                           kind_counts=dict(Counter(q['kind'] for q in values)),
                           min_distinct_spacing_mm=float(np.diff(positions).min()) if len(positions)>1 else None))
    return dict(candidates=result, native_replay=receipt, replayed_original_sources=checks,
                all_input_candidates_retained=True, no_truth_model_or_weld_decision=True,
                no_along_arc_35mm_offsets=True, original_geometry_unchanged=True)
