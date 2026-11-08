"""Retain original material-edge boundaries inside published complete curves.

Only curve representation is split. Original analytic circles and continuous
polylines, parent identities, real gaps and should-weld decisions are retained.
The process meaning of an original input junction stays UNKNOWN.
"""
import copy
import math
import time
import types
import numpy as np
from . import curveonly_01a1ac7c_boundary_events_v187 as native
from . import curveonly_01a1ac7c_endcap_recovery_v471 as provenance
from . import curveonly_01a1ac7c_arc_publication_v24 as publication

NUMERIC = provenance.NUMERIC
CONVERSION = publication.MAX_ERROR
KINDS = {'NATIVE_EDGE_START', 'NATIVE_EDGE_END', 'MATERIAL_PATH_START', 'MATERIAL_PATH_END'}


def bank(packet, decoded, record=None, solids=None):
    if not packet.get('pose', {}).get('passed') or (solids is None and not (record and record.get('STEP'))):
        return dict(available=False, reason='NO_QUALIFIED_NATIVE_MATERIAL_SOURCE', events=[])
    if solids is None:
        raw, receipt = native.native_edges(record, packet)
    else:
        # Same original sampler/enumeration. Reuse the actual role-bound BReps
        # already passed through the model instead of reopening their file.
        cad = types.SimpleNamespace(**dict(native.cad.__dict__, load_material=lambda _: solids))
        replay = types.FunctionType(native.native_edges.__code__, dict(native.__dict__, cad=cad))
        raw, receipt = replay(dict(STEP=dict(path=None, in_memory_materials=True)), packet)
    if not receipt['available']:
        return dict(available=False, reason=receipt['reason'], events=[], native_replay=receipt)
    view = provenance.published_source_view(decoded, packet)
    events, checks = native.source_events(view, raw)
    chosen = [[e for e in row if e['kind'] in KINDS] for row in events]
    return dict(available=True, events=chosen, native_replay=receipt, original_point_replays=checks,
        original_model_sources=len(packet['candidates']), published_source_count=len(view['candidates']),
        no_features_created_for_post_model_sources=True, boundary_process_meaning='UNKNOWN_RETAINED_NOT_CANCELED')


def arc_point(segment, fraction):
    a = segment['arc_parameters']
    c, u, n = (np.asarray(a[k], float) for k in ('center_mm', 'start_direction', 'plane_normal'))
    angle = float(a['sweep_rad']) * fraction
    return c + float(a['radius_mm']) * (math.cos(angle)*u + math.sin(angle)*np.cross(n,u))


def segment_length(segment):
    if segment.get('geometry_type') == 'arc' and segment.get('arc_parameters'):
        a = segment['arc_parameters']
        return abs(float(a['sweep_rad'])) * float(a['radius_mm'])
    return float(native.cad.arclength(segment['points'])[-1])


def project_segment(point, segment):
    """Every closest parameter; multiple traversals remain ambiguous."""
    length = segment_length(segment)
    if length <= 0:
        return []
    if segment.get('geometry_type') == 'arc' and segment.get('arc_parameters'):
        a = segment['arc_parameters']
        c,u,n = (np.asarray(a[k],float) for k in ('center_mm','start_direction','plane_normal'))
        v = np.cross(n,u)
        delta = point-c
        if math.hypot(float(delta@u),float(delta@v)) <= NUMERIC:
            return []
        theta = math.atan2(float(delta@v),float(delta@u))
        sweep = float(a['sweep_rad'])
        choices = [0.,1.]
        choices += [float(np.clip((theta+k*2*math.pi)/sweep,0,1)) for k in [-1,0,1]]
        rows = [dict(fraction=t, distance=float(np.linalg.norm(point-arc_point(segment,t)))) for t in choices]
    else:
        p = np.asarray(segment['points'],float)
        arc = native.cad.arclength(p)
        delta = np.diff(p,axis=0)
        den = (delta*delta).sum(1)
        valid = np.flatnonzero(den > 1e-24)
        rows = []
        for i in valid:
            t = float(np.clip((point-p[i])@delta[i]/den[i],0,1))
            q = p[i]+t*delta[i]
            rows.append(dict(fraction=float((arc[i]+t*math.sqrt(den[i]))/length),
                             distance=float(np.linalg.norm(point-q))))
    if not rows:
        return []
    best = min(r['distance'] for r in rows)
    found = []
    for row in sorted((r for r in rows if r['distance'] <= best+NUMERIC),key=lambda r:r['fraction']):
        if not found or (row['fraction']-found[-1]['fraction'])*length > NUMERIC:
            found.append(row)
    return found


def slices(segment, fractions):
    """Same directed analytic curve, or the exact original polyline segments."""
    edges = [0.,*sorted(fractions),1.]
    result = []
    analytic = segment.get('geometry_type') == 'arc' and segment.get('arc_parameters')
    points = np.asarray(segment['points'],float)
    cumulative = native.cad.arclength(points)
    full_length = segment_length(segment)
    for left,right in zip(edges[:-1],edges[1:]):
        assert (right-left)*full_length > NUMERIC
        part = copy.deepcopy(segment)
        if analytic:
            a = copy.deepcopy(segment['arc_parameters'])
            angle = float(a['sweep_rad'])*left
            u,n = (np.asarray(a[k],float) for k in ('start_direction','plane_normal'))
            a['start_direction'] = (math.cos(angle)*u+math.sin(angle)*np.cross(n,u)).tolist()
            a['sweep_rad'] = float(a['sweep_rad'])*(right-left)
            a['is_closed'] = False
            part['arc_parameters'] = a
            length = abs(a['sweep_rad'])*float(a['radius_mm'])
            count = max(3,int(math.ceil(length)),int(math.ceil(abs(a['sweep_rad'])/.02)))+1
            sampled = np.asarray([arc_point(part,t) for t in np.linspace(0.,1.,count)])
            if left == 0.: sampled[0] = points[0]
            if right == 1.: sampled[-1] = points[-1]
        else:
            lo,hi = left*full_length,right*full_length
            def at(s): return np.array([np.interp(s,cumulative,points[:,k]) for k in range(3)])
            sampled = np.vstack([at(lo),points[(cumulative>lo)&(cumulative<hi)],at(hi)])
            length = float(native.cad.arclength(sampled)[-1])
        part.update(points=sampled.tolist(),length_mm=float(length),
            native_representation_split=dict(original_segment_index=segment['segment_index'],
                original_fraction_interval=[left,right],geometry_preserved=True,
                process_cause='UNKNOWN_RETAINED_NOT_CANCELED'))
        result.append(part)
    assert abs(sum(p['length_mm'] for p in result)-full_length) <= NUMERIC*max(1.,full_length)
    return result


def correct(decoded, packet, evidence):
    start = time.perf_counter()
    result = copy.deepcopy(decoded)
    audit = dict(branch='PRESERVE_INPUT_NATIVE_BOUNDARIES_V521',changes=[],declined=[],
        should_weld_and_parent_geometry_unchanged=True,no_original_process_boundary_canceled=True,
        boundary_process_cause='UNKNOWN_RETAINED_NOT_CANCELED',no_name_coordinate_radius_or_count_lookup=True,
        original_publication_conversion_limit_mm=CONVERSION,no_new_fit_or_VALcalibration=True)
    if not evidence.get('available'):
        audit['not_applicable'] = evidence.get('reason','UNQUALIFIED_OR_UNKNOWN_NATIVE_INPUT')
        result['native_boundary_export_rule'] = audit
        result['native_boundary_export_seconds'] = time.perf_counter()-start
        return result
    view = provenance.published_source_view(decoded,packet)
    for pi,parent in enumerate(result['parents']):
        ci = parent['source_candidate']
        if not 0 <= ci < len(evidence['events']):
            audit['declined'].append(dict(parent_index=pi,reason='SOURCE_EVENTS_UNAVAILABLE'))
            continue
        candidate = view['candidates'][ci]
        native_points = np.asarray(candidate['points'],float)
        native_s = native.cad.arclength(native_points)
        length = float(native_s[-1])
        a,b = (float(parent['interval'][k]) for k in ('start_mm','end_mm'))
        closed = np.linalg.norm(native_points[0]-native_points[-1]) <= NUMERIC
        targets = []
        for ev in evidence['events'][ci]:
            q = float(ev['station_mm'])
            turns = range(math.floor(a/length)-1,math.ceil(b/length)+1) if closed else [0]
            for turn in turns:
                station = q+turn*length
                if a+NUMERIC < station < b-NUMERIC:
                    targets.append(dict(ev,station_mm=station,native_station_mm=q))
        unique = []
        for ev in sorted(targets,key=lambda e:e['station_mm']):
            if not unique or ev['station_mm']-unique[-1]['station_mm'] > NUMERIC:
                unique.append(ev)
        if not unique: continue
        segments = parent['segments']
        if any(bool(s.get('arc_parameters')) != (s.get('geometry_type') == 'arc') for s in segments):
            audit['declined'].append(dict(parent_index=pi,reason='UNRESOLVED_REPRESENTATION_SCHEMA'))
            continue
        lengths = [segment_length(s) for s in segments]
        offsets = np.r_[0.,np.cumsum(lengths)]
        selected = []
        for ev in unique:
            q = ev['native_station_mm']
            point = np.array([np.interp(q,native_s,native_points[:,k]) for k in range(3)])
            options = []
            for si,segment in enumerate(segments):
                for projection in project_segment(point,segment):
                    options.append(dict(segment=si,**projection,
                        published_station_mm=float(offsets[si]+projection['fraction']*lengths[si])))
            if not options:
                audit['declined'].append(dict(parent_index=pi,source_event=ev,reason='NO_VALID_PUBLISHED_PROJECTION'))
                continue
            best = min(x['distance'] for x in options)
            near = sorted((x for x in options if x['distance'] <= best+NUMERIC),key=lambda x:x['published_station_mm'])
            places = []
            for option in near:
                if not places or option['published_station_mm']-places[-1]['published_station_mm'] > NUMERIC:
                    places.append(option)
            if best > CONVERSION+NUMERIC or len(places) != 1:
                audit['declined'].append(dict(parent_index=pi,source_event=ev,reason='OUTSIDE_OR_AMBIGUOUS_PUBLISHED_CONVERSION',
                    best_distance_mm=best,distinct_places=len(places)))
                continue
            choice = places[0]
            if np.min(np.abs(offsets-choice['published_station_mm'])) <= NUMERIC:
                continue  # An existing emitted boundary already preserves it.
            selected.append(dict(choice,source_event=ev))
        selected.sort(key=lambda x:x['source_event']['station_mm'])
        if any(x['published_station_mm'] >= y['published_station_mm']-NUMERIC for x,y in zip(selected[:-1],selected[1:])):
            audit['declined'].append(dict(parent_index=pi,reason='NON_MONOTONE_OR_REPEATED_SOURCE_BOUNDARY_MAPPING'))
            continue
        if not selected: continue
        split = []
        for si,segment in enumerate(segments):
            cuts = sorted({r['fraction'] for r in selected if r['segment']==si})
            split.extend(slices(segment,cuts) if cuts else [copy.deepcopy(segment)])
        for si,segment in enumerate(split):segment['segment_index']=si
        original_count = len(segments)
        parent['segments'] = split
        parent['native_representation_boundaries'] = selected
        audit['changes'].append(dict(parent_index=pi,source_candidate=ci,before_segments=original_count,
            after_segments=len(split),boundaries=selected,analytic_or_polyline_geometry_exact=True,
            original_parent_points_and_fields_retained=True))
    result['published_segments'] = sum(len(p['segments']) for p in result['parents'])
    result['native_boundary_export_rule'] = audit
    result['native_boundary_export_seconds'] = time.perf_counter()-start
    return result
