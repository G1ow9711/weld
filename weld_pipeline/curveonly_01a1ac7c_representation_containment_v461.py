"""Deduplicate certified recovered units using independently checked conversions.

The native source proves identity and containment. Each published output must
separately meet the existing 0.1 mm conversion limit with a continuous bound.
Neither the 0.025 mm source coincidence limit nor the 3 mm score limit changes.
"""
import copy
import math
import numpy as np
from . import curveonly_01a1ac7c_periodic_unit_rules_v445 as topology
from . import curveonly_01a1ac7c_arc_publication_v24 as local

cad, publication = local.cad, local.old
NUMERIC = topology.original.PERIODIC_NUMERIC_MM
ALLOWED_REASONS = {'ARC_TURN_AT_MOST_30_DEGREES', 'BOUNDED_LOCAL_GEOMETRY_REPRESENTATION'}
PROCESS_KEYS = ('split_reason', 'split_reasons', 'segmentation_reason', 'process_events',
                'process_split_points', 'process_breaks', 'true_gap_intervals',
                'real_gap_intervals', 'cut_reason', 'cut_points')


def process_geometry_only(parent):
    if any(parent.get(k) for k in PROCESS_KEYS):
        return False
    segments = parent.get('segments', [])
    return bool(segments) and all(s.get('reason') in ALLOWED_REASONS and
        not any(s.get(k) for k in PROCESS_KEYS) for s in segments)


def recovery_certificate(decoded, parent):
    ci = parent['source_candidate']; a = parent['interval']['start_mm']; b = parent['interval']['end_mm']
    for cert in decoded.get('wall_base_chain_rule', {}).get('certificates', []):
        if cert.get('candidate') != ci or not cert.get('verified_dense'):
            continue
        if not cert.get('existing_model_native_join_intent', {}).get('anchors'):
            continue
        for span in cert.get('anchored_weld_intervals', []):
            if abs(span['start_mm'] - a) <= NUMERIC and abs(span['end_mm'] - b) <= NUMERIC:
                return dict(kind='VERIFIED_FINITE_NATIVE_RECOVERY_WITH_MODEL_CONTINUITY',
                            candidate=ci, start_mm=a, end_mm=b,
                            checked_stations=cert.get('checked_stations'))
    for cert in decoded.get('paired_wall_recovery_rule', {}).get('certificates', []):
        if cert.get('candidate') != ci:
            continue
        geometry = cert.get('finite_geometry_component', {})
        model = cert.get('own_model_coverage', {})
        if not (geometry.get('passed') and geometry.get('verified_dense') and model.get('passed')):
            continue
        if abs(cert['start_mm'] - a) <= NUMERIC and abs(cert['end_mm'] - b) <= NUMERIC:
            return dict(kind='VERIFIED_FINITE_PAIRED_RECOVERY_WITH_OWN_MODEL_COVERAGE',
                        candidate=ci, start_mm=a, end_mm=b)
    return None


def continuous_conversion(parent, graph):
    """Recompute a bidirectional bound; stored vertex residuals are insufficient."""
    value = graph.interval(parent)
    if value is None or not process_geometry_only(parent):
        return dict(passed=False, reason='UNKNOWN_NATIVE_RANGE_OR_PROCESS_SEMANTICS')
    ci, a, b = value
    declared = parent.get('source_conversion_continuous_error_upper_mm',
                          parent.get('source_polyline_conversion_error_upper_mm'))
    if declared is None or not np.isfinite(declared) or not 0 <= declared <= local.MAX_ERROR:
        return dict(passed=False, reason='NO_EXISTING_QUALIFIED_CONVERSION_RECORD')
    native = cad.slice_curve(np.asarray(graph.packet['candidates'][ci]['points']), a, b)
    points = np.asarray(parent.get('points', []), float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        return dict(passed=False, reason='INVALID_PUBLISHED_POINTS')
    pieces=[]; arc_to_source=[]; sagittas=[]; checks=[]
    for index, segment in enumerate(parent['segments']):
        ap = segment.get('arc_parameters')
        if segment.get('geometry_type') != 'arc' or not ap or ap.get('units') != 'mm' or ap.get('is_closed'):
            return dict(passed=False, reason='UNSUPPORTED_OR_CLOSED_REPRESENTATION')
        radius = float(ap['radius_mm']); sweep = float(ap['sweep_rad'])
        center=np.asarray(ap['center_mm'], float); u=np.asarray(ap['start_direction'], float)
        normal=np.asarray(ap['plane_normal'], float); emitted=np.asarray(segment['points'], float)
        if not np.isfinite(np.r_[radius,sweep,center,u,normal]).all() or radius <= 0 or not 0 < abs(sweep) <= local.TURN_LIMIT + 1e-12:
            return dict(passed=False, reason='INVALID_ARC_PARAMETERS')
        if max(abs(np.linalg.norm(u)-1.), abs(np.linalg.norm(normal)-1.), abs(u@normal)) > NUMERIC:
            return dict(passed=False, reason='INVALID_ARC_FRAME')
        if len(emitted) < 2 or emitted.ndim != 2 or emitted.shape[1] != 3:
            return dict(passed=False, reason='INVALID_ARC_SAMPLING')
        v=np.cross(normal,u)
        angles=np.linspace(0.,sweep,len(emitted))
        analytic=center+radius*(np.cos(angles)[:,None]*u+np.sin(angles)[:,None]*v)
        residual=float(np.linalg.norm(analytic-emitted,axis=1).max())
        if residual > NUMERIC or abs(float(segment['length_mm'])-radius*abs(sweep)) > NUMERIC:
            return dict(passed=False, reason='PUBLISHED_ARC_POINTS_OR_LENGTH_DISAGREE')
        if pieces and np.linalg.norm(pieces[-1][-1]-emitted[0]) > NUMERIC:
            return dict(passed=False, reason='ACTUAL_PUBLISHED_GAP_PRESERVED')
        pieces.append(emitted)
        count=max(3,int(math.ceil(radius*abs(sweep)/local.QUERY_SPACING))+1)
        grid=np.linspace(0.,sweep,count)
        queries=center+radius*(np.cos(grid)[:,None]*u+np.sin(grid)[:,None]*v)
        upper=float(publication.distance_upper(queries,native).max()+radius*abs(sweep)/(count-1)/2)
        sagitta=float(radius*(1-math.cos(sweep/(len(emitted)-1)/2)))
        arc_to_source.append(upper);sagittas.append(sagitta)
        checks.append(dict(segment_index=index,analytic_point_residual_mm=residual,
                           analytic_to_native_continuous_bound_mm=upper))
    joined=np.vstack([pieces[0],*[p[1:] for p in pieces[1:]]])
    if joined.shape != points.shape or np.linalg.norm(joined-points,axis=1).max() > NUMERIC:
        return dict(passed=False, reason='PARENT_AND_CHILD_GEOMETRY_DISAGREE')
    source_queries=cad.resample(native,local.QUERY_SPACING)
    length=float(cad.arclength(native)[-1])
    source_to_arc=float(publication.distance_upper(source_queries,points).max()+length/(len(source_queries)-1)/2+max(sagittas))
    bound=max(source_to_arc,max(arc_to_source))
    return dict(passed=bound <= local.MAX_ERROR, continuous_error_upper_mm=bound,
                source_to_analytic_bound_mm=source_to_arc,
                analytic_to_source_bound_mm=max(arc_to_source),
                unchanged_publication_limit_mm=local.MAX_ERROR,
                source_samples=len(source_queries), original_recorded_residual_mm=float(declared),
                original_record_not_assumed_continuous=True,
                query_spacing_mm=local.QUERY_SPACING, segment_checks=checks,
                endpoint_inclusive_arclength_Lipschitz_bound=True)


def correct(decoded, packet):
    parents=decoded['parents']; graph=topology.NativeUnits(packet)
    certified={i: recovery_certificate(decoded,p) for i,p in enumerate(parents)}
    covers=[i for i,c in certified.items() if c is not None and graph.open(parents[i])]
    evidence=[]; declined=[]; checked={}
    for i, parent in enumerate(parents):
        if not graph.open(parent) or not process_geometry_only(parent):
            continue
        for j in covers:
            other=parents[j]
            if i==j or parent['source_candidate'] != other['source_candidate']:
                continue
            if not topology.original.same_category(parent,other):
                continue
            native=graph.contains(other,parent)
            if native is None:
                continue
            a,b=other['interval']['start_mm'],other['interval']['end_mm']
            x,y=parent['interval']['start_mm'],parent['interval']['end_mm']
            # Strict containment in this exact original source, not a length cutoff.
            if x < a-NUMERIC or y > b+NUMERIC or b-a <= y-x+NUMERIC:
                continue
            for k in (i,j):
                if k not in checked:checked[k]=continuous_conversion(parents[k],graph)
            if not (checked[i]['passed'] and checked[j]['passed']):
                declined.append(dict(parent_index=i,covering_parent_index=j,
                    reason='CONVERSION_NOT_INDEPENDENTLY_CERTIFIED',
                    inner=checked[i],outer=checked[j]));continue
            budget=checked[i]['continuous_error_upper_mm']+checked[j]['continuous_error_upper_mm']+NUMERIC
            sanity=float(publication.distance_upper(parent['points'],other['points']).max())
            if sanity > budget:
                declined.append(dict(parent_index=i,covering_parent_index=j,
                    reason='INDEPENDENT_PUBLISHED_GEOMETRY_SANITY_FAILED',error_mm=sanity,budget_mm=budget));continue
            evidence.append(dict(removed_parent_index=i,covering_parent_index=j,
                candidate=parent['source_candidate'],native_containment=native,
                recovery=certified[j],inner_conversion=checked[i],outer_conversion=checked[j],
                published_vertex_sanity_error_mm=sanity,derived_conversion_budget_mm=budget,
                legacy_inner_closed_flag=parent.get('is_closed'),canonical_native_interval_is_open=True,
                no_length_threshold=True));break
    remove={c['removed_parent_index'] for c in evidence};result=copy.deepcopy(decoded)
    if remove:
        result['parents']=[p for i,p in enumerate(result['parents']) if i not in remove]
        for i,p in enumerate(result['parents']):p['parent_id']=f'curve_{i:05d}'
        result['geometry_units']=len(result['parents'])
        result['published_segments']=sum(len(p['segments']) for p in result['parents'])
    result['representation_containment_rule']=dict(branch='V461_NATIVE_IDENTITY_AND_CHECKED_CONTINUOUS_CONVERSIONS',
        certificates=evidence,declined=declined,removed_parent_indices=sorted(remove),
        unchanged_native_source_projection_and_material_gap_tolerances=True,
        actual_closed_and_explicit_process_units_preserved=True,
        no_new_fit_or_VALcalibration=True,retained_fields_exact_except_parent_id=True)
    return result
