"""Lossless carrier-to-joint computation map. No targets or STEP are inputs.

Geometric sharing never declares two hypotheses mutually exclusive. Original
profiles remain authoritative for side, finite support, unknown gaps and output.
"""
import json
import math
import numpy as np
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier

NUMERIC_MM = 1e-5  # Numerical curve coincidence, unrelated to 3 mm acceptance.


def relation_key(profile):
    faces = tuple(int(x) for x in profile.get('face_indices', []))
    side = profile.get('weld_side', profile.get('side_hypothesis'))
    return (faces, json.dumps(side, sort_keys=True),
            'LOCAL_FACE_RELATION' if len(faces) == 2 else 'UNRESOLVED_JOINT')


def canonical(curve):
    if curve.kind == 'line':
        direction = curve.direction.copy()
        if direction[np.argmax(np.abs(direction))] < 0:
            direction *= -1
        origin = curve.ends[0] - float(curve.ends[0] @ direction) * direction
        start = float(curve.ends[0] @ direction)
        sign = 1. if curve.direction @ direction > 0 else -1.
        descriptor = np.r_[origin, direction]
        key = ('line', *np.round(origin, 5), *np.round(direction, 9))
        return key, descriptor, start, sign, 0.
    arc = curve.arc
    normal = arc.plane_normal.copy()
    if normal[np.argmax(np.abs(normal))] < 0:
        normal *= -1
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    u = axis - (axis @ normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    start = float(np.arctan2(arc.start_direction @ v, arc.start_direction @ u)
                  % (2 * np.pi) * arc.radius_mm)
    sign = float(np.sign(arc.sweep_rad) * np.sign(arc.plane_normal @ normal))
    descriptor = np.r_[arc.center_mm, normal, arc.radius_mm]
    key = ('arc', *np.round(arc.center_mm, 5), *np.round(normal, 9), round(arc.radius_mm, 5))
    return key, descriptor, start, sign, float(2 * np.pi * arc.radius_mm)


def intervals(start, sign, length, circumference):
    lo, hi = sorted((start, start + sign * length))
    if not circumference:
        return [(lo, hi)]
    if abs(length - circumference) <= 1e-8:
        return [(0., circumference)]
    lo %= circumference
    hi = lo + length
    if hi > circumference:
        return [(lo, circumference), (0., hi - circumference)]
    return [(lo, hi)]


def family_coordinates(curve, descriptor):
    d = np.asarray(descriptor)
    if curve.kind == 'line':
        return float((curve.ends[0]-d[:3])@d[3:6]), float(np.sign(curve.direction@d[3:6]))
    n = d[3:6]; axis = np.eye(3)[np.argmin(np.abs(n))]
    u = axis-(axis@n)*n; u /= np.linalg.norm(u); v = np.cross(n,u)
    radial = curve.at(0)-d[:3]
    return float(np.arctan2(radial@v,radial@u)%(2*np.pi)*d[6]), float(np.sign(curve.arc.sweep_rad)*np.sign(curve.arc.plane_normal@n))


def union(rows):
    result = []
    for lo, hi in sorted(rows):
        # No acceptance-tolerance trimming, bridging or coordinate snapping.
        if result and lo <= result[-1][1]:
            result[-1][1] = max(result[-1][1], hi)
        else:
            result.append([lo, hi])
    return result


def build(profiles):
    relations = []; relation_ids = {}; families = []; family_lookup = {}
    profile_relation = np.empty(len(profiles), np.int32)
    profile_family = np.empty(len(profiles), np.int32)
    transform = np.empty((len(profiles), 3), np.float64)
    original_length = 0.; dense_count = 0; max_numeric_error = 0.
    for index, profile in enumerate(profiles):
        rk = relation_key(profile)
        if rk not in relation_ids:
            relation_ids[rk] = len(relations)
            relations.append(dict(ordered_face_indices=list(rk[0]), side_hypothesis=json.loads(rk[1]),
                                  status=rk[2], material_identity='UNKNOWN', profiles=[], families=[]))
        rid = relation_ids[rk]; curve = CurveCarrier(profile['representation'])
        key, descriptor, start, sign, circumference = canonical(curve)
        # Quantization is a lookup aid, never a merge permission. Verify each
        # match; boundary-bin misses only reduce sharing, never lose a profile.
        lookup = (rid, key); candidates = family_lookup.setdefault(lookup, [])
        fid = None
        for candidate in candidates:
            previous = families[candidate]['descriptor']
            error = float(np.max(np.abs(np.asarray(previous) - descriptor)))
            # Also bound endpoint error: small angular error can grow with span.
            if curve.kind == 'line':
                delta = curve.ends - np.asarray(previous[:3])
                d = np.asarray(previous[3:6])
                error = max(error, float(np.linalg.norm(delta - (delta @ d)[:, None] * d, axis=1).max()))
            else:
                previous = np.asarray(previous)
                error = max(error, float(np.linalg.norm(previous[:3]-descriptor[:3])
                    + abs(previous[6]-descriptor[6])
                    + 2*descriptor[6]*np.linalg.norm(previous[3:6]-descriptor[3:6])))
            if error <= NUMERIC_MM:
                fid = candidate; max_numeric_error = max(max_numeric_error, error); break
        if fid is None:
            fid = len(families); candidates.append(fid); relations[rid]['families'].append(fid)
            families.append(dict(relation=rid, kind=curve.kind, descriptor=descriptor.tolist(),
                                 circumference_mm=circumference, representative_profile=index,
                                 profiles=[], raw_domains=[]))
        # Always use the selected family's frame, including circles whose
        # near-tied normal components could choose a different phase origin.
        start, sign = family_coordinates(curve, families[fid]['descriptor'])
        families[fid]['profiles'].append(index)
        families[fid]['raw_domains'].extend(intervals(start, sign, curve.length, circumference))
        relations[rid]['profiles'].append(index)
        profile_relation[index] = rid; profile_family[index] = fid
        transform[index] = start, sign, curve.length
        original_length += curve.length
        dense_count += max(1, int(math.ceil(curve.length / 2))) + 1
    unique_length = 0.; fine_upper = 0; coarse_upper = 0
    for family in families:
        domains = union(family.pop('raw_domains')); family['domains_mm'] = domains
        length = sum(hi - lo for lo, hi in domains); unique_length += length
        fine_upper += sum(max(1, int(math.ceil((hi-lo)/2)))+1 for lo, hi in domains)
        coarse_upper += sum(max(1, int(math.ceil((hi-lo)/8))) for lo, hi in domains)
    arrays = dict(profile_relation=profile_relation, profile_family=profile_family,
                  local_to_canonical=transform)
    stats = dict(original_profiles=len(profiles), relations=len(relations), geometry_families=len(families),
                 unresolved_relations=sum(r['status']=='UNRESOLVED_JOINT' for r in relations),
                 relation_stage_network_tokens=len(relations), original_length_mm=original_length,
                 shared_unique_length_mm=unique_length, original_dense_2mm_queries=dense_count,
                 all_relations_selected_shared_2mm_upper=fine_upper,
                 all_relations_selected_coarse_8mm_upper=coarse_upper,
                 relation_token_compression=len(profiles)/max(1,len(relations)),
                 duplicate_length_compression=original_length/max(1e-12,unique_length),
                 max_numeric_sharing_error_mm=max_numeric_error,
                 actual_selected_fine_queries=None, selected_query_budget_pass=None,
                 relation_input_budget_pass=len(relations)<=100000,
                 hypotheses_deleted=0, target_reads=0, hard_exclusivity_edges=0)
    assert sorted(i for r in relations for i in r['profiles']) == list(range(len(profiles)))
    assert sorted(i for f in families for i in f['profiles']) == list(range(len(profiles)))
    return dict(relations=relations, families=families, stats=stats), arrays


def controls():
    from weld_pipeline.contract_curve_geometry import CircularArc
    def line(lo,hi,faces=(0,1),side=None,y=0):
        return dict(representation=dict(kind='line',endpoints_mm=[[lo,y,0],[hi,y,0]]),
                    face_indices=list(faces),weld_side=side,material_pair=[7,7])
    profiles=[line(0,100),line(0,50),line(50,100),line(0,100,side='front'),
              line(0,100,side='back'),line(0,100,faces=(2,3)),line(0,100,y=.001),
              line(100,0),line(0,40),line(60,100),line(47,53)]
    graph,a=build(profiles)
    assert len(a['profile_relation'])==11 and graph['stats']['hard_exclusivity_edges']==0
    assert a['profile_family'][0]==a['profile_family'][1]==a['profile_family'][2]==a['profile_family'][7]
    assert len(set(a['profile_family'][[0,3,4,5,6]].tolist()))==5
    assert a['local_to_canonical'][7,1]==-1
    assert graph['stats']['hypotheses_deleted']==0
    assert union([(0,40),(60,100)])==[[0,40],[60,100]]
    arcs=[]
    for phase,sweep in [(0,2*np.pi),(1.75*np.pi,.5*np.pi),(.5*np.pi,.5*np.pi),(np.pi,2*np.pi)]:
        c=CircularArc([0,0,0],100,[0,0,1],[np.cos(phase),np.sin(phase),0],sweep,abs(sweep-2*np.pi)<1e-9)
        arcs.append(dict(representation=dict(kind='arc',arc_parameters=c.to_parameters()),face_indices=[0,1]))
    g,b=build(arcs)
    assert g['stats']['geometry_families']==1 and len(b['profile_family'])==4
    assert abs(g['stats']['shared_unique_length_mm']-200*np.pi)<1e-9
    return dict(ordered_faces_and_sides_not_merged=True,material_ID_not_merge_authority=True,
                parallel_offset_curves_separate=True,adjacent_and_overlapping_hypotheses_retained=True,
                confidence_sum_not_used=True,positive_gaps_not_bridged=True,
                circular_origin_and_reversed_traversal_preserved=True,
                path_score_decoder_not_implemented_here=True,optimizer_updates=0)
