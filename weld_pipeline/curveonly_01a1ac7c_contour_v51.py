"""Append complete finite circular contours within native sampling error.

This supplies shape candidates only. Material boundaries and UNKNOWN joint
identity remain explicit; a closed material contour is not a weld decision.
"""
import math
import time
import types
import numpy as np
from . import curveonly_01a1ac7c_composite_v48 as previous

NATIVE_SHAPE_ERROR_MM = previous.ENDPOINT_MM


class NoJointCertificate:
    calls = 0
    broadphase_faces = 0

    def face(self, *args, **kwargs):
        return None


def compatible_circle(a, b):
    x, y = a.get('circle'), b.get('circle')
    if x is None or y is None:
        return False
    return bool(np.linalg.norm(np.asarray(x['center'])-y['center']) <= NATIVE_SHAPE_ERROR_MM
        and abs(x['radius']-y['radius']) <= NATIVE_SHAPE_ERROR_MM
        and abs(np.asarray(x['normal'])@y['normal']) >= 1-1e-8)


def single_circle_shape(candidate):
    fit = candidate.get('circle')
    if fit is None or fit['max_fit_error_mm'] > NATIVE_SHAPE_ERROR_MM:
        return False
    if abs(abs(fit['sweep'])-2*math.pi) > 1e-6:
        return False
    p = np.asarray(candidate['points'])
    if np.linalg.norm(p[0]-p[-1]) > 1e-10:
        return False
    n = np.asarray(fit['normal']); u = np.asarray(fit['start']); v = np.cross(n,u)
    q = p-np.asarray(fit['center'])
    angles = np.unwrap(np.arctan2(q@v,q@u))
    increments = np.diff(angles)*np.sign(fit['sweep'])
    return bool(increments.min(initial=0.) >= -1e-9)


def append(features, packet):
    start = time.perf_counter()
    candidates = packet['candidates']; original_n = packet.get('finite_continuation',{}).get('original_candidates',len(candidates))
    used = {m['curve'] for c in candidates[original_n:] for m in c.get('composite_carrier_members',[])}
    subset = [i for i,c in enumerate(candidates[:original_n]) if i not in used and c.get('material',-1)>=0
        and c.get('STEP') and c.get('pose') and c.get('circle') is not None
        and np.linalg.norm(np.asarray(c['points'])[0]-c['points'][-1])>=.1]
    summary = dict(original_candidate_prefix=len(candidates),new_candidates=0,native_shape_budget_mm=NATIVE_SHAPE_ERROR_MM,
        original_geometry_and_features_preserved=True,unused_original_circle_candidates=len(subset),
        new_point_queries=0,new_finite_face_queries=0,joint_identity_and_process_accepted=False)
    if len(subset)<2:
        return features,dict(packet,finite_contour=dict(summary,seconds=time.perf_counter()-start))
    offsets=[0]; blocks=[]
    for ci in subset:
        a,b=packet['offsets'][ci:ci+2];blocks.append(features[a:b]);offsets.append(offsets[-1]+b-a)
    source=dict(candidates=[candidates[i] for i in subset],offsets=offsets)
    fn=types.FunctionType(previous.append.__code__,dict(previous.__dict__,same_circle=compatible_circle))
    x,p=fn(np.concatenate(blocks),source,NoJointCertificate())
    additions=[];observations=[];final_offsets=list(packet['offsets']);rejected=0
    for j,c in enumerate(p['candidates'][len(subset):],start=len(subset)):
        if not single_circle_shape(c):
            rejected+=1
            continue
        c=dict(c)
        for key in ('composite_carrier_members','source_material_path'):
            c[key]=[dict(m,curve=subset[m['curve']]) for m in c[key]]
        c['source_connections']=[dict(q,left=[subset[q['left'][0]],q['left'][1]],right=[subset[q['right'][0]],q['right'][1]],
            relation=dict(q['relation'],native_shape_budget_mm=NATIVE_SHAPE_ERROR_MM,
                combined_circle_max_fit_error_mm=c['circle']['max_fit_error_mm'],
                certificate='SINGLE_FINITE_CIRCULAR_CONTOUR_SHAPE_ONLY_NO_JOINT_OR_PROCESS_CERTIFICATE')) for q in c['source_connections']]
        c['source_kind']='NATIVE_BUDGET_FINITE_CIRCULAR_CONTOUR_CANDIDATE'
        c['source_shape_certificate']=dict(maximum_fit_error_mm=c['circle']['max_fit_error_mm'],
            native_sampling_budget_mm=NATIVE_SHAPE_ERROR_MM,single_turn_monotonic=True,
            no_points_moved=True,joint_and_process_identity='UNKNOWN')
        a,b=p['offsets'][j:j+2];observations.append(x[a:b]);additions.append(c);final_offsets.append(final_offsets[-1]+b-a)
    out=np.concatenate([features,*observations]) if observations else features
    assert np.array_equal(out[:len(features)],features)
    summary.update(new_candidates=len(additions),shape_rejected=rejected,seconds=time.perf_counter()-start)
    return out,dict(packet,candidates=candidates+additions,offsets=final_offsets,finite_contour=summary)
