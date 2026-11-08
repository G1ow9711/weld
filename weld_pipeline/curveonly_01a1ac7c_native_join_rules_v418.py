"""Require an existing model unit to cross a native join before chain recovery.

The crossing margin is the existing endpoint tolerance. Geometric proximity
alone and a set of separate model parents never supply this continuity intent.
Original finite support, real passage and whole-parent guards still apply.
"""
import types
import numpy as np
from . import curveonly_01a1ac7c_wall_base_rules_v398 as old

ENDPOINT_UNCERTAINTY_MM=3.


def continuity(decoded,packet):
    lengths=[float(old.finite.cad.arclength(c['points'])[-1]) for c in packet['candidates']]
    context=types.SimpleNamespace(packet=packet,lengths=lengths)
    parents=[]
    for i,p in enumerate(decoded['parents']):
        r=p['interval'];ci=p['source_candidate']
        if r['start_mm'] < -old.step.NUMERIC or r['end_mm']>lengths[ci]+old.step.NUMERIC:continue
        spans=old.provenance.primitive_intervals(ci,r['start_mm'],r['end_mm'],packet,lengths)
        parents.append((i,p,spans))
    rows=[]
    for ci,c in enumerate(packet['candidates']):
        path=c.get('source_material_path')
        if not(path and c.get('STEP') and c.get('pose')):continue
        points=np.asarray(c['points'])
        if np.linalg.norm(points[0]-points[-1])<=old.step.NUMERIC:continue
        joins=[dict(native_s_mm=float(a['end_mm']),left_curve=int(a['curve']),right_curve=int(b['curve']))
            for a,b in zip(path[:-1],path[1:]) if abs(a['end_mm']-b['start_mm'])<=old.step.NUMERIC]
        if not joins:continue
        incoming=old.provenance.primitive_intervals(ci,0.,lengths[ci],packet,lengths)
        anchors=[]
        for i,parent,spans in parents:
            if old.overlap(incoming,spans)<=ENDPOINT_UNCERTAINTY_MM:continue
            native=old.step.native_parent(parent,context)
            projection=old.aliases.coincident_parameters(native,points)
            if projection is None:continue
            s,error=projection;lo,hi=float(s.min()),float(s.max())
            crossed=[dict(j,left_coverage_mm=j['native_s_mm']-lo,right_coverage_mm=hi-j['native_s_mm'])
                for j in joins if lo < j['native_s_mm']-ENDPOINT_UNCERTAINTY_MM-old.step.NUMERIC
                and hi>j['native_s_mm']+ENDPOINT_UNCERTAINTY_MM+old.step.NUMERIC]
            if crossed:anchors.append(dict(parent_index=i,source_candidate=parent['source_candidate'],
                native_interval_mm=[lo,hi],ordered_projection_error_mm=float(error),crossed_joins=crossed))
        if anchors:rows.append(dict(candidate=ci,source_length_mm=lengths[ci],joins=joins,anchors=anchors))
    return dict(candidates_with_crossing_model_parent=rows,
        endpoint_uncertainty_mm=ENDPOINT_UNCERTAINTY_MM,
        continuity_is_model_intent_evidence_not_final_should_weld_or_process_equivalence=True)


def inspect(decoded,features,packet,inventory,solids,cached_observations=None):
    if not solids or not packet.get('pose',{}).get('passed'):
        return [],dict(branch='V418_MODEL_NATIVE_JOIN_GUARDED_RECOVERY',
            not_applicable='NO_QUALIFIED_FINITE_STEP',input_retained=True),None
    evidence=continuity(decoded,packet)
    admitted={v['candidate']:v for v in evidence['candidates_with_crossing_model_parent']}
    if not admitted:
        return [],dict(branch='V418_MODEL_NATIVE_JOIN_GUARDED_RECOVERY',certificates=[],
            native_model_continuity=evidence,input_retained=True,
            reason='NO_SINGLE_MODEL_PARENT_CROSSES_NATIVE_JOIN_BEYOND_ENDPOINT_UNCERTAINTY'),None
    # Keep all original geometry and source provenance. Reject an ineligible
    # recovery before its dense query; do not alter candidates or model parents.
    def guarded_observe(ci,context,verify=True):
        if ci not in admitted:return dict(candidate=ci,verified_dense=False,
            reason='NO_SINGLE_MODEL_PARENT_CROSS_JOIN_INTENT')
        return old.observation.observe(ci,context,verify=verify)
    observation=types.SimpleNamespace(**dict(old.observation.__dict__,observe=guarded_observe))
    run=types.FunctionType(old.inspect.__code__,dict(old.__dict__,observation=observation))
    cache={ci:cert for ci,cert in (cached_observations or {}).items() if ci in admitted}
    chosen,audit,context=run(decoded,features,packet,inventory,solids,cache)
    for cert in chosen:cert['existing_model_native_join_intent']=admitted[cert['candidate']]
    audit.update(previous_branch=audit['branch'],branch='V418_MODEL_NATIVE_JOIN_GUARDED_RECOVERY',
        native_model_continuity=evidence,
        no_new_geometry_or_process_boundary_equivalence_assertion=True)
    return chosen,audit,context


def correct(decoded,features,packet,inventory,solids,cached_observations=None):
    run=types.FunctionType(old.correct.__code__,dict(old.__dict__,inspect=inspect))
    result=run(decoded,features,packet,inventory,solids,cached_observations)
    result['native_join_recovery_version']=418
    return result
