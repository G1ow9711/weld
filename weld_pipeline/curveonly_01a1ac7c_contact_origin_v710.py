"""Retry failed original T-joint certificates with a numerical origin check.

The old ray search excludes its first NUMERIC millimetres. A classifier may
still call an origin just outside the solid, even within that same numerical
band. A verified finite shell distance in this band is the missing near hit.
All material, source, complete-curve and process conditions remain unchanged.
"""
from __future__ import annotations
import copy,time,types
import numpy as np
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_adaptive_sections_v365 as original
from . import curveonly_01a1ac7c_complete_parent_guard_v367 as guard
from . import curveonly_01a1ac7c_learned_unit_rules_v440 as publication
from . import curveonly_01a1ac7c_native_boundary_provenance_v527 as boundaries
from . import curveonly_01a1ac7c_paired_face_rules_v407 as face_gate
from OCP.TopAbs import TopAbs_SHELL

NUMERIC=original.NUMERIC
CONTACT=original.CONTACT
VERSION='FINITE_CONTACT_RAY_ORIGIN_NUMERICAL_BAND_V710'


def wall_gap(geometry,material,point,direction):
    value=original.old.wall_gap(geometry,material,point,direction)
    if value is not None and value<CONTACT:return value
    cache=getattr(geometry,'_origin_shells_v710',None)
    if cache is None:cache={};geometry._origin_shells_v710=cache
    if material not in cache:cache[material]=list(cad.items(geometry.solids[material],TopAbs_SHELL))
    if not cache[material]:return value
    native=(np.asarray(point)-geometry.t)@geometry.R
    distance=min(cad.finite_distance(shell,native)for shell in cache[material])
    if not np.isfinite(distance)or distance>NUMERIC:return value
    if not hasattr(geometry,'_origin_repairs_v710'):geometry._origin_repairs_v710=[]
    geometry._origin_repairs_v710.append(dict(material=int(material),point_mm=np.asarray(point).tolist(),
        old_positive_ray_gap_mm=value,finite_shell_distance_mm=float(distance),existing_numeric_band_mm=NUMERIC,
        solid_interior_zero_not_used_as_boundary=True))
    return float(distance)


upright=types.FunctionType(original.upright.__code__,dict(original.upright.__globals__,old=types.SimpleNamespace(wall_gap=wall_gap)))
certify=types.FunctionType(original.certify.__code__,dict(original.certify.__globals__,upright=upright))


def correct(decoded,features,packet,solids,inventory,source_expert):
    started=time.perf_counter();result=copy.deepcopy(decoded)
    audit=dict(branch=VERSION,added=[],declined=[],existing_numeric_mm=NUMERIC,existing_contact_mm=CONTACT,
        source_points_base_axis_learned_thresholds_and_process_rules_unchanged=True,original_parents_retained_exact=True,
        no_training_no_workpiece_name_dimension_coordinate_or_fixed_weld_count=True)
    failures=[r for r in decoded.get('structural_rule_v338',{}).get('edge_wall_rejected',[])
        if r.get('reason')=='LOCAL_FINITE_PAIR_SUPPORT_NOT_CONTINUOUS']
    if not failures or not solids or not packet.get('pose',{}).get('passed'):
        audit['not_applicable']='NO_FAILED_ORIGINAL_FINITE_CONTINUITY_CERTIFICATE_OR_NO_QUALIFIED_STEP'
        result['contact_origin_rule']=audit;return result
    lengths=[float(cad.arclength(np.asarray(c['points']))[-1])for c in packet['candidates']]
    if any(not 0<=p.get('source_candidate',-1)<len(lengths)for p in decoded['parents']):
        audit['not_applicable']='UNKNOWN_PUBLISHED_NATIVE_SOURCE_DOMAIN';result['contact_origin_rule']=audit;return result
    selected=[dict(p,candidate=p['source_candidate'])for p in decoded['parents']]
    occupied=set()
    for p in selected:
        q=p.get('interval')
        if q is None:
            audit['not_applicable']='UNKNOWN_PUBLISHED_NATIVE_INTERVAL';result['contact_origin_rule']=audit;return result
        occupied.update(ci for ci,a,b in original.provenance.primitive_intervals(p['candidate'],q['start_mm'],q['end_mm'],packet,lengths))
    rows=original.prior.candidate_rows(features,packet);geometry=None;proposals=[];certificates=[]
    for failure in failures:
        ci=failure['candidate']
        if not isinstance(ci,int)or not 0<=ci<len(lengths):continue
        spans=original.provenance.primitive_intervals(ci,0.,lengths[ci],packet,lengths)
        if any(i in occupied for i,a,b in spans):continue
        conflicts=guard.conflicts(ci,lengths[ci],selected,packet,lengths)
        if conflicts:
            audit['declined'].append(dict(candidate=ci,reason='PRESERVE_EXISTING_CONTINUOUS_PARENT',conflicts=conflicts));continue
        if geometry is None:geometry=original.finite.Geometry(solids,packet,inventory)
        first=len(getattr(geometry,'_origin_repairs_v710',[]))
        cert,reason=certify(ci,rows[ci],packet,geometry,inventory,selected)
        repairs=getattr(geometry,'_origin_repairs_v710',[])[first:]
        if cert is None:
            audit['declined'].append(dict(candidate=ci,original_failure=failure,new_failure=reason,origin_repairs=repairs));continue
        if not repairs:
            audit['declined'].append(dict(candidate=ci,reason='NO_PROVED_NEAR_ORIGIN_REPAIR'));continue
        codes=original.prior.previous.intervals_to_codes(rows[ci],[dict(a=0.,b=lengths[ci],confidence=1.)])
        proposal=original.prior.codec.proposals(packet,[rows[ci]],[codes])[0]
        proposals.append(proposal);certificates.append(dict(candidate=ci,original_failure=failure,complete_certificate=cert,origin_repairs=repairs))
        occupied.update(i for i,a,b in spans)
    if proposals:
        previous=publication.publication._fallback;publication.publication._fallback=publication.corner.fallback
        try:extra=publication.exporter.decode(features,packet,publication.source_runtime.Proposals(proposals,dict(contact_origin_recovery=True)))
        finally:publication.publication._fallback=previous
        extra=boundaries.correct(extra,packet,boundaries.bank(packet,extra,solids=solids))
        combined=dict(decoded,parents=[*decoded['parents'],*extra['parents']])
        chosen,gate_audit=face_gate.inspect(combined,features,packet,inventory,solids,source_expert)
        removed={r['parent_index']for r in chosen};offset=len(decoded['parents']);audit['unchanged_final_face_review']=gate_audit
        if any(i<offset for i in removed):
            audit['declined'].append(dict(reason='NEW_PROPOSAL_CONFLICTS_WITH_EXISTING_FINAL_FACE_DECISION',certificates=certificates))
        else:
            for i,parent in enumerate(extra['parents']):
                if offset+i in removed:
                    audit['declined'].append(dict(candidate=parent['source_candidate'],reason='UNCHANGED_FINAL_FACE_MODEL_REJECTION'));continue
                parent['parent_id']=f"curve_{len(result['parents']):05d}";parent['recovery_source']=VERSION
                result['parents'].append(parent);audit['added'].append(next(c for c in certificates if c['candidate']==parent['source_candidate']))
            result.update(geometry_units=len(result['parents']),published_segments=sum(len(p['segments'])for p in result['parents']))
    assert result['parents'][:len(decoded['parents'])]==decoded['parents']
    audit['seconds']=time.perf_counter()-started;result['contact_origin_rule']=audit;return result
