"""Resolve endpoint uncertainty only at an independently certified native join.

A general interior model gap stays protected. Both native outer endpoints must
already coincide, and actual finite end caps must map to the same interior
boundary of the selected opposite complete model unit.
"""
import types
import numpy as np
from . import curveonly_01a1ac7c_endcap_connections_v467 as original
from . import curveonly_01a1ac7c_representation_containment_v461 as semantics

kernel,old,step,finite=original.kernel,original.old,original.step,original.finite
NUMERIC=original.NUMERIC
references=original.references


def finite_cap_mapping(ci,q,reference,context):
    # Query the existing finite end-face alternative even when a general
    # through-wall ray happens to succeed at the same boundary.
    caps=original.caps
    proxy=types.SimpleNamespace(**dict(caps.previous.previous.__dict__,
        counterpart=lambda *args:dict(passed=False,reason='REQUIRE_ACTUAL_FINITE_END_FACE_WITNESS')))
    fn=types.FunctionType(caps.previous.counterpart.__code__,
        dict(caps.previous.__dict__,previous=proxy,convex_caps=caps.convex_caps))
    return fn(ci,q,reference['candidate'],reference['interval'],context)


def actual_join(ci,join,left,right,reference,context):
    lroot,rroot=left['root'],right['root']
    lq=0. if left['reversed'] else context.lengths[lroot]
    rq=context.lengths[rroot] if right['reversed'] else 0.
    a,_=finite.point_tangent(np.asarray(context.packet['candidates'][lroot]['points']),
        finite.cad.arclength(context.packet['candidates'][lroot]['points']),lq)
    b,_=finite.point_tangent(np.asarray(context.packet['candidates'][rroot]['points']),
        finite.cad.arclength(context.packet['candidates'][rroot]['points']),rq)
    if np.linalg.norm(a-b)>NUMERIC:return None
    x=finite_cap_mapping(lroot,lq,reference,context);y=finite_cap_mapping(rroot,rq,reference,context)
    if not(x.get('passed') and y.get('passed')):return None
    if x.get('method')!='ACTUAL_CONVEX_FINITE_END_FACE' or y.get('method')!='ACTUAL_CONVEX_FINITE_END_FACE':return None
    if x['wall']==y['wall'] or x['base']!=y['base'] or x['base']!=reference['base']:return None
    if abs(x['peer_native_s_mm']-y['peer_native_s_mm'])>NUMERIC:return None
    if np.linalg.norm(np.asarray(x['inner_point_mm'])-y['inner_point_mm'])>NUMERIC:return None
    peer=x['peer_native_s_mm'];interval=reference['interval']
    if not interval['start_mm']+original.MODEL_ENDPOINT_MM<peer<interval['end_mm']-original.MODEL_ENDPOINT_MM:return None
    boundaries=[dict(left=u,right=v) for u,v in zip(kernel.leaves(reference['candidate'],context)[:-1],
        kernel.leaves(reference['candidate'],context)[1:]) if abs(u['end_mm']-peer)<=NUMERIC and
        abs(v['start_mm']-peer)<=NUMERIC and {u['material'],v['material']}=={x['wall'],y['wall']}]
    if len(boundaries)!=1:return None
    return dict(candidate=ci,native_join_mm=join,actual_native_outer_endpoint_error_mm=float(np.linalg.norm(a-b)),
        reference_candidate=reference['candidate'],peer_native_join_mm=peer,
        left_finite_cap=x,right_finite_cap=y,peer_material_boundary=boundaries[0],
        no_actual_source_gap_filled=True)


def own_coverage(decoded,packet,context,ci,start,end):
    checked=original.own_coverage(decoded,packet,context,ci,start,end)
    if checked.get('reason')!='PRESERVE_UNEXPLAINED_INTERIOR_OWN_MODEL_GAP':return checked
    ranges=checked['model_ranges_mm'];eps=max(kernel.source_uncertainty(v['root'],context) for v in kernel.leaves(ci,context))
    gaps=[(a[1],b[0]) for a,b in zip(ranges[:-1],ranges[1:]) if b[0]-a[1]>2*eps]
    refs=original.references(decoded,packet);leaves=kernel.leaves(ci,context);certificates=[];consistent=None
    for a,b in gaps:
        options=[]
        for left,right in zip(leaves[:-1],leaves[1:]):
            join=left['end_mm']
            if abs(join-right['start_mm'])>NUMERIC or not a-NUMERIC<=join<=b+NUMERIC:continue
            if join-a>original.MODEL_ENDPOINT_MM or b-join>original.MODEL_ENDPOINT_MM:continue
            for ref in refs:
                cert=actual_join(ci,join,left,right,ref,context)
                if cert is not None:options.append(cert)
        allowed={c['reference_candidate'] for c in options}
        if not allowed:return checked
        consistent=allowed if consistent is None else consistent&allowed
        if not consistent:return checked
        certificates.append(dict(model_gap_mm=[a,b],native_join_certificates=options))
    incoming=old.provenance.primitive_intervals(ci,start,end,packet,context.lengths);indices=[]
    for i,p in enumerate(decoded['parents']):
        r=p['interval'];spans=old.provenance.primitive_intervals(p['source_candidate'],r['start_mm'],r['end_mm'],packet,context.lengths)
        if old.overlap(incoming,spans)>NUMERIC:
            if any(p.get(k) for k in [*semantics.PROCESS_KEYS,'_confirmed_process_cut_ends']):return checked
            if any(any(s.get(k) for k in [*semantics.PROCESS_KEYS,'_confirmed_process_cut_ends']) for s in p.get('segments',[])):return checked
            indices.append(i)
    assert indices and len({decoded['parents'][i].get('weld_side') for i in indices})==1
    return dict(passed=True,model_parent_indices=indices,model_ranges_mm=ranges,
        weld_side=decoded['parents'][indices[0]].get('weld_side'),source_uncertainty_mm=eps,
        original_model_endpoint_uncertainty_mm=original.MODEL_ENDPOINT_MM,
        confidence=min(decoded['parents'][i]['confidence'] for i in indices),
        only_certified_native_join_endpoint_uncertainty_resolved=True,
        certified_internal_join_model_gaps=certificates,compatible_reference_candidates=sorted(consistent))


def propose(decoded,packet,inventory,solids,cached_geometry=None):
    fn=types.FunctionType(original.propose.__code__,dict(original.__dict__,own_coverage=own_coverage))
    result=fn(decoded,packet,inventory,solids,cached_geometry);kept=[]
    for proposal in result['proposals']:
        ref=proposal['junction']['reference']['candidate'];compatible=True
        for part in proposal['parts']:
            c=part['coverage']
            if 'compatible_reference_candidates' in c and ref not in c['compatible_reference_candidates']:compatible=False
            for i in c['model_parent_indices']:
                p=decoded['parents'][i]
                if any(p.get(k) for k in [*semantics.PROCESS_KEYS,'_confirmed_process_cut_ends']):compatible=False
        if compatible:kept.append(proposal)
        else:result['rejected'].append(dict(reason='PRESERVE_EXPLICIT_PROCESS_CUT_OR_INCONSISTENT_REFERENCE'))
    result['proposals']=kept
    result['model_gap_refinement']='V469_EXISTING_ENDPOINT_UNCERTAINTY_AT_ACTUAL_FINITE_NATIVE_JOINS_ONLY'
    return result
