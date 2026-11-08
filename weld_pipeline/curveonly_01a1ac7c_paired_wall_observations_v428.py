"""Observe opposite wall/base boundaries using an existing recovered model unit.

An actual finite wall cross-section supplies correspondence. Each boundary
keeps its own process gaps and must have its own positive model anchors.
"""
import copy
import types
import numpy as np
from . import curveonly_01a1ac7c_native_join_rules_v418 as recovery

old=recovery.old
observation=old.observation
step,finite=old.step,old.finite
NUMERIC=step.NUMERIC


def leaves(ci,context,stack=()):
    if ci in stack:raise ValueError('Cyclic input curve path')
    candidate=context.packet['candidates'][ci];path=candidate.get('source_material_path')
    if not path:return [dict(root=ci,material=candidate.get('material',-1),start_mm=0.,
        end_mm=context.lengths[ci],reversed=False)]
    result=[]
    for item in path:
        child=item['curve'];L=context.lengths[child]
        assert abs(item['end_mm']-item['start_mm']-L)<=1e-7*max(1.,L)
        for leaf in leaves(child,context,stack+(ci,)):
            a,b=leaf['start_mm'],leaf['end_mm']
            if item['reversed']:a,b=L-b,L-a
            result.append(dict(leaf,start_mm=item['start_mm']+a,end_mm=item['start_mm']+b,
                reversed=bool(leaf['reversed'])^bool(item['reversed'])))
    return sorted(result,key=lambda x:x['start_mm'])


def wall_sequence(rows):
    result=[]
    for row in rows:
        if not result or result[-1]!=row['material']:result.append(row['material'])
    return result


def contiguous_subsequence(short,long):
    return bool(short) and any(short==long[i:i+len(short)] for i in range(len(long)-len(short)+1))


def closest(point,points):
    p=np.asarray(points,float);s=finite.cad.arclength(p);v=np.diff(p,axis=0)
    t=np.clip(((point-p[:-1])*v).sum(1)/np.maximum((v*v).sum(1),1e-300),0.,1.)
    distances=np.linalg.norm(point-p[:-1]-t[:,None]*v,axis=1);i=int(np.argmin(distances))
    return float(s[i]+t[i]*(s[i+1]-s[i])),float(distances[i])


def source_uncertainty(ci,context):
    source=context.packet['candidates'][ci]
    return NUMERIC+float(source.get('source_curve_discretization_mm',.025))+float((source.get('circle') or {}).get('max_fit_error_mm',0.))


def counterpart(outer,q,inner,peer_interval,context):
    # Query the primitive tangent at joins. A compound tangent can belong to
    # the next wall while its endpoint is assigned to the previous wall.
    root,local=observation.primitive(outer,q,context)
    if root is None:return dict(passed=False,reason='NO_OWNED_NATIVE_STATION')
    state=observation.station(root,local,context)
    if state['status']!='POSITIVE_FINITE_BASE_FILLET':return dict(passed=False,reason='OUTER_FINITE_JOINT_UNRESOLVED',state=state)
    evidence=state['evidence'];base,wall=evidence['base'],evidence['wall'];sec=context.view(base,1.)
    p,_=finite.point_tangent(np.asarray(context.packet['candidates'][root]['points']),
        finite.cad.arclength(np.asarray(context.packet['candidates'][root]['points'])),local)
    inward=-np.asarray(evidence['free']);axis=sec.geometry.base;eps=evidence['numerical_source_uncertainty_mm']
    try:occupied=observation.section(sec,wall,p+axis*evidence['adaptive_probe_height_mm'],inward)
    except observation.UnresolvedSection:return dict(passed=False,reason='CROSS_WALL_INTERSECTOR_UNRESOLVED')
    spans=[(a,b) for a,b in occupied if abs(a)<=eps and b>eps]
    if len(spans)!=1:return dict(passed=False,reason='NO_UNIQUE_WALL_CROSS_SECTION',occupied_mm=occupied)
    a,b=spans[0];exit_point=p+inward*b;matches=[]
    for leaf in leaves(inner,context):
        if leaf['material']!=wall:continue
        inner_root=leaf['root'];native=context.packet['candidates'][inner_root]['points']
        station,error=closest(exit_point,native);budget=eps+source_uncertainty(inner_root,context)
        if error>budget:continue
        mapped=leaf['start_mm']+(context.lengths[inner_root]-station if leaf['reversed'] else station)
        if mapped<peer_interval['start_mm']-budget or mapped>peer_interval['end_mm']+budget:continue
        opposite=observation.station(inner_root,station,context)
        if opposite['status']!='POSITIVE_FINITE_BASE_FILLET':continue
        actual=opposite['evidence']
        if actual['wall']!=wall or actual['base']!=base:continue
        dot=float(np.asarray(actual['free'])@np.asarray(evidence['free']))
        if dot>=0.:continue
        matches.append(dict(inner_primitive=inner_root,inner_station_mm=station,peer_native_s_mm=float(mapped),
            cross_wall_exit_to_peer_error_mm=error,source_uncertainty_budget_mm=budget,opposite_free_dot=dot))
    if len(matches)!=1:return dict(passed=False,reason='NO_UNIQUE_OPPOSITE_NATIVE_BOUNDARY',matches=matches,
        outer_primitive=root,wall=wall,base=base,exit_point_mm=exit_point.tolist(),cross_wall_depth_mm=float(b-a))
    return dict(passed=True,outer_candidate=outer,outer_native_s_mm=float(q),outer_primitive=root,
        wall=wall,base=base,cross_wall_depth_mm=float(b-a),cross_section_probe_height_mm=evidence['adaptive_probe_height_mm'],
        outer_point_mm=p.tolist(),**matches[0])


def reference_parents(decoded,context):
    audit=decoded.get('wall_base_chain_rule',{});n=int(audit.get('new_parent_count',0))
    if not 0<n<=len(decoded['parents']):return []
    certificates={c['candidate']:c for c in audit.get('certificates',[]) if c.get('verified_dense') and
        c.get('existing_model_native_join_intent',{}).get('anchors')}
    result=[]
    for i,parent in enumerate(decoded['parents'][-n:],len(decoded['parents'])-n):
        ci=parent['source_candidate'];r=parent['interval'];cert=certificates.get(ci)
        if cert is None or parent['weld_type']!='horizontal' or parent.get('is_closed'):continue
        if not any(abs(r['start_mm']-v['start_mm'])<=NUMERIC and abs(r['end_mm']-v['end_mm'])<=NUMERIC
                for v in cert['anchored_weld_intervals']):continue
        result.append(dict(parent_index=i,candidate=ci,interval=r,base=cert['base'],
            wall_sequence=wall_sequence(leaves(ci,context))))
    return result


def geometry_choices(decoded,features,packet,inventory,solids,allowed,cached_observations):
    def observe(ci,context,verify=True):
        if ci not in allowed:return dict(candidate=ci,verified_dense=False,reason='NO_PAIRED_RECOVERED_MODEL_REFERENCE')
        return observation.observe(ci,context,verify=verify)
    proxy=types.SimpleNamespace(**dict(observation.__dict__,observe=observe))
    inspect=types.FunctionType(old.inspect.__code__,dict(old.__dict__,observation=proxy))
    cache=None if cached_observations is None else {ci:v for ci,v in cached_observations.items() if ci in allowed}
    return inspect(decoded,features,packet,inventory,solids,cache)


def assembly_end(mapping,reference,context):
    if not mapping.get('passed'):return False
    q,budget=mapping['peer_native_s_mm'],mapping['source_uncertainty_budget_mm'];r=reference['interval'];L=context.lengths[reference['candidate']]
    return bool((abs(q)<=budget and abs(r['start_mm'])<=NUMERIC) or
        (abs(q-L)<=budget and abs(r['end_mm']-L)<=NUMERIC))


def observe(decoded,features,packet,inventory,solids,cached_observations=None,verify=False):
    if not solids or not packet.get('pose',{}).get('passed'):
        return dict(candidates=[],reason='NO_QUALIFIED_FINITE_STEP',not_final_model_output=True)
    if not any(c.get('verified_dense') and c.get('existing_model_native_join_intent',{}).get('anchors')
            for c in decoded.get('wall_base_chain_rule',{}).get('certificates',[])):
        return dict(candidates=[],reason='NO_VERIFIED_RECOVERED_MODEL_REFERENCE',not_final_model_output=True)
    context=step.Context(solids,packet,inventory);references=reference_parents(decoded,context)
    if not references:return dict(candidates=[],reason='NO_VERIFIED_RECOVERED_MODEL_REFERENCE',not_final_model_output=True)
    possible={}
    for ci,c in enumerate(packet['candidates']):
        if not(c.get('STEP') and c.get('pose') and c.get('source_material_path')):continue
        sequence=wall_sequence(leaves(ci,context))
        peers=[r for r in references if ci!=r['candidate'] and (contiguous_subsequence(sequence,r['wall_sequence']) or
            contiguous_subsequence(sequence,list(reversed(r['wall_sequence']))))]
        if peers:possible[ci]=peers
    choices,geometry_audit,_=geometry_choices(decoded,features,packet,inventory,solids,set(possible),cached_observations)
    results=[]
    for cert in choices:
        ci=cert['candidate'];points=np.asarray(packet['candidates'][ci]['points']);native=finite.cad.arclength(points)
        for reference in possible[ci]:
            if cert['base']!=reference['base']:continue
            components=[]
            for span in cert['anchored_weld_intervals']:
                a,b=span['start_mm'],span['end_mm'];left=counterpart(ci,a,reference['candidate'],reference['interval'],context)
                right=counterpart(ci,b,reference['candidate'],reference['interval'],context)
                left_gap=any(abs(a-v['end_mm'])<=NUMERIC for v in cert['transverse_blocks_with_model_preservation'])
                right_gap=any(abs(b-v['start_mm'])<=NUMERIC for v in cert['transverse_blocks_with_model_preservation'])
                boundary_left=left_gap or (a<=NUMERIC and assembly_end(left,reference,context))
                boundary_right=right_gap or (b>=context.lengths[ci]-NUMERIC and assembly_end(right,reference,context))
                samples=np.unique(np.r_[a,b,[max(a,v['start_mm'])+(min(b,v['end_mm'])-max(a,v['start_mm']))/2.
                    for v in cert['cells'] if min(b,v['end_mm'])>max(a,v['start_mm'])+NUMERIC]])
                if verify:samples=np.unique(np.r_[samples,native[(native>a)&(native<b)],np.linspace(a,b,int(np.ceil(b-a))+2)])
                failures=[];sample_rows=[]
                for q in samples:
                    match=counterpart(ci,float(q),reference['candidate'],reference['interval'],context)
                    if not match['passed']:failures.append(dict(q_mm=float(q),evidence=match))
                    sample_rows.append(match)
                components.append(dict(start_mm=a,end_mm=b,passed=bool(boundary_left and boundary_right and not failures),
                    boundary_left=dict(passed=bool(boundary_left),own_true_gap=left_gap,mapping=left),
                    boundary_right=dict(passed=bool(boundary_right),own_true_gap=right_gap,mapping=right),
                    model_parent_indices=span['model_parent_indices'],checked_stations=len(samples),failures=failures,
                    correspondence=sample_rows,verified_dense=bool(verify and not failures)))
            results.append(dict(candidate=ci,reference=reference,geometry_certificate=cert,components=components))
    return dict(candidates=results,reference_parents=references,geometry_rejections=geometry_audit.get('rejected',[]),
        not_final_model_output=True,source_geometry_and_model_evidence_only=True,no_new_fit_or_VALcalibration=True)
