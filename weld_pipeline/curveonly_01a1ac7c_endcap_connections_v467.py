"""Propose a connected nominal carrier from a learned opposite-side unit.

Actual convex finite end caps must meet at the SAME interior native boundary of
one qualified complete model unit. Each side retains its own positive source
vote, model coverage and true transverse gaps. No spatial gap cutoff is used.
This module emits evidence/proposals, not accepted weld geometry.
"""
import copy,types
import numpy as np
from . import curveonly_01a1ac7c_paired_wall_native_caps_v432 as caps
from . import curveonly_01a1ac7c_periodic_unit_rules_v445 as topology

kernel=caps.previous.previous
old,step,finite=kernel.old,kernel.step,kernel.finite
NUMERIC=step.NUMERIC
MODEL_ENDPOINT_MM=3.  # Existing model endpoint uncertainty, never material-gap tolerance.


def references(decoded,packet):
    graph=topology.NativeUnits(packet);rows=[]
    for cert in decoded.get('wall_base_chain_rule',{}).get('certificates',[]):
        ci=cert.get('candidate')
        if not cert.get('verified_dense') or not cert.get('existing_model_native_join_intent',{}).get('anchors'):continue
        for i,p in enumerate(decoded['parents']):
            r=p['interval']
            if p['source_candidate']!=ci or p['weld_type']!='horizontal' or not graph.open(p):continue
            if abs(r['start_mm'])>NUMERIC or abs(r['end_mm']-graph.lengths[ci])>NUMERIC:continue
            if not any(abs(r['start_mm']-s['start_mm'])<=NUMERIC and abs(r['end_mm']-s['end_mm'])<=NUMERIC for s in cert['anchored_weld_intervals']):continue
            rows.append(dict(parent_index=i,candidate=ci,interval=copy.deepcopy(r),base=cert['base'],
                confidence=p['confidence'],wall_sequence=kernel.wall_sequence(kernel.leaves(ci,graph))))
    return rows


def end_mappings(reference,packet,context,graph):
    rows=[]
    for ci,c in enumerate(packet['candidates']):
        if graph.parents[ci] or ci==reference['candidate'] or not(c.get('STEP') and c.get('pose')):continue
        points=np.asarray(c['points'])
        if np.linalg.norm(points[0]-points[-1])<=NUMERIC:continue
        for end,q in [(0,0.),(1,graph.lengths[ci])]:
            root,local=kernel.observation.primitive(ci,q,context)
            if root is None:continue
            wall=packet['candidates'][root].get('material',-1)
            if wall not in reference['wall_sequence']:continue
            mapped=caps.counterpart(ci,q,reference['candidate'],reference['interval'],context)
            if not mapped.get('passed') or mapped.get('method')!='ACTUAL_CONVEX_FINITE_END_FACE':continue
            if mapped['base']!=reference['base']:continue
            inner=mapped['peer_native_s_mm']
            if not MODEL_ENDPOINT_MM+NUMERIC<inner<graph.lengths[reference['candidate']]-MODEL_ENDPOINT_MM-NUMERIC:continue
            rows.append(dict(candidate=ci,end=end,native_s_mm=q,root=root,wall=wall,mapping=mapped))
    return rows


def qualified_links(reference,ends,packet,context):
    links=[];path=kernel.leaves(reference['candidate'],context)
    for i,a in enumerate(ends):
        for b in ends[i+1:]:
            if a['candidate']==b['candidate'] or a['wall']==b['wall']:continue
            x,y=a['mapping'],b['mapping'];q=x['peer_native_s_mm']
            if abs(q-y['peer_native_s_mm'])>NUMERIC:continue
            if np.linalg.norm(np.asarray(x['inner_point_mm'])-y['inner_point_mm'])>NUMERIC:continue
            boundaries=[dict(left=left,right=right) for left,right in zip(path[:-1],path[1:])
                if abs(left['end_mm']-q)<=NUMERIC and abs(right['start_mm']-q)<=NUMERIC
                and {left['material'],right['material']}=={a['wall'],b['wall']}]
            if len(boundaries)!=1:continue
            # Both finite convex end-face certificates contain the same actual
            # inner point; same coordinates alone without these witnesses fail.
            gap=float(np.linalg.norm(np.asarray(x['outer_point_mm'])-y['outer_point_mm']))
            if gap<=NUMERIC:continue
            links.append(dict(left=a,right=b,reference=reference,
                reference_native_boundary=boundaries[0],same_reference_station_mm=q,
                actual_finite_caps_meet_at_inner_boundary=True,
                native_outer_clearance_mm=gap,no_gap_size_threshold=True,
                nominal_clearance_is_not_itself_a_process_gap_or_weld_permission=True))
    choices={}
    for i,link in enumerate(links):
        for end in [link['left'],link['right']]:choices.setdefault((end['candidate'],end['end']),[]).append(i)
    return [link for i,link in enumerate(links) if all(len(choices[end['candidate'],end['end']])==1 for end in [link['left'],link['right']])]


def own_coverage(decoded,packet,context,ci,start,end):
    incoming=old.provenance.primitive_intervals(ci,start,end,packet,context.lengths)
    indices=[];ranges=[];sides=set();eps=max(kernel.source_uncertainty(v['root'],context) for v in kernel.leaves(ci,context))
    for i,parent in enumerate(decoded['parents']):
        r=parent['interval'];spans=old.provenance.primitive_intervals(parent['source_candidate'],r['start_mm'],r['end_mm'],packet,context.lengths)
        if old.overlap(incoming,spans)<=NUMERIC:continue
        if parent['weld_type']!='horizontal' or not topology.NativeUnits(packet).open(parent):
            return dict(passed=False,reason='PRESERVE_OTHER_TYPE_CLOSED_OR_UNRESOLVED_PARENT',parent_index=i)
        mapped=old.aliases.coincident_parameters(step.native_parent(parent,context),packet['candidates'][ci]['points'])
        if mapped is None:return dict(passed=False,reason='UNMAPPED_OWN_PARENT',parent_index=i)
        s,error=mapped;lo,hi=float(s.min()),float(s.max())
        if lo<start-eps or hi>end+eps:return dict(passed=False,reason='PRESERVE_LONGER_EXISTING_PARENT',parent_index=i)
        indices.append(i);ranges.append([lo,hi]);sides.add(parent.get('weld_side'))
    if not indices or len(sides)!=1:return dict(passed=False,reason='NO_UNIQUE_OWN_MODEL_SIDE_OR_ANCHOR')
    merged=finite.union(ranges)
    if merged[0][0]>start+MODEL_ENDPOINT_MM or merged[-1][1]<end-MODEL_ENDPOINT_MM:
        return dict(passed=False,reason='OWN_MODEL_ENDPOINTS_DO_NOT_COVER_COMPONENT',model_ranges_mm=merged)
    # Interior empty model ranges are protected at source precision. Endpoint
    # uncertainty applies only to actual, separately certified component ends.
    if any(b[0]-a[1]>2*eps for a,b in zip(merged[:-1],merged[1:])):
        return dict(passed=False,reason='PRESERVE_UNEXPLAINED_INTERIOR_OWN_MODEL_GAP',model_ranges_mm=merged)
    return dict(passed=True,model_parent_indices=indices,model_ranges_mm=merged,weld_side=next(iter(sides)),
        original_model_endpoint_uncertainty_mm=MODEL_ENDPOINT_MM,source_uncertainty_mm=eps,
        confidence=min(decoded['parents'][i]['confidence'] for i in indices))


def boundary(ci,q,is_start,cert,reference,context):
    blocks=cert['transverse_blocks_with_model_preservation']
    true_gap=any(abs(q-v['end_mm' if is_start else 'start_mm'])<=NUMERIC for v in blocks)
    if true_gap:return dict(passed=True,own_true_gap=True)
    mapping=caps.counterpart(ci,q,reference['candidate'],reference['interval'],context)
    passed=kernel.assembly_end(mapping,reference,context)
    return dict(passed=bool(passed),own_true_gap=False,mapping=mapping)


def propose(decoded,packet,inventory,solids,cached_geometry=None):
    refs=references(decoded,packet);result=dict(references=refs,junctions=[],proposals=[],rejected=[],not_final_model_output=True)
    if not refs:return dict(result,reason='NO_QUALIFIED_OPPOSITE_COMPLETE_MODEL_UNIT')
    if not solids or not packet.get('pose',{}).get('passed'):return dict(result,reason='NO_QUALIFIED_FINITE_STEP')
    context=step.Context(solids,packet,inventory);graph=topology.NativeUnits(packet)
    for ref in refs:result['junctions']+=qualified_links(ref,end_mappings(ref,packet,context,graph),packet,context)
    votes={int(v['candidate']):v for v in decoded.get('paired_face_source_rule',{}).get('source_votes',[])}
    eligible=[link for link in result['junctions'] if all(votes.get(end['candidate'],{}).get('endorsed') for end in [link['left'],link['right']])]
    if not eligible:return dict(result,reason='NO_UNIQUE_MODEL_POSITIVE_FINITE_ENDCAP_JUNCTION')
    allowed={end['candidate'] for link in eligible for end in [link['left'],link['right']]}
    def observe(ci,ctx,verify=True):
        if ci not in allowed:return dict(candidate=ci,verified_dense=False,reason='NO_MODEL_QUALIFIED_ENDCAP_JUNCTION')
        return old.observation.observe(ci,ctx,verify=verify)
    step_proxy=types.SimpleNamespace(**dict(old.step.__dict__,Context=lambda *args:context))
    obs_proxy=types.SimpleNamespace(**dict(old.observation.__dict__,observe=observe))
    inspect=types.FunctionType(old.inspect.__code__,dict(old.__dict__,step=step_proxy,observation=obs_proxy))
    choices,geometry_audit,_=inspect(decoded,None,packet,inventory,solids,cached_geometry)
    certs={c['candidate']:c for c in choices};result['geometry_rejections']=geometry_audit['rejected']
    for link in eligible:
        parts=[];bad=None
        for end in [link['left'],link['right']]:
            ci=end['candidate'];cert=certs.get(ci)
            if cert is None:bad='NO_FULLY_QUALIFIED_OWN_NATIVE_JOINT';break
            spans=[span for span in cert['anchored_weld_intervals'] if abs(span['start_mm' if end['end']==0 else 'end_mm']-end['native_s_mm'])<=NUMERIC]
            if len(spans)!=1:bad='TRUE_PROCESS_GAP_OR_AMBIGUOUS_COMPONENT_AT_JUNCTION';break
            span=spans[0];coverage=own_coverage(decoded,packet,context,ci,span['start_mm'],span['end_mm'])
            outside=span['end_mm' if end['end']==0 else 'start_mm']
            bound=boundary(ci,outside,end['end']==1,cert,link['reference'],context)
            if not coverage['passed'] or not bound['passed']:
                bad=dict(reason='UNQUALIFIED_OWN_MODEL_COVERAGE_OR_EXTERNAL_BOUNDARY',coverage=coverage,boundary=bound);break
            parts.append(dict(candidate=ci,joining_end=end['end'],component=span,coverage=coverage,
                external_boundary=bound,true_process_blocks=cert['transverse_blocks_with_model_preservation'],
                checked_stations=cert.get('checked_stations'),verified_dense=cert['verified_dense'],source_vote=votes[ci]))
        if bad:result['rejected'].append(dict(junction=link,reason=bad));continue
        if parts[0]['coverage']['weld_side']!=parts[1]['coverage']['weld_side']:
            result['rejected'].append(dict(junction=link,reason='DIFFERENT_OWN_MODEL_SIDES'));continue
        indices=[i for p in parts for i in p['coverage']['model_parent_indices']]
        if len(set(indices))!=len(indices):
            result['rejected'].append(dict(junction=link,reason='SAME_PARENT_ALREADY_CROSSES_CONNECTION'));continue
        result['proposals'].append(dict(junction=link,parts=parts,
            own_model_parent_indices=sorted(indices),confidence=min(link['reference']['confidence'],*[p['coverage']['confidence'] for p in parts]),
            nominal_continuity_hypothesis_from_current_models=True,
            actual_observation_status='UNKNOWN_AT_SUBRESOLUTION_NOMINAL_CLEARANCE',
            source_clearance_retained=True,no_new_fit_or_VALcalibration=True))
    return result
