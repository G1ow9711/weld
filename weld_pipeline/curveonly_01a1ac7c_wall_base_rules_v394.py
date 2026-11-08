"""Recover model-anchored compound curved walls on a finite base.

Native material transitions remain explicit. Whole support, original source
uncertainty and measured transverse passage constrain the recovered intervals.
"""
import copy
import time
import numpy as np
from . import curveonly_01a1ac7c_wall_base_domains_v392 as observation
from . import curveonly_01a1ac7c_complete_parent_guard_v367 as guard

step=observation.step
finite,prior,provenance,aliases=step.finite,step.prior,step.provenance,step.aliases
baseline=step.baseline


def overlap(a,b):
    return sum(max(0.,min(y,v)-max(x,u)) for root,x,y in a for other,u,v in b if root==other)


def inspect(decoded,features,packet,inventory,solids,cached_observations=None):
    if not solids or not packet.get('pose',{}).get('passed'):
        return [],dict(not_applicable='NO_QUALIFIED_FINITE_STEP',input_retained=True),None
    context=step.Context(solids,packet,inventory)
    selected=[dict(candidate=p['source_candidate'],interval=p['interval']) for p in decoded['parents']]
    oldspans=[provenance.primitive_intervals(p['candidate'],p['interval']['start_mm'],p['interval']['end_mm'],packet,context.lengths) for p in selected]
    chosen=[];rejected=[];seen=[];used=[]
    candidates=[i for i,c in enumerate(packet['candidates']) if c.get('source_material_path') and c.get('STEP') and c.get('pose')]
    for ci in sorted(candidates,key=lambda i:-context.lengths[i]):
        c=packet['candidates'][ci];length=context.lengths[ci];points=np.asarray(c['points'])
        if np.linalg.norm(points[0]-points[-1])<=step.NUMERIC:continue
        incoming=provenance.primitive_intervals(ci,0.,length,packet,context.lengths)
        indices=[i for i,spans in enumerate(oldspans) if overlap(incoming,spans)>provenance.MIN_OVERLAP_MM]
        if not indices:continue
        if any(overlap(incoming,spans)>provenance.MIN_OVERLAP_MM for spans in used):continue
        if any(decoded['parents'][i]['weld_type']!='horizontal' for i in indices):continue
        if len({decoded['parents'][i].get('weld_side') for i in indices})!=1:
            rejected.append(dict(candidate=ci,reason='AMBIGUOUS_MODEL_WELD_SIDE'));continue
        conflicts=guard.conflicts(ci,length,selected,packet,context.lengths)
        if conflicts:
            rejected.append(dict(candidate=ci,reason='PRESERVE_MODEL_PARENT_OUTSIDE_THIS_NATIVE_PROOF',conflicts=conflicts));continue
        cert=copy.deepcopy(cached_observations[ci]) if cached_observations is not None and ci in cached_observations else observation.observe(ci,context,verify=True)
        seen.append(cert)
        if not cert.get('verified_dense'):
            rejected.append(cert);continue
        sec=context.view(cert['base'],1.);unresolved=[]
        for event in cert['transverse_review']:
            if event['measured_two_board_passage_mm'] is not None:continue
            q=(event['start_mm']+event['end_mm'])/2.
            p,_=finite.point_tangent(points,finite.cad.arclength(points),q)
            native=(p-sec.geometry.t)@sec.geometry.R
            distance,_=finite.events.native.finite_distance(solids[event['material']],native)
            event['original_body_minimum_distance_mm']=float(distance)
            state=observation.station(ci,q,context)['evidence'];free=np.asarray(state['free'])
            box=sec.bounds[event['material']]
            corners=np.array([[x,y,z] for x in (box[0],box[3]) for y in (box[1],box[4]) for z in (box[2],box[5])])
            forward=float(np.max((corners@sec.geometry.R.T+sec.geometry.t-p)@free))
            event['original_body_maximum_forward_weldside_extent_mm']=forward
            if forward < -state['numerical_source_uncertainty_mm']:
                event['no_free_side_passage_reason']='ENTIRE_FINITE_BODY_BEHIND_WELDING_SIDE'
            elif distance>finite.PASSAGE_MM:
                event['no_free_side_passage_reason']='ORIGINAL_BODY_DISTANCE_LOWER_BOUND_EXCEEDS_PROCESS_PASSAGE'
            else:unresolved.append(event)
        if unresolved:
            rejected.append(dict(candidate=ci,reason='UNRESOLVED_NEAR_TRANSVERSE_PASSAGE',events=unresolved));continue
        model_ranges=[]
        for i in indices:
            projected=aliases.coincident_parameters(step.native_parent(decoded['parents'][i],context),points)
            if projected is not None:model_ranges.append([float(projected[0].min()),float(projected[0].max())])
        model_ranges=finite.union(model_ranges);blocks=copy.deepcopy(cert['transverse_blocks'])
        for event in cert['transverse_review']:
            if event['measured_two_board_passage_mm'] is not None and any(
                    abs(a[1]-event['start_mm'])<=3. and abs(b[0]-event['end_mm'])<=3.
                    for a,b in zip(model_ranges[:-1],model_ranges[1:])):
                if not any(abs(v['start_mm']-event['start_mm'])<=step.NUMERIC and abs(v['end_mm']-event['end_mm'])<=step.NUMERIC for v in blocks):
                    blocks.append(dict(event,decision='PRESERVED_MODEL_GAP_AT_VERIFIED_TRANSVERSE_PAIR'))
        intervals=[]
        for a,b in finite.complement(length,blocks):
            native=provenance.primitive_intervals(ci,a,b,packet,context.lengths)
            anchors=[i for i in indices if overlap(native,oldspans[i])>provenance.MIN_OVERLAP_MM]
            if anchors:intervals.append(dict(start_mm=a,end_mm=b,model_parent_indices=anchors))
        if not intervals:
            rejected.append(dict(candidate=ci,reason='NO_MODEL_INTENT_IN_ALLOWED_COMPONENT'));continue
        cert.update(replaced_parent_indices=indices,anchored_weld_intervals=intervals,
            model_native_ranges_mm=model_ranges,transverse_blocks_with_model_preservation=blocks)
        chosen.append(cert);used.append(incoming)
    return chosen,dict(branch='V394_COMPOUND_CURVED_WALL_ON_FINITE_BASE',certificates=chosen,rejected=rejected,
        observations=seen,cached_geometry_observations=cached_observations is not None,
        no_new_fit_or_VALcalibration=True,no_name_coordinate_radius_count_or_dimension_lookup=True),context


def correct(decoded,features,packet,inventory,solids,cached_observations=None):
    start=time.perf_counter();chosen,audit,context=inspect(decoded,features,packet,inventory,solids,cached_observations)
    result=copy.deepcopy(decoded)
    if chosen:
        rows=prior.candidate_rows(features,packet);remove=set();replacements=[]
        for cert in chosen:
            ci=cert['candidate'];row=rows[ci];remove.update(cert['replaced_parent_indices']);intervals=[]
            for v in cert['anchored_weld_intervals']:
                intervals.append(dict(a=v['start_mm'],b=v['end_mm'],confidence=max(decoded['parents'][i]['confidence'] for i in v['model_parent_indices'])))
            replacements+=prior.codec.proposals(packet,[row],[prior.previous.intervals_to_codes(row,intervals)])
        selected,selection=provenance.select(replacements,packet);assert len(selected)==len(replacements)
        old=baseline.backbone.publication._fallback;baseline.backbone.publication._fallback=baseline.backbone.corner.fallback
        try:exported=baseline.backbone.exporter.decode(features,packet,baseline.backbone.Proposals(selected,{}))
        finally:baseline.backbone.publication._fallback=old
        parents=[copy.deepcopy(p) for i,p in enumerate(decoded['parents']) if i not in remove]+exported['parents']
        for i,p in enumerate(parents):p['parent_id']=f'curve_{i:05d}'
        result.update(parents=parents,geometry_units=len(parents),published_segments=sum(len(p['segments']) for p in parents))
        audit.update(replaced_upstream_parent_indices=sorted(remove),new_parent_count=len(exported['parents']),
            new_source_selection=selection,upstream_parents_not_targeted_retained_exact=True)
    result['wall_base_chain_rule']=audit;result['wall_base_chain_rule_seconds']=time.perf_counter()-start
    return result
