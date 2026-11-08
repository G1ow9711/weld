"""Publish a model-qualified connected source without changing model inputs.

New source geometry is an explicit post-model provenance view. Original model
features/candidates are never changed or supplied with fabricated observations.
"""
import copy
import numpy as np
from . import curveonly_01a1ac7c_endcap_model_coverage_v469 as observation
from . import curveonly_01a1ac7c_rule_runtime_v345 as exporter

old,finite=observation.old,observation.finite
NUMERIC=observation.NUMERIC


def derived_source(proposal,packet):
    parts=proposal['parts'];rows=[];points=[];lengths=[];reverse=[];components=[]
    for index,part in enumerate(parts):
        ci=part['candidate'];source=packet['candidates'][ci]
        rev=part['joining_end']==(0 if index==0 else 1)
        p=np.asarray(source['points'],float);length=float(finite.cad.arclength(p)[-1])
        a,b=part['component']['start_mm'],part['component']['end_mm']
        components.append([length-b,length-a] if rev else [a,b])
        points.append(p[::-1] if rev else p);lengths.append(length);reverse.append(rev)
    gap=float(np.linalg.norm(points[0][-1]-points[1][0]));offset=lengths[0]+gap
    assert abs(components[0][1]-lengths[0])<=NUMERIC and abs(components[1][0])<=NUMERIC
    for index,part in enumerate(parts):
        start=0. if index==0 else offset
        rows.append(dict(curve=part['candidate'],material=packet['candidates'][part['candidate']].get('material',-1),
            start_mm=start,end_mm=start+lengths[index],reversed=reverse[index]))
    combined=np.vstack(points);total=float(finite.cad.arclength(combined)[-1])
    assert abs(total-(sum(lengths)+gap))<=1e-7*max(1.,total)
    candidate=dict(points=combined.tolist(),material=-2,STEP=True,pose=True,circle=finite.cad.circle_fit(combined),
        source_material_path=rows,source_curve_count=sum(packet['candidates'][p['candidate']].get('source_curve_count',1) for p in parts),
        source_connections=[copy.deepcopy(proposal['junction'])],
        nominal_clearance_intervals=[dict(start_mm=lengths[0],end_mm=offset,length_mm=gap,
            source='MEASURED_ORIGINAL_FINITE_ENDCAP_OUTER_CLEARANCE',
            weld_gap_or_process_cut_status='MODEL_CONTINUITY_HYPOTHESIS_OBSERVATION_UNKNOWN')],
        source_length_mm=sum(lengths),nominal_clearance_length_mm=gap,
        source_kind='POST_MODEL_FINITE_ENDCAP_CONNECTED_NOMINAL_CARRIER',
        joint_identity_accepted=False,process_continuity_accepted=False,
        source_geometry_only_no_model_feature_rows=True,
        original_child_clearances_and_process_blocks_retained=True)
    return candidate,dict(start_mm=components[0][0],end_mm=offset+components[1][1],index=0,predicted_count=1)


def source_view(packet,derived):
    """An output-source geometry view, deliberately lacking feature offsets."""
    return dict(candidates=[*packet['candidates'],*derived],base_axis=packet.get('base_axis'),pose=packet.get('pose'),
        view='POST_MODEL_PUBLISHED_SOURCE_PROVENANCE_NOT_MODEL_FEATURE_INPUT',
        original_candidate_count=len(packet['candidates']),no_model_feature_rows_constructed=True,
        original_model_geometry_and_feature_packet_unchanged=True)


def published_source_view(decoded,packet):
    data=decoded.get('endcap_derived_source_geometry')
    if data is None:return packet
    assert data['original_candidate_count']==len(packet['candidates'])
    return source_view(packet,data['candidates'])


def correct(decoded,packet,inventory,solids,cached_observation=None):
    if 'endcap_connection_recovery_rule' in decoded:return copy.deepcopy(decoded)
    observed=observation.propose(decoded,packet,inventory,solids) if cached_observation is None else copy.deepcopy(cached_observation)
    proposals=observed.get('proposals',[]);uses={}
    for i,proposal in enumerate(proposals):
        for parent in proposal['own_model_parent_indices']:uses.setdefault(parent,[]).append(i)
    selected=[p for i,p in enumerate(proposals) if all(len(uses[j])==1 for j in p['own_model_parent_indices'])]
    result=copy.deepcopy(decoded);derived=[];raw=[];certs=[];remove=set()
    for proposal in selected:
        candidate,interval=derived_source(proposal,packet);ci=len(packet['candidates'])+len(derived)
        points=finite.cad.slice_curve(candidate['points'],interval['start_mm'],interval['end_mm'])
        assert np.linalg.norm(points[0]-points[-1])>NUMERIC
        side=proposal['parts'][0]['coverage']['weld_side']
        raw.append(dict(candidate=ci,points=points.tolist(),weld_type='horizontal',weld_side=side,
            confidence=proposal['confidence'],interval=interval))
        derived.append(candidate);remove.update(proposal['own_model_parent_indices'])
        certs.append(dict(derived_candidate=ci,interval=interval,proposal=proposal,
            geometry_from_original_own_side_sources_only=True,no_opposite_or_truth_geometry_copied=True,
            original_own_true_process_blocks_retained=[p['true_process_blocks'] for p in proposal['parts']]))
    if raw:
        view=source_view(packet,derived)
        previous=exporter.publication._fallback;exporter.publication._fallback=exporter.corner.fallback
        try:published=exporter.exporter.decode(None,view,exporter.Proposals(raw,dict(post_model_geometry_only=True)))
        finally:exporter.publication._fallback=previous
        assert len(published['parents'])==len(raw)
        for parent in published['parents']:
            match=next(r for r in raw if r['candidate']==parent['source_candidate'])
            parent['weld_side']=match['weld_side']
            parent['nominal_connection_observation_status']='UNKNOWN_AT_SUBRESOLUTION_CLEARANCE'
        result['parents']=[p for i,p in enumerate(result['parents']) if i not in remove]+published['parents']
        for i,p in enumerate(result['parents']):p['parent_id']=f'curve_{i:05d}'
        result.update(geometry_units=len(result['parents']),published_segments=sum(len(p['segments']) for p in result['parents']),
            endcap_derived_source_geometry=dict(original_candidate_count=len(packet['candidates']),candidates=derived,
                created_after_original_model_forward=True,not_used_as_model_features=True))
    result['endcap_connection_recovery_rule']=dict(branch='V471_MODEL_QUALIFIED_FINITE_ENDCAP_SOURCE_CONNECTION',
        certificates=certs,replaced_upstream_parent_indices=sorted(remove),new_parent_count=len(raw),
        rejected_proposals=observed.get('rejected',[]),references=observed.get('references',[]),
        competing_parent_proposals_preserved=len(proposals)-len(selected),
        exact_bound_connection_cache_used=cached_observation is not None,
        no_gap_length_name_coordinate_radius_or_fixed_count_lookup=True,
        no_new_fit_or_VALcalibration=True,original_model_inputs_and_untargeted_parent_fields_preserved=True)
    return result
