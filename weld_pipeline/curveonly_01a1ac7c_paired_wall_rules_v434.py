"""Recover a complete opposite boundary with its own contiguous model coverage."""
import copy
import numpy as np
from . import curveonly_01a1ac7c_paired_wall_native_caps_v432 as observation

kernel=observation.previous.previous
old,step,finite=kernel.old,kernel.step,kernel.finite
NUMERIC=step.NUMERIC


def coverage(decoded,packet,candidate,start,end):
    lengths=[float(finite.cad.arclength(c['points'])[-1]) for c in packet['candidates']]
    context=type('NativeContext',(),dict(packet=packet,lengths=lengths))()
    incoming=old.provenance.primitive_intervals(candidate,start,end,packet,lengths)
    uncertainty=2.*max(kernel.source_uncertainty(v['root'],context) for v in kernel.leaves(candidate,context))
    indices=[];ranges=[];sides=set()
    for i,parent in enumerate(decoded['parents']):
        r=parent['interval'];ci=parent['source_candidate']
        spans=old.provenance.primitive_intervals(ci,r['start_mm'],r['end_mm'],packet,lengths)
        if old.overlap(incoming,spans)<=NUMERIC:continue
        if parent['weld_type']!='horizontal' or parent.get('is_closed'):
            return dict(passed=False,reason='PRESERVE_OTHER_TYPE_OR_CLOSED_MODEL_PARENT',parent_index=i)
        projected=old.aliases.coincident_parameters(step.native_parent(parent,context),packet['candidates'][candidate]['points'])
        if projected is None:return dict(passed=False,reason='PRESERVE_UNMAPPED_SHARED_MODEL_PARENT',parent_index=i)
        s,error=projected;lo,hi=float(s.min()),float(s.max())
        if lo<start-uncertainty or hi>end+uncertainty:
            return dict(passed=False,reason='PRESERVE_COMPLETE_PARENT_OUTSIDE_COMPONENT',parent_index=i,
                projected_interval_mm=[lo,hi],component_mm=[start,end])
        indices.append(i);ranges.append([lo,hi]);sides.add(parent.get('weld_side'))
    if not indices:return dict(passed=False,reason='NO_OWN_OUTER_MODEL_INTENT')
    if len(sides)!=1:return dict(passed=False,reason='AMBIGUOUS_OWN_MODEL_SIDE')
    merged=finite.union(ranges)
    if merged[0][0]>start+uncertainty or merged[-1][1]<end-uncertainty or any(
            b[0]-a[1]>uncertainty for a,b in zip(merged[:-1],merged[1:])):
        return dict(passed=False,reason='PRESERVE_UNEXPLAINED_OWN_MODEL_COVERAGE_GAP',
            model_ranges_mm=merged,component_mm=[start,end],source_uncertainty_budget_mm=uncertainty)
    if len(indices)==1:return dict(passed=False,reason='ALREADY_ONE_COMPLETE_MODEL_PARENT',parent_index=indices[0])
    return dict(passed=True,model_parent_indices=indices,model_ranges_mm=merged,
        source_uncertainty_budget_mm=uncertainty,weld_side=next(iter(sides)),
        confidence=min(decoded['parents'][i]['confidence'] for i in indices))


def select(decoded,packet,observed):
    chosen=[];rejected=[];used=set()
    for pair in observed.get('candidates',[]):
        for component in pair['components']:
            if not component['passed'] or not component['verified_dense']:continue
            ci=pair['candidate'];a,b=component['start_mm'],component['end_mm']
            model=coverage(decoded,packet,ci,a,b)
            if not model['passed']:
                rejected.append(dict(candidate=ci,start_mm=a,end_mm=b,evidence=model));continue
            indices=set(model['model_parent_indices'])
            if indices & used:
                rejected.append(dict(candidate=ci,start_mm=a,end_mm=b,reason='COMPETING_PAIRED_RECOVERY'));continue
            confidence=min(model['confidence'],decoded['parents'][pair['reference']['parent_index']]['confidence'])
            chosen.append(dict(candidate=ci,start_mm=a,end_mm=b,reference=pair['reference'],
                finite_geometry_component=component,own_model_coverage=model,confidence=confidence))
            used.update(indices)
    return chosen,dict(branch='V434_PAIRED_FINITE_WALL_MODEL_COVERAGE',certificates=chosen,rejected=rejected,
        no_unconditional_opposite_side_weld=True,no_unobserved_long_range_extension=True,
        no_new_fit_or_VALcalibration=True)


def correct(decoded,features,packet,inventory,solids,cached_observation=None):
    if cached_observation is None:
        observed=observation.observe(decoded,features,packet,inventory,solids,verify=True)
    else:observed=copy.deepcopy(cached_observation)
    chosen,audit=select(decoded,packet,observed);result=copy.deepcopy(decoded)
    if chosen:
        rows=old.prior.candidate_rows(features,packet);remove=set();proposals=[]
        for cert in chosen:
            ci=cert['candidate'];row=rows[ci];remove.update(cert['own_model_coverage']['model_parent_indices'])
            intervals=[dict(a=cert['start_mm'],b=cert['end_mm'],confidence=cert['confidence'])]
            proposals+=old.prior.codec.proposals(packet,[row],[old.prior.previous.intervals_to_codes(row,intervals)])
        selected,selection=old.provenance.select(proposals,packet);assert len(selected)==len(proposals)
        previous=old.baseline.backbone.publication._fallback
        old.baseline.backbone.publication._fallback=old.baseline.backbone.corner.fallback
        try:published=old.baseline.backbone.exporter.decode(features,packet,old.baseline.backbone.Proposals(selected,{}))
        finally:old.baseline.backbone.publication._fallback=previous
        for parent in published['parents']:
            matches=[c for c in chosen if c['candidate']==parent['source_candidate'] and
                abs(c['start_mm']-parent['interval']['start_mm'])<=NUMERIC and
                abs(c['end_mm']-parent['interval']['end_mm'])<=NUMERIC]
            assert len(matches)==1
            parent['weld_side']=matches[0]['own_model_coverage']['weld_side']
        assert len(published['parents'])==len(chosen)
        result['parents']=[p for i,p in enumerate(result['parents']) if i not in remove]+published['parents']
        for i,p in enumerate(result['parents']):p['parent_id']=f'curve_{i:05d}'
        result['geometry_units']=len(result['parents'])
        result['published_segments']=sum(len(p['segments']) for p in result['parents'])
        audit.update(replaced_upstream_parent_indices=sorted(remove),new_parent_count=len(published['parents']),
            source_selection=selection,untargeted_parent_fields_retained_exact_except_parent_id=True)
    audit.update(exact_bound_paired_geometry_cache_used=cached_observation is not None,
        geometry_observation_reason=observed.get('reason'),reference_parents=observed.get('reference_parents',[]))
    result['paired_wall_recovery_rule']=audit
    return result
