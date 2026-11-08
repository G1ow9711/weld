"""Publish model-anchored joints within verified partial finite contact domains.

The native carrier, material ends, local sections and transverse passage supply
the range. A model proposal supplies should-weld intent for each component.
An optional exact-input geometry cache is for reproducible precision controls;
the normal runtime recomputes the same unlabelled observations.
"""
import copy
import time
import numpy as np
from . import curveonly_01a1ac7c_partial_contact_v382 as observation
from . import curveonly_01a1ac7c_complete_parent_guard_v367 as guard

step=observation.step
finite,prior,provenance=step.finite,step.prior,step.provenance
baseline=step.baseline


def inspect(decoded,features,packet,inventory,solids,cached_observations=None):
    if not solids or not packet.get('pose',{}).get('passed'):
        return [],dict(not_applicable='NO_QUALIFIED_FINITE_STEP',input_retained=True),None
    context=observation.Context(solids,packet,inventory)
    sources=sorted({p['source_candidate'] for p in decoded['parents']})
    selected=[dict(candidate=p['source_candidate'],interval=p['interval']) for p in decoded['parents']]
    chosen=[];rejected=[];observations=[]
    for ci in sources:
        c=packet['candidates'][ci]
        if not(c.get('STEP') and c.get('pose') and c.get('circle')):continue
        if not step.resolution.curvature_observation(c)['numerically_resolved_curvature']:continue
        own=[i for i,p in enumerate(decoded['parents']) if p['source_candidate']==ci]
        if any(decoded['parents'][i]['weld_type']!='horizontal' for i in own):continue
        if len({decoded['parents'][i].get('weld_side') for i in own})!=1:
            rejected.append(dict(candidate=ci,reason='AMBIGUOUS_MODEL_WELD_SIDE'));continue
        if cached_observations is not None and ci in cached_observations:
            cert=copy.deepcopy(cached_observations[ci])
        else:cert=observation.observe(ci,decoded['parents'],context,verify=True)
        observations.append(cert)
        if not cert.get('applicable'):
            rejected.append(cert);continue
        assert cert['verified_dense']
        conflicts=guard.conflicts(ci,context.lengths[ci],selected,packet,context.lengths)
        if conflicts:
            rejected.append(dict(candidate=ci,reason='PRESERVE_PARENT_OUTSIDE_PROOF_DOMAIN',conflicts=conflicts));continue
        incoming=provenance.primitive_intervals(ci,0.,context.lengths[ci],packet,context.lengths)
        overlaps=[]
        for i,p in enumerate(selected):
            if i in own:continue
            old=provenance.primitive_intervals(p['candidate'],p['interval']['start_mm'],p['interval']['end_mm'],packet,context.lengths)
            shared=sum(max(0.,min(b,y)-max(a,x)) for root,a,b in old for other,x,y in incoming if root==other)
            if shared>provenance.MIN_OVERLAP_MM:overlaps.append(dict(parent_index=i,shared_native_length_mm=shared))
        if overlaps:
            rejected.append(dict(candidate=ci,reason='PRESERVE_UNTARGETED_MODEL_REPRESENTATION',overlaps=overlaps));continue
        cert['replaced_parent_indices']=own
        chosen.append(cert)
    return chosen,dict(branch='V384_MODEL_ANCHORED_PARTIAL_FINITE_CONTACT',certificates=chosen,
        rejected=rejected,observations=observations,cached_geometry_observations=cached_observations is not None,
        no_new_fit_or_VALcalibration=True,no_fixed_name_coordinate_radius_count_or_dimension=True),context


def correct(decoded,features,packet,inventory,solids,cached_observations=None):
    start=time.perf_counter()
    chosen,audit,context=inspect(decoded,features,packet,inventory,solids,cached_observations)
    result=copy.deepcopy(decoded)
    if chosen:
        rows=prior.candidate_rows(features,packet);replacements=[];remove=set()
        for cert in chosen:
            ci=cert['candidate'];row=rows[ci]
            remove.update(cert['replaced_parent_indices'])
            intervals=[]
            for span in cert['anchored_weld_intervals']:
                confidence=max(decoded['parents'][i]['confidence'] for i in span['model_parent_indices'])
                intervals.append(dict(a=span['start_mm'],b=span['end_mm'],confidence=confidence))
            replacements+=prior.codec.proposals(packet,[row],[prior.previous.intervals_to_codes(row,intervals)])
        selected,source_selection=provenance.select(replacements,packet)
        assert len(selected)==len(replacements),'Competing partial-domain certificates must remain unresolved'
        old=baseline.backbone.publication._fallback
        baseline.backbone.publication._fallback=baseline.backbone.corner.fallback
        try:exported=baseline.backbone.exporter.decode(features,packet,baseline.backbone.Proposals(selected,{}))
        finally:baseline.backbone.publication._fallback=old
        parents=[copy.deepcopy(p) for i,p in enumerate(decoded['parents']) if i not in remove]+exported['parents']
        for i,parent in enumerate(parents):parent['parent_id']=f'curve_{i:05d}'
        result['parents']=parents;result['geometry_units']=len(parents)
        result['published_segments']=sum(len(p['segments']) for p in parents)
        audit.update(replaced_upstream_parent_indices=sorted(remove),new_parent_count=len(exported['parents']),
            new_source_selection=source_selection,upstream_parents_not_targeted_retained_exact=True)
    result['partial_finite_contact_rule']=audit
    result['partial_finite_contact_rule_seconds']=time.perf_counter()-start
    return result
