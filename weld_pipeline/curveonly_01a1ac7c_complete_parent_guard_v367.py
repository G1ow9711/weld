"""Local geometry recovery cannot truncate an existing longer source parent.

The invariant is native interval coverage, independent of names, counts, radius,
pose or manufacturing dimensions. A partial certificate does not authorize
removing the remainder of a continuous model proposal. Keep the model proposal
and decline that overlapping local recovery instead of fragmenting the parent.
"""
import types
import numpy as np
from . import curveonly_01a1ac7c_adaptive_sections_v365 as adaptive

prior,provenance,finite,aliases=adaptive.prior,adaptive.provenance,adaptive.finite,adaptive.aliases
NUMERIC=adaptive.NUMERIC


def conflicts(ci,length,selected,packet,lengths):
    incoming=provenance.primitive_intervals(ci,0.,length,packet,lengths)
    result=[]
    for index,p in enumerate(selected):
        interval=p['interval']
        old=provenance.primitive_intervals(p['candidate'],interval['start_mm'],interval['end_mm'],packet,lengths)
        overlap=sum(max(0.,min(b,y)-max(a,x)) for root,a,b in old for other,x,y in incoming if root==other)
        if overlap<=provenance.MIN_OVERLAP_MM:
            continue
        outside=[]
        for root,a,b in old:
            covered=finite.union([(max(a,x),min(b,y)) for other,x,y in incoming if root==other and min(b,y)>max(a,x)])
            if not any(a>=x-NUMERIC and b<=y+NUMERIC for x,y in covered):
                outside.append([root,a,b])
        if outside:
            result.append(dict(prior_proposal=index,prior_candidate=p['candidate'],prior_interval=interval,
                shared_native_length_mm=overlap,prior_footprints_outside_this_recovery=outside))
    return result


def certify(ci,row,packet,geometry,inventory,selected):
    lengths=[float(finite.cad.arclength(np.asarray(c['points']))[-1]) for c in packet['candidates']]
    keep=conflicts(ci,row['length_mm'],selected,packet,lengths)
    if keep:
        return None,dict(candidate=ci,reason='PRESERVE_LONGER_CONTINUOUS_MODEL_PARENT',
            conflicts=keep,no_should_weld_negative_inferred=True)
    return adaptive.certify(ci,row,packet,geometry,inventory,selected)


def apply(raw,features,packet,inventory,solids):
    fn=types.FunctionType(adaptive.old.apply.__code__,dict(adaptive.old.__dict__,certify=certify,
        seed_mask=lambda x:np.ones(len(x),dtype=bool)))
    proposals,audit=fn(raw,features,packet,inventory,solids)
    audit['branch']='V367_ADAPTIVE_SECTIONS_WITH_COMPLETE_PARENT_PRESERVATION'
    audit['fixed_1_8_and_minus1_plus3_seed_probes_removed']=True
    audit['partial_recovery_cannot_delete_longer_continuous_model_parent']=True
    for p in proposals:
        if p.get('structural_rule')=='FINITE_CURVED_BASE_EDGE_THROUGH_UPRIGHT_FILLET_V348':
            p['structural_rule']='ADAPTIVE_FINITE_SECTION_RECOVERY_WITH_PARENT_GUARD_V367'
    return proposals,audit
