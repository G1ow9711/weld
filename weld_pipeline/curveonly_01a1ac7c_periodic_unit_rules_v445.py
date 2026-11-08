"""Preserve current units whose native intervals wrap a proven periodic source."""
import types
import numpy as np
from . import curveonly_01a1ac7c_learned_unit_rules_v440 as original

cad=original.cad


class NativeUnits(original.NativeUnits):
    def atoms(self,p):
        ci=original.candidate(p);q=p.get('interval')
        if not q or not 0<=ci<len(self.lengths):return None
        a,b=float(q['start_mm']),float(q['end_mm']);length=self.lengths[ci]
        if not np.isfinite([a,b]).all() or length<=0. or a<0. or a>=length+1e-8 or b<=a:return None
        if b>length+1e-8:
            points=np.asarray(self.packet['candidates'][ci]['points'])
            if np.linalg.norm(points[0]-points[-1])>original.PERIODIC_NUMERIC_MM or b-a>length+original.PERIODIC_NUMERIC_MM:return None
        return original.provenance.primitive_intervals(ci,a,b,self.packet,self.lengths)

    def overlap(self,a,b):
        aa,bb=self.atoms(a),self.atoms(b)
        # Unknown native projection cannot certify that an old parent is absent.
        if aa is None or bb is None:return [dict(unresolved_native_interval_preserve_current=True)]
        return [dict(primitive=ci,start_mm=max(x,u),end_mm=min(y,v))
            for ci,x,y in aa for cj,u,v in bb if ci==cj and min(y,v)>max(x,u)]


inspect=types.FunctionType(original.inspect.__code__,dict(original.__dict__,NativeUnits=NativeUnits))
_correct=types.FunctionType(original.correct.__code__,dict(original.__dict__,inspect=inspect))


def correct(decoded,features,packet,raw,expert,expert_proposals=None):
    out=_correct(decoded,features,packet,raw,expert,expert_proposals)
    out['learned_unit_source_rule'].update(periodic_native_overlap_guard='V445_ORIGINAL_PROVEN_ONE_TURN_SOURCE_PROVENANCE',
        unknown_interval_is_not_treated_as_no_overlap=True)
    return out
