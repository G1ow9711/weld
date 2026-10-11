"""Evaluation-only array caching; original D3 code, samples and gates unchanged.

Each call uses an isolated function-global mapping, avoiding monkey patches.
The original exported prediction and target objects are not modified.
"""
from types import FunctionType
import numpy as np
from .contract_geometry_d3_409 import score_source
from .sjg_native_d3_adapter_1056 import native_targets

def score_native_parents_cached(decoded,target,direction_semantics=None):
 parents=[]
 for p in decoded['parents']:
  segments=[dict(s,points=np.asarray(s['points'],np.float64)) for s in p['segments']]
  parents.append(dict(p,segments=segments))
 cached=dict(decoded,parents=parents)
 environment=dict(score_source.__globals__);original_reference=environment['reference_segment']
 def reference(c):
  r=original_reference(c);return dict(r,points=np.asarray(r['points'],np.float64))
 environment['reference_segment']=reference
 scorer=FunctionType(score_source.__code__,environment,score_source.__name__,score_source.__defaults__,score_source.__closure__)
 result=scorer(cached,native_targets(target),direction_semantics)
 result.update(coordinate_adapter='legacy field alias only; original native XYZ unchanged',native_coordinate_transform_applied=False,full_contract_pass=False)
 return result
