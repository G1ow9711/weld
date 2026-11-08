"""Obtain actual finite-cap witnesses regardless of generic ray-query success."""
import types
from . import curveonly_01a1ac7c_endcap_model_coverage_v469 as coverage

original=coverage.original
kernel,old,step,finite=coverage.kernel,coverage.old,coverage.step,coverage.finite
NUMERIC=coverage.NUMERIC
references=coverage.references


def cap_counterpart(ci,q,inner,interval,context):
    return coverage.finite_cap_mapping(ci,q,dict(candidate=inner,interval=interval),context)


caps_proxy=types.SimpleNamespace(**dict(original.caps.__dict__,counterpart=cap_counterpart))
end_mappings=types.FunctionType(original.end_mappings.__code__,dict(original.__dict__,caps=caps_proxy))
root_proxy=types.SimpleNamespace(**dict(original.__dict__,end_mappings=end_mappings))
_propose=types.FunctionType(coverage.propose.__code__,dict(coverage.__dict__,original=root_proxy))


def propose(decoded,packet,inventory,solids,cached_geometry=None):
    result=_propose(decoded,packet,inventory,solids,cached_geometry)
    result['finite_cap_query_policy']='V476_ALWAYS_INDEPENDENT_ACTUAL_END_FACE_WITNESS'
    return result
