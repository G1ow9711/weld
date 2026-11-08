"""Connected-source export with independently queried finite endcaps."""
import types
from . import curveonly_01a1ac7c_endcap_recovery_v471 as previous
from . import curveonly_01a1ac7c_independent_endcaps_v476 as observation

old,finite,NUMERIC=previous.old,previous.finite,previous.NUMERIC
derived_source,source_view,published_source_view=previous.derived_source,previous.source_view,previous.published_source_view
_correct=types.FunctionType(previous.correct.__code__,dict(previous.__dict__,observation=observation))


def correct(decoded,packet,inventory,solids,cached_observation=None):
    result=_correct(decoded,packet,inventory,solids,cached_observation)
    result['endcap_connection_recovery_rule']['independent_finite_endcaps_version']=476
    return result
