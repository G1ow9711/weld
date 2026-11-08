"""Actual neural model with adaptive finite sections and whole-parent protection."""
import types
from . import curveonly_01a1ac7c_rule_runtime_v345 as backbone
from . import curveonly_01a1ac7c_complete_parent_guard_v367 as rules
from . import curveonly_01a1ac7c_recovery_resolution_v353 as resolution


def decode(features,packet,model,solids=None,inventory=None):
    current=types.FunctionType(backbone.decode.__code__,dict(backbone.__dict__,rules=rules))(
        features,packet,model,solids,inventory)
    result=resolution.resolve(current,packet,lambda:backbone.decode(features,packet,model,solids,inventory))
    result['adaptive_local_section_runtime']='V368_ADAPTIVE_WITH_COMPLETE_PARENT_GUARD'
    return result


def extract(input_record,model):
    return types.FunctionType(backbone.extract.__code__,dict(backbone.__dict__,decode=decode))(input_record,model)
