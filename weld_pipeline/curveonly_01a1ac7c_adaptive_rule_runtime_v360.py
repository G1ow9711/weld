"""Actual R35 neural model with finite-section-adaptive edge/wall observation."""
import types
from . import curveonly_01a1ac7c_rule_runtime_v345 as backbone
from . import curveonly_01a1ac7c_adaptive_sections_v358 as rules
from . import curveonly_01a1ac7c_recovery_resolution_v353 as resolution


def decode(features,packet,model,solids=None,inventory=None):
    fn=types.FunctionType(backbone.decode.__code__,dict(backbone.__dict__,rules=rules))
    current=fn(features,packet,model,solids,inventory)
    result=resolution.resolve(current,packet,lambda:backbone.decode(features,packet,model,solids,inventory))
    result['adaptive_local_section_runtime']='V360_WITH_V358_FINITE_SECTIONS'
    return result


def extract(input_record,model):
    fn=types.FunctionType(backbone.extract.__code__,dict(backbone.__dict__,decode=decode))
    return fn(input_record,model)
