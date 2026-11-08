"""Actual curve model followed by finite plate-step side disambiguation."""
import types
from . import curveonly_01a1ac7c_complete_parent_runtime_v368 as baseline
from . import curveonly_01a1ac7c_step_joint_rules_v373 as rules


def decode(features,packet,model,solids=None,inventory=None):
    if solids and inventory is None:
        inventory=baseline.backbone.geometry.build_geometry(packet,solids)
    before=baseline.decode(features,packet,model,solids,inventory)
    return rules.correct(before,features,packet,inventory,solids)


def extract(input_record,model):
    return types.FunctionType(baseline.backbone.extract.__code__,dict(baseline.backbone.__dict__,decode=decode))(input_record,model)
