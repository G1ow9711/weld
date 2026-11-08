"""Frozen curve model with input-derived partial finite-contact range recovery."""
import types
from . import curveonly_01a1ac7c_step_joint_runtime_v375 as baseline
from . import curveonly_01a1ac7c_partial_contact_rules_v384 as rules


def decode(features,packet,model,solids=None,inventory=None):
    if solids and inventory is None:
        inventory=baseline.baseline.backbone.geometry.build_geometry(packet,solids)
    before=baseline.decode(features,packet,model,solids,inventory)
    return rules.correct(before,features,packet,inventory,solids)


def extract(input_record,model):
    root=baseline.baseline.backbone
    return types.FunctionType(root.extract.__code__,dict(root.__dict__,decode=decode))(input_record,model)
