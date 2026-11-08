"""Curve model with finite compound wall/base range and passage recovery."""
import types
from . import curveonly_01a1ac7c_partial_contact_runtime_v385 as baseline
from . import curveonly_01a1ac7c_wall_base_rules_v394 as rules

root=baseline.baseline.baseline.backbone


def decode(features,packet,model,solids=None,inventory=None):
    if solids and inventory is None:inventory=root.geometry.build_geometry(packet,solids)
    before=baseline.decode(features,packet,model,solids,inventory)
    return rules.correct(before,features,packet,inventory,solids)


def extract(input_record,model):
    return types.FunctionType(root.extract.__code__,dict(root.__dict__,decode=decode))(input_record,model)
