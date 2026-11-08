"""Frozen curve model with finite opposite boundary and own coverage recovery."""
import types
from . import curveonly_01a1ac7c_native_join_runtime_v419 as baseline
from . import curveonly_01a1ac7c_paired_wall_rules_v434 as rules


def decode(features,packet,model,solids=None,inventory=None):
    if solids and inventory is None:inventory=baseline.baseline.baseline.root.geometry.build_geometry(packet,solids)
    before=baseline.decode(features,packet,model,solids,inventory)
    return rules.correct(before,features,packet,inventory,solids)


def extract(input_record,model):
    root=baseline.baseline.baseline.root
    return types.FunctionType(root.extract.__code__,dict(root.__dict__,decode=decode))(input_record,model)
