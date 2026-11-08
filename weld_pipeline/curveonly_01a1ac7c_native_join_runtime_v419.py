"""Frozen modality-aware curve model followed by guarded native chain recovery."""
import types
from . import curveonly_01a1ac7c_modality_runtime_v412 as baseline
from . import curveonly_01a1ac7c_native_join_rules_v418 as rules


def decode(features,packet,model,solids=None,inventory=None):
    if solids and inventory is None:inventory=baseline.baseline.root.geometry.build_geometry(packet,solids)
    current=baseline.decode(features,packet,model,solids,inventory)
    return rules.correct(current,features,packet,inventory,solids)


def extract(input_record,model):
    root=baseline.baseline.root
    return types.FunctionType(root.extract.__code__,dict(root.__dict__,decode=decode))(input_record,model)
