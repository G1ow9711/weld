"""All adopted model stages, then model-qualified finite-endcap recovery."""
import types
from . import curveonly_01a1ac7c_representation_runtime_v462 as baseline
from . import curveonly_01a1ac7c_rule_runtime_v345 as root
from . import curveonly_01a1ac7c_endcap_recovery_v471 as rules


def decode(features,packet,model,solids=None,inventory=None):
    if solids and inventory is None:inventory=root.geometry.build_geometry(packet,solids)
    before=baseline.decode(features,packet,model,solids,inventory)
    return rules.correct(before,packet,inventory,solids)


def extract(input_record,model):
    return types.FunctionType(root.extract.__code__,dict(root.__dict__,decode=decode))(input_record,model)
