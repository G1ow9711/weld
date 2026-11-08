"""Apply source-identical representation containment after all adopted rules."""
import types
from . import curveonly_01a1ac7c_coplanar_source_runtime_v457 as baseline
from . import curveonly_01a1ac7c_rule_runtime_v345 as root
from . import curveonly_01a1ac7c_representation_containment_v461 as rules


def decode(features,packet,model,solids=None,inventory=None):
    before=baseline.decode(features,packet,model,solids,inventory)
    return rules.correct(before,packet)


def extract(input_record,model):
    return types.FunctionType(root.extract.__code__,dict(root.__dict__,decode=decode))(input_record,model)
