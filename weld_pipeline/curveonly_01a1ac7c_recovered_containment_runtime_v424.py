"""Current complete curve model with certified native interval deduplication."""
import types
from . import curveonly_01a1ac7c_native_join_runtime_v419 as baseline
from . import curveonly_01a1ac7c_recovered_containment_v423 as rules


def decode(features,packet,model,solids=None,inventory=None):
    return rules.correct(baseline.decode(features,packet,model,solids,inventory))


def extract(input_record,model):
    root=baseline.baseline.baseline.root
    return types.FunctionType(root.extract.__code__,dict(root.__dict__,decode=decode))(input_record,model)
