"""Same actual R35 model/publisher with the additional finite edge-wall rule."""
import types
from . import curveonly_01a1ac7c_rule_runtime_v345 as previous
from . import curveonly_01a1ac7c_edge_wall_rules_v348 as rules


def decode(features, packet, model, solids=None, inventory=None):
    fn = types.FunctionType(previous.decode.__code__, dict(previous.__dict__, rules=rules))
    return fn(features, packet, model, solids, inventory)


def extract(input_record, model):
    fn = types.FunctionType(previous.extract.__code__, dict(previous.__dict__, decode=decode))
    return fn(input_record, model)
