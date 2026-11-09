"""Original frozen complete curve pipeline plus the numerical ray-origin fix."""
import types
from . import curveonly_01a1ac7c_native_boundary_runtime_v528 as previous
from . import curveonly_01a1ac7c_contact_origin_v710 as rules


def forward(features,packet,model,solids,inventory,xyz,generated=None):
    before,packet,features,inventory=previous.forward(features,packet,model,solids,inventory,xyz,generated)
    return rules.correct(before,features,packet,solids,inventory,model.paired.source_expert),packet,features,inventory


def decode_existing(features,packet,model,solids,inventory,return_before=False):
    before=previous.decode_existing(features,packet,model,solids,inventory,False)
    result=rules.correct(before,features,packet,solids,inventory,model.paired.source_expert)
    return (result,before)if return_before else result


extract=types.FunctionType(previous.extract.__code__,dict(previous.extract.__globals__,forward=forward),argdefs=previous.extract.__defaults__)
