"""R4 composition: bounded native sampling; all34 model features unchanged."""
import types
from .curveonly_01a1ac7c_runtime_v2 import *
from . import curveonly_01a1ac7c as original
from . import curveonly_01a1ac7c_registration_v2 as registration
from . import curveonly_01a1ac7c_sampler_v5 as sampler

def input_curves(record):
    sampler.reset_audit()
    function=types.FunctionType(original.input_curves.__code__,dict(original.__dict__,choose_pose=registration.choose_pose,curve_points=sampler.curve_points))
    features,packet=function(record);packet['native_sampling']=sampler.audit()
    return features,packet
