"""Controlled composition: change only input registration, retain 34D model."""
import types
from .curveonly_01a1ac7c import *
from . import curveonly_01a1ac7c as original
from . import curveonly_01a1ac7c_registration_v2 as registration

def input_curves(record):
    function=types.FunctionType(original.input_curves.__code__,dict(original.__dict__,choose_pose=registration.choose_pose))
    return function(record)
