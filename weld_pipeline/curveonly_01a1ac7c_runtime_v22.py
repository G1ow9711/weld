"""R15 ordered-set model with certified bounded hull pose rescue."""
import types
from .curveonly_01a1ac7c_runtime_v19 import *
from . import curveonly_01a1ac7c_runtime_v19 as previous
from . import curveonly_01a1ac7c_registration_v21 as registration


def input_curves(record):
    f=types.FunctionType(previous.input_curves.__code__,dict(previous.__dict__,registration=registration))
    return f(record)
