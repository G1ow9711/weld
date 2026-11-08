"""Native-boundary export with explicit original/generated provenance domains."""
import types
from . import curveonly_01a1ac7c_native_boundary_runtime_v523 as previous
from . import curveonly_01a1ac7c_native_boundary_provenance_v527 as rules

baseline=previous.baseline
forward=types.FunctionType(previous.forward.__code__,dict(previous.__dict__,rules=rules))
decode_existing=types.FunctionType(previous.decode_existing.__code__,dict(previous.__dict__,rules=rules))
extract=types.FunctionType(previous.extract.__code__,dict(previous.__dict__,rules=rules,forward=forward))
