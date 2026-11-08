"""Same learned-unit runtime with conservative native periodic overlap."""
import types
from . import curveonly_01a1ac7c_learned_unit_runtime_v441 as previous
from . import curveonly_01a1ac7c_periodic_unit_rules_v445 as rules

actual_raw=previous.actual_raw
baseline=previous.baseline
decode=types.FunctionType(previous.decode.__code__,dict(previous.__dict__,rules=rules),argdefs=previous.decode.__defaults__)
extract=types.FunctionType(previous.extract.__code__,dict(previous.__dict__,decode=decode))
