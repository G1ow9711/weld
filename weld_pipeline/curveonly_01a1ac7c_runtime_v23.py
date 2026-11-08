"""R15 stable-model control: old point source and decoder, new pose rescue."""
import types,time
from .curveonly_01a1ac7c_runtime_v6 import *
from . import curveonly_01a1ac7c as original
from . import curveonly_01a1ac7c_runtime_v19 as gate
from . import curveonly_01a1ac7c_registration_v21 as registration


class OriginalPointPath:
    @staticmethod
    def proposals(xyz):
        start=time.perf_counter();curves=original.point_curves(xyz)
        return curves,dict(method='UNCHANGED_STABLE_R5_POINT_SOURCE',seconds=time.perf_counter()-start)


def input_curves(record):
    f=types.FunctionType(gate.input_curves.__code__,dict(gate.__dict__,registration=registration,projected=OriginalPointPath))
    return f(record)
