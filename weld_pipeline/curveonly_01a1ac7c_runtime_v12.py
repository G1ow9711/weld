"""R6: independent raw planar-contact trace proposals; STEP path unchanged."""
import types
from .curveonly_01a1ac7c_runtime_v6 import *
from . import curveonly_01a1ac7c as original
from . import curveonly_01a1ac7c_registration_v2 as registration
from . import curveonly_01a1ac7c_sampler_v5 as sampler
from . import curveonly_01a1ac7c_projected_v11 as projected


def input_curves(record):
    sampler.reset_audit(); traces = []
    def point_curves(xyz):
        curves, audit = projected.proposals(xyz); traces.append(audit)
        return curves
    function = types.FunctionType(original.input_curves.__code__, dict(original.__dict__,
        choose_pose=registration.choose_pose, curve_points=sampler.curve_points, point_curves=point_curves))
    features, packet = function(record)
    packet['native_sampling'] = sampler.audit()
    packet['point_trace_proposals'] = traces
    return features, packet
