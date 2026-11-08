"""R21: raw-XYZ radial support repair; all STEP numerics unchanged."""
import types,time
from .curveonly_01a1ac7c_runtime_v30 import *
from . import curveonly_01a1ac7c_runtime_v19 as source
from . import curveonly_01a1ac7c_registration_v21 as registration
from . import curveonly_01a1ac7c_projected_v43 as projected
from . import curveonly_01a1ac7c_frame_v30 as frame


def input_curves(record):
    start=time.perf_counter()
    fn=types.FunctionType(source.input_curves.__code__,dict(source.__dict__,registration=registration,projected=projected))
    x,packet=fn(record);x,packet=frame.build(x,packet)
    packet['timing']=dict(packet['timing'],normal_frame=packet['normal_frame_canonicalization']['CPU_seconds'],
        carrier_union=packet['carrier_union']['CPU_seconds'],input_to_features=time.perf_counter()-start)
    return x,packet
