"""R18: same curve-geometry learner with stable input plane-normal signs."""
import time
from .curveonly_01a1ac7c_runtime_v28 import *
from . import curveonly_01a1ac7c_runtime_v25 as source
from . import curveonly_01a1ac7c_frame_v30 as frame


def input_curves(record):
    start=time.perf_counter();x,packet=source.input_curves(record);x,packet=frame.build(x,packet)
    packet['timing']=dict(packet['timing'],normal_frame=packet['normal_frame_canonicalization']['CPU_seconds'],
        carrier_union=packet['carrier_union']['CPU_seconds'],input_to_features=time.perf_counter()-start)
    return x,packet
