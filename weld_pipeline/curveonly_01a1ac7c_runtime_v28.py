"""R17 geometry component: input-only carrier union and learned intervals."""
import time
from .curveonly_01a1ac7c_runtime_v25 import *
from . import curveonly_01a1ac7c_runtime_v25 as previous
from . import curveonly_01a1ac7c_carrier_union_v28 as union


def input_curves(record):
    start=time.perf_counter();x,packet=previous.input_curves(record);x,packet=union.build(x,packet)
    packet['timing']=dict(packet['timing'],carrier_union=packet['carrier_union']['CPU_seconds'],input_to_features=time.perf_counter()-start)
    return x,packet
