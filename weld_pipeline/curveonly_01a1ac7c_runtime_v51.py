"""R23 source-only complete finite contour candidates; R22 model interface."""
import time
from .curveonly_01a1ac7c_runtime_v48 import *
from . import curveonly_01a1ac7c_runtime_v48 as previous
from . import curveonly_01a1ac7c_contour_v51 as contour


def input_curves(record):
    start=time.perf_counter();x,p=previous.input_curves(record);tick=time.perf_counter()
    x,p=contour.append(x,p)
    p['timing']=dict(p['timing'],finite_contour=time.perf_counter()-tick,input_to_features=time.perf_counter()-start)
    return x,p


def decode(features,packet,model,threshold=.5):
    result=previous.decode(features,packet,model,threshold)
    for parent in result['parents']:
        c=packet['candidates'][parent['source_candidate']]
        if 'source_shape_certificate' in c:
            parent['source_shape_certificate']=c['source_shape_certificate']
    return result
