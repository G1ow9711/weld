"""Original finite sources plus local material observations; frozen export."""
import time,types
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_runtime_v48 as composite_source
from . import curveonly_01a1ac7c_runtime_v51 as contour_source
from . import curveonly_01a1ac7c_runtime_v56 as previous
from . import curveonly_01a1ac7c_local_material_v94 as material


def input_curves(record):
    start=time.perf_counter();captured={};proxy=types.ModuleType('curveonly_local_material_capture');proxy.__dict__.update(cad.__dict__)
    def load(path):
        solids=cad.load_material(path);captured['solids']=solids;return solids
    proxy.load_material=load
    raw=types.FunctionType(composite_source.input_curves.__code__,dict(composite_source.__dict__,cad=proxy))
    source=types.FunctionType(contour_source.input_curves.__code__,dict(contour_source.__dict__,previous=types.SimpleNamespace(input_curves=raw)))
    features,packet=source(record);values,receipt=material.build(packet,captured.get('solids'))
    packet=dict(packet,local_finite_material_v94=dict(values=values.tolist(),receipt=receipt))
    packet['timing']=dict(packet['timing'],local_finite_material_seconds=receipt['seconds'],input_to_features=time.perf_counter()-start)
    return features,packet


def decode(features,packet,model,threshold=.5):
    return previous.decode(features,packet,model,threshold)


arclength=cad.arclength
