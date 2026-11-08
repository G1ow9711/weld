"""Small local curve model: 238 finite observations plus direction windows."""
import sys
import types
import time
import numpy as np
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_local_intervals_v95 as legacy
from . import curveonly_01a1ac7c_range_field_v150 as native
from . import curveonly_01a1ac7c_material_direction_v160 as direction
from . import curveonly_01a1ac7c_local_runtime_v112 as previous
from OCP.BRepCheck import BRepCheck_Analyzer

INPUT_COLUMNS=34
OBSERVATION_COLUMNS=len(legacy.RAW_COLUMNS)+12+len(direction.NAMES)
WINDOW_MM=legacy.WINDOW_MM
FEATURES=OBSERVATION_COLUMNS*len(WINDOW_MM)+len(WINDOW_MM)
assert OBSERVATION_COLUMNS==56 and FEATURES==399


def observations(features,packet):
    features=np.asarray(features)
    assert features.ndim==2 and features.shape[1]==INPUT_COLUMNS
    old=legacy.observations(features,packet)
    values=np.asarray(packet['local_material_direction_v160']['values'],np.float32)
    assert values.shape==(len(features),len(direction.NAMES))
    result=np.column_stack([old,values]).astype(np.float32)
    assert result.shape==(len(features),OBSERVATION_COLUMNS) and np.isfinite(result).all()
    return result


def encode(features,packet):
    return types.FunctionType(legacy.encode.__code__,
        dict(legacy.__dict__,observations=observations,FEATURES=FEATURES))(features,packet)


class DirectionCurveSet(native.NativeFieldCurveSet):
    """Binary classifier and complement CTMC; no global count/template weights."""
    def predict(self,features,packet):
        return types.FunctionType(native.NativeFieldCurveSet.predict.__code__,
            dict(native.__dict__,section=sys.modules[__name__]))(self,features,packet)


class OldLocalCurveSet(native.NativeFieldCurveSet):
    def predict(self,features,packet):
        return types.FunctionType(native.NativeFieldCurveSet.predict.__code__,
            dict(native.__dict__,section=legacy))(self,features,packet)


def input_curves(record):
    start=time.perf_counter();captured={}
    proxy=types.ModuleType('curveonly_direction_material_capture');proxy.__dict__.update(cad.__dict__)
    def load(path):
        solids=cad.load_material(path);captured['solids']=solids;return solids
    proxy.load_material=load
    read=types.FunctionType(previous.input_curves.__code__,dict(previous.__dict__,cad=proxy))
    features,packet=read(record);solids=captured.get('solids')
    validity=[bool(BRepCheck_Analyzer(q,True).IsValid())for q in solids]if solids else []
    values,receipt=direction.build(packet,solids if all(validity)else None)
    if solids and not all(validity):receipt.update(reason='INVALID_FINITE_MATERIAL_BREP')
    if not solids:receipt.update(reason='NO_STEP_INPUT')
    packet=dict(packet,local_material_direction_v160=dict(values=values.tolist(),receipt=dict(receipt,real_solid_validity=validity)))
    packet['timing']=dict(packet['timing'],material_direction_seconds=receipt['seconds'],input_to_features=time.perf_counter()-start)
    return features,packet


def decode(features,packet,model,threshold=.5):
    assert isinstance(model,(DirectionCurveSet,OldLocalCurveSet))
    return previous.decode(features,packet,model,threshold)


arclength=cad.arclength
