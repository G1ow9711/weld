"""R27 geometry pipeline with finite material role observations; same decode."""
import time,types
from .curveonly_01a1ac7c_runtime_v56 import *
from . import curveonly_01a1ac7c_runtime_v56 as previous
from . import curveonly_01a1ac7c_runtime_v48 as composite_source
from . import curveonly_01a1ac7c_runtime_v51 as contour_source
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_material_roles_v81 as roles


def input_curves(record):
    start=time.perf_counter();captured={};proxy=types.ModuleType('curveonly_role_material_capture');proxy.__dict__.update(cad.__dict__)
    def load(path):
        solids=cad.load_material(path);captured['solids']=solids;return solids
    proxy.load_material=load
    raw=types.FunctionType(composite_source.input_curves.__code__,dict(composite_source.__dict__,cad=proxy))
    composite=types.SimpleNamespace(input_curves=raw)
    source=types.FunctionType(contour_source.input_curves.__code__,dict(contour_source.__dict__,previous=composite))
    features,packet=source(record)
    packet=dict(packet,finite_material_roles_v81=roles.build(packet,captured.get('solids')))
    packet['timing']=dict(packet['timing'],material_roles=packet['finite_material_roles_v81']['seconds'],input_to_features=time.perf_counter()-start)
    return features,packet


def decode(features,packet,model,threshold=.5):
    result=previous.decode(features,packet,model,threshold);context=packet['finite_material_roles_v81']
    result['finite_material_role_observation']=dict(available=context['available'],feature_columns=8,
        relative_height_extents_not_contract_attributes=True,nearest_material_identity_not_confirmed_joint=True)
    return result
