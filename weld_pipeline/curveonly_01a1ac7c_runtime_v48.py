"""R22 finite composite candidates from original material/XYZ inputs."""
import time,types
from .curveonly_01a1ac7c_runtime_v43 import *
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_runtime_v19 as source
from . import curveonly_01a1ac7c_registration_v21 as registration
from . import curveonly_01a1ac7c_projected_v43 as projected
from . import curveonly_01a1ac7c_frame_v30 as frame
from . import curveonly_01a1ac7c_composite_v48 as composite
from . import curveonly_01a1ac7c_contact_relations_v47 as relations
from . import curveonly_01a1ac7c_runtime_v43 as previous


def input_curves(record):
    start=time.perf_counter();captured={};proxy=types.ModuleType('curveonly_material_capture');proxy.__dict__.update(cad.__dict__)
    def load_material(path):
        solids=cad.load_material(path);captured['solids']=solids;return solids
    proxy.load_material=load_material
    fn=types.FunctionType(source.input_curves.__code__,dict(source.__dict__,original=proxy,registration=registration,projected=projected))
    x,p=fn(record);x,p=frame.build(x,p);tick=time.perf_counter()
    context=relations.Context(captured['solids'],p['pose']['transform_STEP_to_XYZ']) if p.get('pose',{}).get('passed') and 'solids' in captured else None
    x,p=composite.append(x,p,context)
    p['timing']=dict(p['timing'],normal_frame=p['normal_frame_canonicalization']['CPU_seconds'],
        carrier_union=p['carrier_union']['CPU_seconds'],finite_composite=time.perf_counter()-tick,
        input_to_features=time.perf_counter()-start)
    return x,p


def decode(features,packet,model,threshold=.5):
    result=previous.decode(features,packet,model,threshold)
    for parent in result['parents']:
        c=packet['candidates'][parent['source_candidate']]
        if 'composite_carrier_members' not in c:continue
        parent['source_material_path']=c['source_material_path'];parent['source_connections']=c['source_connections']
        parent['nominal_clearance_intervals']=c['nominal_clearance_intervals']
        parent['source_kind']=c['source_kind'];parent['source_joint_and_process_identity_accepted']=False
    return result
