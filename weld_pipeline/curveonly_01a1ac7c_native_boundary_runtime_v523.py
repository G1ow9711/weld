"""Current complete-curve model with retained original native boundaries."""
from . import curveonly_01a1ac7c_final_face_runtime_v514 as baseline
from . import curveonly_01a1ac7c_native_boundary_export_v521 as rules


def forward(features,packet,model,solids,inventory,xyz,generated=None):
    before,packet,features,inventory=baseline.forward(features,packet,model,solids,inventory,xyz,generated)
    evidence=rules.bank(packet,before,solids=solids)
    return rules.correct(before,packet,evidence),packet,features,inventory


def decode_existing(features,packet,model,solids,inventory,return_before=False):
    before=baseline.decode_existing(features,packet,model,solids,inventory)
    evidence=rules.bank(packet,before,solids=solids)
    result=rules.correct(before,packet,evidence)
    return (result,before,evidence) if return_before else result


def extract(record,model):
    source=baseline.baseline.sources
    features,packet=source.baseline.root.backbone.input_curves(record)
    step=record.get('STEP')
    solids=source.observations.cad.load_material(step['path']) if step and packet.get('pose',{}).get('passed') else None
    inventory=source.geometry.build_geometry(packet,solids) if solids else source.geometry.native_only(packet)
    xyz=source.observations.cad.read_xyz(record['raw']['path'])
    result,_,_,_=forward(features,packet,model,solids,inventory,xyz)
    return result
