"""Mixed finite sources, original models, finite identity and explicit geometry."""
import copy
from . import curveonly_01a1ac7c_mixed_source_runtime_v492 as sources
from . import curveonly_01a1ac7c_finite_curve_alias_v502 as rules
from . import curveonly_01a1ac7c_point_geometry_schema_v509 as schema


def correct(before,packet,inventory,solids):
    return schema.correct(rules.correct(before,packet,inventory,solids))


def forward(features,packet,model,solids,inventory,xyz,generated=None):
    before,packet,features,inventory=sources.forward(features,packet,model,solids,inventory,xyz,generated)
    return correct(before,packet,inventory,solids),packet,features,inventory


def decode_existing(features,packet,model,solids,inventory,return_before=False):
    """Same V492 model/scope stage on already verified actual observations."""
    before=sources.baseline.decode(features,packet,model,solids,inventory)
    receipt=packet.get('mixed_source_observations_v487',dict(new_candidates=0,original_prefix_exact=True))
    linear=set(receipt.get('certified_linear_provenance_only_candidates',[]))
    removed=[i for i,p in enumerate(before['parents']) if p['source_candidate'] in linear]
    if removed:
        before=copy.deepcopy(before);before['parents']=[p for i,p in enumerate(before['parents']) if i not in removed]
    before['mixed_finite_source_rule']=dict(receipt=receipt,removed_isolated_linear_parent_indices=removed,
        actual_model_and_material_events=True,sources_are_proposals_not_should_weld=True,
        all_nominal_clearances_retained=True,no_new_fit_or_part_lookup=True)
    result=correct(before,packet,inventory,solids)
    return (result,before) if return_before else result


def extract(record,model):
    features,packet=sources.baseline.root.backbone.input_curves(record)
    step=record.get('STEP');solids=sources.observations.cad.load_material(step['path']) if step and packet.get('pose',{}).get('passed') else None
    inventory=sources.geometry.build_geometry(packet,solids) if solids else sources.geometry.native_only(packet)
    xyz=sources.observations.cad.read_xyz(record['raw']['path'])
    result,_,_,_=forward(features,packet,model,solids,inventory,xyz)
    return result
