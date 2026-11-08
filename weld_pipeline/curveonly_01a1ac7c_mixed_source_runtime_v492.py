"""General finite curved-to-linear source completion before frozen inference."""
import copy
from . import curveonly_01a1ac7c_independent_endcap_runtime_v479 as baseline
from . import curveonly_01a1ac7c_mixed_joint_sources_v485 as sources
from . import curveonly_01a1ac7c_mixed_source_observations_v489 as observations
from . import curveonly_01a1ac7c_event_interval_model_v304 as geometry


def forward(features,packet,model,solids,inventory,xyz,generated=None):
    if generated is None:generated=sources.build(packet,inventory,solids)
    if generated['candidates']:
        features,packet,receipt=observations.append(features,packet,generated,xyz,solids)
        inventory=geometry.build_geometry(packet,solids)
    else:receipt=dict(new_candidates=0,original_prefix_exact=True)
    result=baseline.decode(features,packet,model,solids,inventory)
    # Straight primitives carry material provenance for curved parents. The
    # curve-only model never publishes an isolated certified straight source.
    linear=set(receipt.get('certified_linear_provenance_only_candidates',[]))
    removed=[i for i,p in enumerate(result['parents']) if p['source_candidate'] in linear]
    if removed:
        result=copy.deepcopy(result)
        result['parents']=[p for i,p in enumerate(result['parents']) if i not in removed]
    result['mixed_finite_source_rule']=dict(receipt=receipt,
        removed_isolated_linear_parent_indices=removed,actual_model_and_material_events=True,
        sources_are_proposals_not_should_weld=True,all_nominal_clearances_retained=True,
        no_new_fit_or_part_lookup=True)
    return result,packet,features,inventory


def extract(record,model):
    features,packet=baseline.root.backbone.input_curves(record)
    step=record.get('STEP');solids=observations.cad.load_material(step['path']) if step and packet.get('pose',{}).get('passed') else None
    inventory=geometry.build_geometry(packet,solids) if solids else geometry.native_only(packet)
    xyz=observations.cad.read_xyz(record['raw']['path'])
    result,_,_,_=forward(features,packet,model,solids,inventory,xyz)
    return result
