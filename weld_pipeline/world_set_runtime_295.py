"""Pure-PCD free decoding with unordered material estimates, official E unknown."""
import numpy as np
from .world_set_runtime_285 import decode_world_predictions
from .world_set_model_295 import VERSION
from .contract_curve_geometry import CircularArc

def decode_local_predictions(output,model=None,source=None):
    if output["version"]!=VERSION: raise ValueError("Registered local model output required")
    adapter=dict(output); adapter["version"]="PURE_RAW_WORLD_PARENT_QUERY_CHILD_SEQUENCE_285_v1"
    decoded=decode_world_predictions(adapter)
    decoded["version"]="PREDICTED_LOCAL_ANALYTIC_UNORDERED_AUX_RUNTIME_295_v1"
    decoded["runtime_attributes_trained"]=False
    decoded["unordered_auxiliary_branch_available"]=model is not None and source is not None
    if model is None or source is None: return decoded
    embeddings=[]; positions=[]; segments=[]
    for parent in decoded["parents"]:
        for child in parent["segments"]:
            if child["kind"]=="line":
                a,b=np.asarray(child["endpoints_mm"],np.float64)
                pos=a+np.linspace(0.,1.,5)[:,None]*(b-a)
            else:
                arc=CircularArc.from_parameters(child["arc_parameters"])
                pos=arc.point_at(np.linspace(0.,arc.length_mm,5))
            positions.append(pos); embeddings.append(output["child_embeddings"][parent["query_index"],child["segment_index"]]); segments.append(child)
    if segments:
        import torch
        auxiliary=model.predict_materials(source,torch.stack(embeddings),np.asarray(positions))
        thickness=auxiliary["thickness_mm"].detach().numpy(); normals=auxiliary["CAD_outward_unit_normals_world"].detach().numpy()
        for i,child in enumerate(segments):
            child["unordered_material_estimate"]=dict(source="PCD_LOCAL_LEARNED_UNREVIEWED_CAD_AUXILIARY",material_names=["material_A","material_B"],
                fractions=[0.,.25,.5,.75,1.],thickness_mm=thickness[i].tolist(),CAD_outward_unit_normals_world=normals[i].tolist(),
                evidence_status="MODEL_CANDIDATE_NOT_CUSTOMER_APPROVED",formal_E_status="UNKNOWN_SIDE_CONVENTION",
                runtime_GT_or_STEP_reads=0,paired_material_order_consistent_within_child=True)
    decoded["formal_E_status"]="UNKNOWN_SIDE_CONVENTION"
    return decoded
