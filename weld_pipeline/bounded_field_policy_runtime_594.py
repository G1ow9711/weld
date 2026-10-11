"""Current finite surfaces and own predicted charts only; no TRAIN imports."""
import numpy as np
import torch
from weld_pipeline.contract_field_acceptance_297 import FIELDS
from weld_pipeline.rotation_aware_support_trim_599 import build_finite_surface_index
from weld_pipeline.legacy_incremental_attribute_runtime_404 import (
    feature_path_for_own_prediction, positions_on_own_candidate, decode_fields,
)
from weld_pipeline.legacy_incremental_correspondence_414 import canonical_child_fields


def current_index(xyz, sequences):
    index = build_finite_surface_index(xyz)
    for sequence in sequences:
        center = (sequence["xyz"][0] + sequence["xyz"][-1]) / 2.
        normals = np.asarray(sequence["features"])[len(sequence["features"]) // 2, 60:66].reshape(2, 3)
        for normal in normals:
            if np.linalg.norm(normal) > 1e-8:
                index.register_lazy_plane(normal, center)
    return index


def explain_fields(fields,hypotheses,output,row=0,negative_chart=False,whole=False):
    swaps={6:7,7:6,8:9,9:8,10:11,11:10}
    for j,name in enumerate(FIELDS):
        fi=swaps.get(j,j) if negative_chart else j
        item=fields[name]
        if fi in (6,7,8,9):
            item['source']='CURRENT_XYZ_PHYSICAL_HYPOTHESIS_MODEL_ASSOCIATION'
            item['finite_physical_hypotheses']=len(hypotheses['fields'][fi])
            item['association_status']=output['association_reasons'][row][fi]
            item['plate_identity_confirmed']=False
            item['estimated_sampling_budget_not_accuracy_certificate']=True
            if item['status']=='unknown':
                item['reason']=item['association_status']+'; '+str(item.get('reason') or 'LEARNED_UNKNOWN_STATE')
            for record in output['association_rows_TRAIN_differentiable']:
                if record['row']==row and record['field']==fi and record['real_candidate_count']:
                    item['physical_scalar_value_group_probability']=record['scalar_group_probability']
        elif whole and fi in (0,1,2,3,13,15):
            item['source']='MISSING_WHOLE_CURVE_REFERENCE'
            item['reason']='MISSING_WHOLE_TRUTH'
        else:
            item['source']='CURRENT_XYZ_MODEL_ESTIMATE_SEPARATE_FIELD_HEAD'
            item['field_policy']='BOOLEAN_MODEL_ESTIMATE' if fi in (12,14) else 'MODEL_ESTIMATE_REQUIRES_REFERENCE_REVIEW'
        item['whole_curve_or_signed_side_acceptance_certified']=False
    return fields


@torch.no_grad()
def estimate_parent_attributes(model, predictions, H_sequences, V_sequences, current_original_xyz_mm, force_H_event=False):
    records = []
    all_sequences = V_sequences if V_sequences else H_sequences
    index = current_index(current_original_xyz_mm, all_sequences)
    device = next(model.events.parameters()).device
    for parent in predictions:
        sequences = V_sequences if parent["type"] == "vertical" else H_sequences
        forced = parent.get("original_candidate_sequence_index") if parent["type"] == "vertical" or force_H_event else None
        for ci, segment in enumerate(parent["segments"]):
            data = feature_path_for_own_prediction(segment, sequences, forced)
            record = dict(parent_id=parent["id"], child_index=ci, weld_side="unknown",
                formal_E_truth_certified=False, original_GT_or_STEP_reads=0,
                local_scope_not_expanded_to_whole_curve=True, finite_surface_current_XYZ_only=True)
            if force_H_event: record["forced_original_candidate_sequence_index"] = forced
            if data is None:
                record.update(fields={name: dict(status="unknown", value=None, source=None,
                    reason="OWN_PREDICTION_HAS_NO_ORIGINAL_FEATURE_CHART") for name in FIELDS}, local_estimates=[])
            else:
                seq = sequences[data["own_candidate_index"]]
                if forced is not None: assert forced == data["own_candidate_index"]
                positions, error = positions_on_own_candidate(seq, data["world_positions_mm"])
                assert np.isfinite(positions).all() and np.isfinite(error).all()
                local = torch.from_numpy(data["features"]).to(device)
                canonical = local.flip(0) if data["negative_chart_direction"] else local
                a, z = float(min(positions[0], positions[-1])), float(max(positions[0], positions[-1]))
                if segment["kind"] == "arc" and segment.get("is_closed"):
                    z = a + float(segment["length_mm"])
                whole_hyp = index.ordered_hypotheses(seq, a, z)
                local_hyp = [index.ordered_hypotheses(seq, float(t), float(t)) for t in positions]
                whole = model.path_attributes(canonical.reshape(1, -1), [whole_hyp])
                pointwise = model.attributes(local, local_hyp)
                fields = canonical_child_fields(decode_fields(whole, 0, scope="WHOLE_PATH",
                    whole_known_fields=model.config.get("whole_path_known_supervision_fields")), data["negative_chart_direction"])
                record.update(fields=explain_fields(fields,whole_hyp,whole,0,data["negative_chart_direction"],True), own_candidate_index=data["own_candidate_index"],
                    finite_plate_pairs=len(whole_hyp["pairs"]), plate1_plate2_and_signed_normal_status="UNRESOLVED",
                    local_estimates=[dict(predicted_child_fraction=float(i / 16),
                        world_xyz_mm=data["world_positions_mm"][i].tolist(),
                        fields=explain_fields(decode_fields(pointwise,i),local_hyp[i],pointwise,i),
                        scope="POINTWISE_ESTIMATE_ONLY") for i in range(17)])
            records.append(record)
    return records


def estimate_predicted_attributes(model, predictions, H_sequences, V_sequences, current_original_xyz_mm):
    return estimate_parent_attributes(model, predictions, H_sequences, V_sequences, current_original_xyz_mm)


def estimate_event_H_attributes(model, predictions, current_mixed_sequences, current_original_xyz_mm):
    return estimate_parent_attributes(model, predictions, current_mixed_sequences, [], current_original_xyz_mm, True)
