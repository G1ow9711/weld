"""Current-XYZ metric estimates on each own predicted child, no labels/I/O."""
import numpy as np
import torch
from weld_pipeline.contract_field_acceptance_297 import FIELDS
from weld_pipeline.legacy_incremental_attribute_runtime_404 import (
    feature_path_for_own_prediction, positions_on_own_candidate, decode_fields,
)
from weld_pipeline.legacy_incremental_correspondence_414 import canonical_child_fields
from weld_pipeline.legacy_incremental_metric_context_485 import metric_context


def estimate_parent_attributes(model, predictions, H_sequences, V_sequences, raw_context, force_H_event=False):
    records = []
    device = next(model.events.parameters()).device
    for parent in predictions:
        sequences = V_sequences if parent["type"] == "vertical" else H_sequences
        forced = parent.get("original_candidate_sequence_index") if parent["type"] == "vertical" or force_H_event else None
        if force_H_event:
            assert forced is not None and sequences[forced]["type"] == "horizontal"
            assert sequences[forced].get("curve_kind") != "arc"
        for child_index, segment in enumerate(parent["segments"]):
            data = feature_path_for_own_prediction(segment, sequences, forced)
            record = dict(parent_id=parent["id"], child_index=child_index, weld_side="unknown",
                formal_E_truth_certified=False, original_GT_or_STEP_reads=0, local_scope_not_expanded_to_whole_curve=True)
            if force_H_event:
                record["forced_original_candidate_sequence_index"] = forced
            if data is None:
                record.update(fields={name: dict(status="unknown", value=None, source=None,
                    reason="OWN_PREDICTION_HAS_NO_ORIGINAL_FEATURE_CHART") for name in FIELDS}, local_estimates=[])
            else:
                sequence = sequences[data["own_candidate_index"]]
                if forced is not None:
                    assert data["own_candidate_index"] == forced
                positions, errors = positions_on_own_candidate(sequence, data["world_positions_mm"])
                assert np.isfinite(positions).all() and np.isfinite(errors).all()
                local = torch.from_numpy(data["features"]).to(device)
                canonical = local.flip(0) if data["negative_chart_direction"] else local
                left, right = float(min(positions[0], positions[-1])), float(max(positions[0], positions[-1]))
                if segment["kind"] == "arc" and segment.get("is_closed"):
                    # Context reflects this own declared full curve, retaining
                    # its existing wrapping convention. No geometry is moved.
                    right = left + float(segment["length_mm"])
                whole_context = torch.from_numpy(metric_context(sequence, left, right, raw_context)[None]).to(device)
                point_context = torch.from_numpy(np.stack([metric_context(sequence, at, at, raw_context) for at in positions])).to(device)
                whole = model.path_attributes(canonical.reshape(1, -1), whole_context)
                pointwise = model.attributes(local, point_context)
                fields = canonical_child_fields(decode_fields(whole, 0, scope="WHOLE_PATH",
                    whole_known_fields=model.config.get("whole_path_known_supervision_fields")), data["negative_chart_direction"])
                record.update(fields=fields, own_candidate_index=data["own_candidate_index"],
                    raw_cloud_extent_is_feature_not_board_height_measurement=True,
                    local_estimates=[dict(predicted_child_fraction=float(i / 16), world_xyz_mm=data["world_positions_mm"][i].tolist(),
                        fields=decode_fields(pointwise, i), scope="POINTWISE_ESTIMATE_ONLY") for i in range(17)])
            records.append(record)
    return records


@torch.no_grad()
def estimate_predicted_attributes(model, predictions, H_sequences, V_sequences, raw_context):
    return estimate_parent_attributes(model, predictions, H_sequences, V_sequences, raw_context)


@torch.no_grad()
def estimate_event_H_attributes(model, predictions, current_mixed_sequences, raw_context):
    return estimate_parent_attributes(model, predictions, current_mixed_sequences, [], raw_context, force_H_event=True)
