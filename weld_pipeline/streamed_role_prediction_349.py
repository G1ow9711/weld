"""Same original-PCD finite prediction, scalar role gate and stage observations."""
import gc

import numpy as np
import torch

from .raw_point_groups_285 import RawPointSource
from .carrier_boundary_model_328 import CarrierBoundaryModel
from .carrier_observation_sequences_328 import SequenceLayout, OriginalXYZObservationProvider
from .carrier_boundary_runtime_329 import VERSION
from .fresh_analytic_carrier_provider_327 import fresh_analytic_carriers
from .preallocated_carrier_inference_340 import build_compact_inference_carriers
from .streamed_role_gate_347 import scan_complete_child_start_gate, materialize_complete_heads_after_gate, empty_gate_decoded_result
from .gate_checked_consuming_decoder_347 import decode_gate_checked_consuming_heads
from .current_process_runtime_327 import normalize_process_paths


def source_numeric_buffers(source):
    return {key: getattr(source, key) for key in ("xyz_mm", "group_ids", "counts", "mean_xyz_mm", "center_mm")}


def predict_from_original_source(model, source, progress=None, stage_profile=None,
                                 compare_original_whole_sigmoid=True):
    if type(source) is not RawPointSource or type(model) is not CarrierBoundaryModel:
        raise ValueError("Frozen original model and verified complete original XYZ required")
    if any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in model.modules()):
        raise ValueError("Role-only execution requires unchanged no-hook frozen modules")
    report = progress or (lambda **values: None)
    events = stage_profile or (lambda **values: None)
    def emit(stage, edge, **values):
        events(stage=stage, edge=edge, **values)
    def generation_progress(stage, **values):
        report(stage=stage, **values)
    emit("PURE_XYZ_CARRIER_GENERATION", "BEGIN")
    records, generation = fresh_analytic_carriers(source.xyz_mm, generation_progress)
    emit("PURE_XYZ_CARRIER_GENERATION", "END", original_raw_record_count=len(records))
    emit("COMPACT_EXACT_CARRIER_BUILD", "BEGIN")
    carriers = build_compact_inference_carriers(records)
    emit("COMPACT_EXACT_CARRIER_BUILD", "END", complete_original_carriers=len(carriers))
    del records
    gc.collect()
    emit("GENERATION_TEMPORARIES_RELEASED", "END", complete_original_carriers=len(carriers))
    emit("PURE_XYZ_OBSERVATION_PROVIDER", "BEGIN")
    provider = OriginalXYZObservationProvider(source)
    emit("PURE_XYZ_OBSERVATION_PROVIDER", "END", known_objects=dict(tree_xyz=provider.tree.data,
        group_tree=provider.group_tree.data, available_groups=provider.available_groups))
    model.eval()
    parents, invalid, queries, slots, calls = [], [], 0, 0, 0
    empty, nonempty, differentials, polyline_points = 0, 0, 0, 0
    with torch.no_grad():
        emit("FULL_ORIGINAL_POINT_CONTEXT", "BEGIN")
        tokens, context = model.context(source)
        emit("FULL_ORIGINAL_POINT_CONTEXT", "END", known_objects=dict(tokens=tokens),
            original_points_consumed=context["original_points_consumed"])
        for ordinal, carrier in enumerate(carriers):
            emit("CARRIER_LAYOUT", "BEGIN", carrier_ordinal=ordinal)
            layout = SequenceLayout.from_carrier(carrier)
            detail = layout.rows >= 8192 or ordinal % 500 == 0 or 104000 <= ordinal < 105000
            def profile(**values):
                if detail or values.get("selected_starts", 0):
                    events(**values)
            emit("CARRIER_LAYOUT", "END", carrier_ordinal=ordinal, rows=layout.rows,
                legal_slots=carrier.slot_count,
                known_objects={key: value for key, value in vars(layout).items() if isinstance(value, np.ndarray)})
            gate = scan_complete_child_start_gate(model, tokens, provider, layout, profile,
                compare_original_whole_sigmoid=compare_original_whole_sigmoid)
            calls += gate.role_block_forwards
            differentials += int(gate.whole_carrier_sigmoid_differential_checked)
            if gate.selected_count:
                nonempty += 1
                heads, complete_calls = materialize_complete_heads_after_gate(model, tokens, provider, layout, gate, profile)
                calls += complete_calls
                emit("ORIGINAL_NONZERO_DECODING", "BEGIN", carrier_ordinal=ordinal,
                    rows=layout.rows, legal_slots=carrier.slot_count, selected_starts=gate.selected_count)
                decoded = decode_gate_checked_consuming_heads(model, carrier, layout, heads, gate, queries)
                assert not heads
                emit("ORIGINAL_NONZERO_DECODING", "END", carrier_ordinal=ordinal,
                    selected_starts=gate.selected_count, complete_prediction_children=decoded["total_prediction_children"])
                del heads
            else:
                empty += 1
                decoded = empty_gate_decoded_result(gate, layout, queries)
            parents.extend(decoded["parents"])
            invalid.extend(decoded["invalid_predictions"])
            polyline_points += sum(len(s["points"]) for p in decoded["parents"] for s in p["segments"])
            queries = decoded["next_query_index"]
            slots += carrier.slot_count
            del decoded, layout, gate
            emit("CARRIER_RELEASED", "END", carrier_ordinal=ordinal, complete_carriers=ordinal + 1,
                cumulative_selected_start_outputs=queries, cumulative_parents=len(parents),
                cumulative_invalid_rows=len(invalid), cumulative_emitted_polyline_points=polyline_points)
            if ordinal % 100 == 0 or ordinal + 1 == len(carriers):
                report(stage="FRESH_ALL_CELL_STREAMED_ROLE_GATE_PREDICTION", complete_carriers=ordinal + 1,
                    original_carriers=len(carriers), complete_slots=slots, selected_start_outputs=queries,
                    sequence_forwards=calls, empty_start_carriers=empty, nonempty_start_carriers=nonempty,
                    actual_carriers_with_all_slot_M0_M1_boolean_differential=differentials)
    emit("ORIGINAL_PROCESS_PATH_NORMALIZATION", "BEGIN", cumulative_parents=len(parents), cumulative_invalid_rows=len(invalid))
    result = normalize_process_paths(dict(parents=parents, invalid_predictions=invalid))
    emit("ORIGINAL_PROCESS_PATH_NORMALIZATION", "END", total_prediction_children=result["total_prediction_children"])
    result.update(version=VERSION, full_original_context=context, fresh_XYZ_generation=generation,
        all_exact_alias_capacity_domains=len(carriers), all_declared_slots_processed=slots,
        sequence_forwards=calls, source_candidate_caches_read=False, training_feature_caches_read=False,
        target_count_used=False, target_endpoint_windows_used=False, GT_or_STEP_queries=0,
        signed_board_normal_accuracy="NOT_MEASURED", all_E16_accuracy="NOT_MEASURED", full_contract_pass=False,
        engineering_storage_version="M0_M1_SCALAR_ROLE_GATE_FROZEN_RUNTIME_349_v1",
        role_gate_empty_carriers=empty, role_gate_nonempty_carriers=nonempty,
        actual_carriers_with_all_slot_M0_M1_boolean_differential=differentials,
        actual_differential_reference_storage_mode=bool(compare_original_whole_sigmoid),
        actual_all_slot_boolean_and_ordered_identity_checked=bool(compare_original_whole_sigmoid) and differentials == len(carriers),
        original_five_role_head_no_other_head_calls_during_role_scan=True,
        original_nonzero_full_heads_pointer_links_projection_false_ledger_preserved=True,
        registered_nonfinite_diagnostic_cannot_be_reported_as_empty=True)
    return result
