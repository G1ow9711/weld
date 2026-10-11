"""TRAIN-only complete supervision and current-XYZ metric features.

Correct identity tables affect losses/anchors only; never neural input columns.
The free predictor never imports this module or reads these sealed captures.
"""
from pathlib import Path
import gc
import numpy as np
from weld_pipeline.legacy_incremental_emission_training_data_456 import pack_source as emission_pack
from weld_pipeline.legacy_incremental_supervision_388 import read, digest, load_fresh_training_interfaces
from weld_pipeline.legacy_incremental_metric_context_485 import raw_xyz_context, metric_context, CONTEXT_COUNT
from weld_pipeline.reports.goal_contract_training_20261005.score_legacy_full65_frozen_385 import adapt_predictions


def pack_source(*args, original_pcd_entry, new_H_correct_ids, current325_case_score_path,
                current325_case_score_sha256, current325_free_directory, raw_context=None, **kwargs):
    root, free_directory, mapping_path = Path(args[0]), Path(args[1]), Path(args[2])
    if raw_context is None:
        # Current original PCD is used for unsupervised numeric context only.
        from training.dataio import read_xyz_pcd
        original = root / original_pcd_entry["original_pcd_path"]
        assert digest(original) == original_pcd_entry["original_pcd_sha256"]
        xyz_m = read_xyz_pcd(original)
        assert len(xyz_m) == original_pcd_entry["original_points"]
        raw_context = raw_xyz_context(xyz_m * 1000.)
        del xyz_m
        gc.collect()
    source = emission_pack(*args, **kwargs)
    source["hard_negative_weights"] = np.ones_like(source["hard_negative_weights"])
    mapping = read(mapping_path)
    fresh = load_fresh_training_interfaces((root / mapping["fresh_capture_manifest_path"]).parent,
                                           mapping["fresh_capture_manifest_sha256"])
    sequences = fresh["sequences"]
    assert len(sequences) == len(source["features"])
    canonical_path = root / mapping["E16_canonical_payload_path"]
    assert digest(canonical_path) == mapping["E16_canonical_payload_sha256"]
    children = read(canonical_path)["children"]
    assert [row["original_child_id"] for row in children] == source["path_original_child_ids"]
    by_child = {cid: (record, ci) for record in mapping["mapped_parents"]
                for ci, cid in enumerate(record["original_child_ids"])}
    path_context = np.zeros((len(children), CONTEXT_COUNT), np.float32)
    local_context = []
    for index, child in enumerate(children):
        attached = by_child.get(child["original_child_id"])
        if attached is not None:
            record, ci = attached
            sequence = sequences[record["sequence_index"]]
            a, z = record["child_parameter_endpoints_mm"][ci]
            path_context[index] = metric_context(sequence, min(a, z), max(a, z), raw_context)
        for sample in child["ordered_auxiliary_material_samples"]:
            if attached is not None and sample["pointwise_payload_valid"]:
                position = a + float(sample["fraction"]) * (z - a)
                local_context.append(metric_context(sequence, position, position, raw_context))
            else:
                local_context.append(np.zeros(CONTEXT_COUNT, np.float32))
    source["path_metric_context"] = path_context
    source["local_metric_context"] = np.asarray(local_context, np.float32).reshape(-1, CONTEXT_COUNT)
    assert len(local_context) == len(source["local_features"]) == source["original_local_points"]
    assert np.isfinite(path_context).all() and np.isfinite(source["local_metric_context"]).all()
    original_counts = mapping["category_original_G"]
    group_counts = {0: int(original_counts.get("horizontal_line", 0)),
                    1: int(original_counts.get("vertical_line", 0)),
                    2: sum(int(v) for key, v in original_counts.items() if key not in ("horizontal_line", "vertical_line"))}
    assert sum(group_counts.values()) == source["original_G"]
    source["original_G_by_group"] = group_counts
    protected = set(source["protected_original_child_ids_training_only"]) | set(new_H_correct_ids)
    for records in source["parents"]:
        for record in records:
            ids = record["original_child_ids"]
            canonical_ids = ids[::-1] if record["parent_chart_direction"] == -1 else ids
            record["protected_keep_child_mask"] = np.asarray([cid in protected for cid in canonical_ids], bool)
            keep_world = np.asarray(record["canonical_original_world_endpoints_mm"], np.float32).copy()
            # 130 V references retain all303, including the ten currently lost.
            if "corrective_keep_child_mask" in record:
                mask = np.asarray(record["corrective_keep_child_mask"], bool)
                keep_world[mask] = np.asarray(record["corrective_keep_world_endpoints_mm"])[mask]
            record["protected_keep_world_endpoints_mm"] = keep_world
            record["protected_complete_parent"] = all(cid in protected for cid in ids)
    # E3's newly correct H can add TRAIN query anchors only when the full GT
    # parent matched one emitted parent with the same current candidate/count.
    score_path = Path(current325_case_score_path)
    assert digest(score_path) == current325_case_score_sha256
    case = read(score_path)
    current_folder = Path(current325_free_directory)
    current_receipt = read(current_folder / "receipt.json")
    prediction_path = current_folder / "shadow_Hline/predictions.json"
    keys = {key.replace("\\", "/"): value for key, value in current_receipt["frozen_outputs_sha256"].items()}
    assert digest(prediction_path) == keys["shadow_Hline/predictions.json"]
    predictions = read(prediction_path)
    adapted = adapt_predictions(predictions, read(current_folder / "shadow_Hline/invalid_positive_ledger.json"))
    matches = {row["original_child_id"]: row for row in case["shadow_E_ordered_geometry"]["matches"]}
    proposals, current_anchors, rejected = {}, {}, 0
    for si, records in enumerate(source["parents"]):
        for record in records:
            if record.get("fixed_train_query") is not None:
                key = (si, int(record["fixed_train_query"]))
                assert key not in current_anchors
                current_anchors[key] = record
            if int(source["group_ids"][si]) != 0 or not record["protected_complete_parent"]:
                continue
            ids, n = record["original_child_ids"], record["original_child_count"]
            if not any(cid in new_H_correct_ids for cid in ids) or not all(cid in matches for cid in ids):
                continue
            owners = {matches[cid]["predicted_parent_index"] for cid in ids}
            if len(owners) != 1:
                continue
            original_index = adapted["parents"][owners.pop()]["original_prediction_index"]
            parent = predictions[original_index]
            if parent["type"] != "horizontal" or len(parent["segments"]) != n:
                continue
            if int(parent["original_candidate_sequence_index"]) != si:
                rejected += 1
                continue
            proposals.setdefault((si, int(parent["learned_parent_query_index"])), []).append(record)
    added_anchors = 0
    for key, records in proposals.items():
        if len(records) != 1 or (key in current_anchors and current_anchors[key] is not records[0]):
            rejected += len(records)
            continue
        record = records[0]
        if record.get("fixed_train_query") is None:
            record["fixed_train_query"] = key[1]
            current_anchors[key] = record
            added_anchors += 1
    for si, records in enumerate(source["parents"]):
        anchored = [int(record["fixed_train_query"]) for record in records if record.get("fixed_train_query") is not None]
        assert len(anchored) == len(set(anchored))
    source.update(raw_xyz_metric_context=raw_context, E3_new_H_correct_id_count_training_only=len(new_H_correct_ids),
        E3_new_complete_H_train_anchors=added_anchors, E3_anchor_conflicts_or_other_owner_kept_unanchored=rejected,
        protected_original_child_ids_training_only=sorted(protected),
        metric_features_have_no_identifiers_or_target_measurements=True,
        approved_whole_truth_fields_created=0, background_extra4_removed_not_negative_examples=True,
        current325_case_score_path_training_only=str(score_path.relative_to(root)))
    return source
