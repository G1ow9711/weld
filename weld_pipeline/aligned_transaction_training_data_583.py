"""TRAIN-only current-query anchors shared with the complete protection pool."""
import numpy as np
import torch

from weld_pipeline.legacy_incremental_supervision_388 import read, digest
from weld_pipeline.legal_allocation_training_data_520 import ROOT, HERE, prepare_inputs
from weld_pipeline.legacy_incremental_emission_training_data_456 import pack_source as emission_pack


class TrainingAlignmentConflict(RuntimeError):
    pass


def load_parent_metadata():
    adoption = read(HERE / "legal_U000_Pro_local_adoption_577.json")
    assert adoption["fully_read_lines"] == 867 and adoption["actual_Pro_exit_code"] == 0
    path = ROOT / adoption["parent_checkpoint_path"]
    assert digest(path) == adoption["parent_checkpoint_sha256"]
    prior = torch.load(path, map_location="cpu")
    audit_dir = HERE / "actual_U000_protected_alignment_audit_578_attempt1"
    assert read(audit_dir / "controller_tool_terminal.json")["actual_terminal"]["exit_code"] == 0
    audit = read(audit_dir / "summary.json")
    assert not audit["structural_conflict_counts"]
    metadata = dict(adoption=adoption,
        prior_adoption=read(HERE / "emission325_Pro_local_adoption_483.json"),
        rows=read(HERE / "actual_legacy_incremental_full65_supervision_413_attempt1/manifest.json"),
        feedback=read(HERE / "corrective260_error_feedback_446_attempt1/manifest.json"),
        earlier=read(HERE / "first2_error_feedback_421_attempt2/manifest.json"),
        baseline=read(HERE / "actual_legacy_original_full65_contract_D3_score_411_attempt1/receipt.json")["results"],
        raw=read(HERE / "owned_emission_free65_466_attempt1/registered_protocol.json")["original_PCDFiles"],
        seals=read(HERE / "actual_emission_free65_resume_478_attempt1/all_predictions_frozen_before_any_GT_scoring.json")["completed_inputs"],
        alignment_directory=audit_dir, alignment_bindings=audit["frozen_inputs_bindings_sha256"])
    return prior, metadata


def prepare_aligned_events_source(index, prior, metadata):
    """Reuse complete supervised interfaces, without constructing material faces."""
    entry = metadata["rows"][index]
    v, h = entry["mapping"]["hybrid"], entry["mapping"]["legacy115"]
    fb, earlier = metadata["feedback"][index], metadata["earlier"][index]
    teacher_ids = sorted(set(metadata["baseline"]["legacy115"]["all_old_correct_child_identity_by_source"][index]) |
                         set(metadata["baseline"]["hybrid"]["all_old_correct_child_identity_by_source"][index]))
    source = emission_pack(ROOT, HERE / "actual_legacy_original_full65_375_attempt1" / ("input_%02d" % index),
        ROOT / v["path"], v["sha256"], prior["original_hybrid"]["plate_heads"],
        prior["config"]["parent_capacity"], prior["config"]["child_capacity"], teacher_ids,
        ROOT / h["path"], h["sha256"], feedback_path=ROOT / fb["path"], feedback_sha256=fb["sha256"],
        first2_feedback_path=ROOT / earlier["path"], first2_feedback_sha256=earlier["sha256"],
        first2_free_directory=HERE / "actual_frozen_incremental_first2_free65_407_attempt3" / ("input_%02d" % index),
        first2_score_path=HERE / "actual_incremental_first2_full65_independent_score_406_attempt1" / ("source_%02d.json" % index))
    sequences, features, base, mapping = prepare_inputs(index, prior, metadata)
    assert np.array_equal(features, source["features"])
    source["base"] = base
    source["group_ids"] = base["allocation_group_ids"]
    source["hard_negative_weights"] = np.ones_like(source["hard_negative_weights"])
    original_counts = mapping["category_original_G"]
    source["original_G_by_group"] = {0: int(original_counts.get("horizontal_line", 0)),
        1: int(original_counts.get("vertical_line", 0)),
        2: sum(int(v) for key,v in original_counts.items() if key not in ("horizontal_line", "vertical_line"))}
    assert sum(source["original_G_by_group"].values()) == source["original_G"]
    current_rows = read(metadata["alignment_directory"] / ("source_%02d.json" % index))["protected_current_query_rows"]
    protected = (set(source["protected_original_child_ids_training_only"]) |
        set(metadata["prior_adoption"]["E3_new_correct_H_identities_training_only_by_source"][index]) |
        {r["original_child_id_TRAIN_only"] for r in current_rows})
    by_parent, by_query = {}, {}
    for row in current_rows:
        key = (row["candidate"], row["query"])
        parent = (row["candidate"], row["original_parent_index_TRAIN_only"])
        if parent in by_parent and by_parent[parent] != key[1]:
            raise TrainingAlignmentConflict("TRAIN_ALIGNMENT_CONFLICT: one GT parent has incompatible protected queries")
        if key in by_query and by_query[key] != parent[1]:
            raise TrainingAlignmentConflict("TRAIN_ALIGNMENT_CONFLICT: one protected query has incompatible GT parents")
        by_parent[parent] = key[1]
        by_query[key] = parent[1]
    changes, current_anchors, released = [], 0, 0
    for si, records in enumerate(source["parents"]):
        for record in records:
            ids = record["original_child_ids"]
            canonical_ids = ids[::-1] if record["parent_chart_direction"] == -1 else ids
            record["protected_keep_child_mask"] = np.asarray([cid in protected for cid in canonical_ids], bool)
            keep_world = np.asarray(record["canonical_original_world_endpoints_mm"], np.float32).copy()
            if "corrective_keep_child_mask" in record:
                mask = np.asarray(record["corrective_keep_child_mask"], bool)
                keep_world[mask] = np.asarray(record["corrective_keep_world_endpoints_mm"])[mask]
            record["protected_keep_world_endpoints_mm"] = keep_world
            record["protected_complete_parent"] = all(cid in protected for cid in ids)
            old = record.get("fixed_train_query")
            current = by_parent.get((si, record["original_parent_index"]))
            if current is not None:
                record["fixed_train_query"] = current
                record["current2098_train_anchor"] = True
                current_anchors += 1
            elif old is not None and (si, int(old)) in by_query:
                record.pop("fixed_train_query", None)
                released += 1
            if record.get("fixed_train_query") != old:
                changes.append(dict(candidate=si, original_parent_index=record["original_parent_index"],
                    old_base_anchor=old, unified_current_anchor=record.get("fixed_train_query"),
                    original_child_ids=ids))
        anchors = [int(r["fixed_train_query"]) for r in records if r.get("fixed_train_query") is not None]
        if len(anchors) != len(set(anchors)):
            raise TrainingAlignmentConflict("TRAIN_ALIGNMENT_CONFLICT: duplicate reserved query after migration")
    located = {(si,r["original_parent_index"]):r for si,records in enumerate(source["parents"]) for r in records}
    for (si, parent), query in by_parent.items():
        record = located.get((si,parent))
        if record is None or record.get("fixed_train_query") != query:
            raise TrainingAlignmentConflict("TRAIN_ALIGNMENT_CONFLICT: protected parent disappeared from loss records")
    source.update(protected_original_child_ids_training_only=sorted(protected),
        current2098_alignment_rows=current_rows, current2098_alignment_changes=changes,
        current2098_train_anchor_count=current_anchors, old_anchor_released_for_current_owner_count=released,
        current_query_alignment_is_TRAIN_only_not_a_runtime_table=True,
        background_extra4_removed_not_negative_examples=True)
    del sequences
    return source
