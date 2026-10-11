"""Bind the verified future dataset without changing any running training job."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "weld_pipeline/reports/full65_material_adoption_338_20261005"
OUT = REPORT / "actual_future3593_material8527_attempt1"
VERIFY = REPORT / "actual_independent_material8527_attempt1"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def binding(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def main():
    build = read(OUT / "receipt.json")
    verified = read(VERIFY / "receipt.json")
    manifest = read(OUT / "manifest.json")
    canonical = read(OUT / "canonical_attribute_registry.json")
    assert len(manifest) == len(canonical) == 65
    assert {row["source_index_metadata_only"] for row in manifest} == set(range(65))
    assert {row["source"] for row in canonical} == set(range(65))
    assert sum(row["original_declared_children"] for row in manifest) == 3593
    assert sum(row["geometry_training_children"] for row in manifest) == 3593
    assert sum(row["original_points"] for row in manifest) == 48928945
    assert digest(OUT / "manifest.json") == build["manifest_sha256"]
    assert digest(OUT / "canonical_attribute_registry.json") == build["canonical_registry_sha256"]

    terminals = []
    for name, expected_pid in [
        ("owned_future3593_material8527_attempt1", 23076),
        ("owned_independent_material8527_attempt1", 29364),
    ]:
        path = REPORT / name / "actual_terminal.json"
        terminal = read(path)
        assert terminal["native_exit_code"] == terminal["launcher_exit_code"] == 0
        assert terminal["held_original_Popen_kernel_handle"] is True
        assert terminal["resource_terminated"] is False
        assert terminal["native_identity"]["process_id"] == expected_pid
        assert terminal["native_identity"]["creation_filetime_ticks"] == terminal["worker_reported_identity"]["creation_filetime_ticks"]
        terminals.append(binding(path))

    for receipt in (build, verified):
        assert receipt["source_groups"] == 65 and receipt["original_G"] == 3593
        assert receipt["future_auxiliary_points"] == 8527
        assert receipt["all_E16_ordinary_cells_unchanged"] == 57488
        assert receipt["reviewed_official_E_cells_added"] == 0
        assert receipt["whole_curve_certificates_added"] == 0
        assert receipt["NN_forwards"] == receipt["actual_optimizer_updates"] == 0
        assert receipt["current600_protocol_or_labels_changed"] is False
        assert receipt["full_contract_pass"] is False
    assert verified["all_original_XYZ_replayed"] == 48928945
    assert verified["original_material_points_preserved"] == 6643
    assert verified["additional_material_points_individually_bound_to_native_provenance"] == 1884
    assert verified["board_exchanges_or_normal_flips"] == 0

    dataset_files = []
    for row in manifest:
        for path_key, sha_key in [("target_path", "target_sha256"), ("attribute_path", "attribute_sha256")]:
            path = ROOT / row[path_key]
            actual = digest(path)
            assert actual == row[sha_key], str(path)
            dataset_files.append({"source": row["source_index_metadata_only"], "role": path_key, "path": row[path_key], "sha256": actual})
    for row in canonical:
        assert digest(ROOT / row["path"]) == row["sha256"], row["path"]
        dataset_files.append({"source": row["source"], "role": "canonical_E16_and_auxiliary", "path": row["path"], "sha256": row["sha256"]})

    descriptor = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "VERIFIED_FUTURE_ALL65_G3593_AUXILIARY8527_DATASET_NOT_YET_USED_FOR_TRAINING",
        "manifest": binding(OUT / "manifest.json"),
        "canonical_attribute_registry": binding(OUT / "canonical_attribute_registry.json"),
        "prepared_receipt": binding(OUT / "receipt.json"),
        "independent_receipt": binding(VERIFY / "receipt.json"),
        "actual_natural_terminal_bindings": terminals,
        "source_packages": 65,
        "original_G": 3593,
        "complete_geometry_children": 3593,
        "recovered15_original_geometry_retained": True,
        "original_XYZ_points": 48928945,
        "original_material_samples": 6643,
        "added_material_samples": 1884,
        "total_material_samples": 8527,
        "newly_covered_children": 409,
        "unchanged_ordinary_E16_cells": 57488,
        "reviewed_official_E_cells_added": 0,
        "whole_curve_certificates_added": 0,
        "auxiliary_samples_are_not_whole_weld_official_truth": True,
        "runtime_STEP_or_GT_or_offline_candidate_bank_reads_allowed": False,
        "current600_protocol_or_labels_changed": False,
        "new_data_training_started": False,
        "next_adoption_gate": build["adoption_gate"],
        "formal_model_accuracy_measured": False,
        "NN_forwards": 0,
        "actual_optimizer_updates": 0,
        "full_contract_pass": False,
        "dataset_file_bindings": dataset_files,
    }
    path = REPORT / "future65_dataset_adoption_descriptor_338.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(descriptor, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    assert read(path) == descriptor
    print(json.dumps({"status": descriptor["status"], "path": str(path), "sha256": digest(path), "dataset_file_bindings": len(dataset_files), "new_data_training_started": False, "full_contract_pass": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
