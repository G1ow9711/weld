"""TRAIN-only prior features, complete supervision and finite XYZ associations."""
from pathlib import Path
import gc
import numpy as np
import torch
from weld_pipeline.legacy_incremental_supervision_388 import read, digest, load_fresh_training_interfaces
from weld_pipeline.legacy_incremental_features_395 import prepare_candidate_inputs
from weld_pipeline.legacy_legal_allocation_model_518 import LegalAllocationContainer, add_current_domain
from weld_pipeline.legacy_scale_count_training_data_487 import pack_source
from weld_pipeline.legacy_incremental_metric_context_485 import raw_xyz_context
from weld_pipeline.finite_plate_support_522 import build_finite_surface_index

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "weld_pipeline/reports/goal_contract_training_20261005"


def load_stage(device="cuda"):
    decision = read(HERE / "scale520_Pro_local_adoption_514.json")
    assert decision["stage"] == "RECOVER325_LEGAL_PATH_FINITE_PLATE_U130_v1"
    assert read(HERE / "scale520_Pro_local_adoption_514_tool_terminal.json")["actual_terminal"]["exit_code"] == 0
    assert read(HERE / "owned_failure_localization_517_attempt1/controller_tool_terminal.json")["actual_terminal"]["exit_code"] == 0
    checkpoint = ROOT / decision["parent_checkpoint_path"]
    assert digest(checkpoint) == decision["parent_checkpoint_sha256"]
    prior = torch.load(checkpoint, map_location="cpu")
    recovery = torch.load(checkpoint.parent / "latest_recovery_state.pt", map_location="cpu")
    assert prior["actual_optimizer_updates"] == recovery["actual_optimizer_updates"] == 325
    assert len(recovery["optimizer_state"]["state"]) == 50
    model = LegalAllocationContainer.from_bundle(prior).to(device).eval()
    groups = []
    for category in ("H", "V"):
        head = getattr(model.events, category)
        groups += [dict(params=head.classification_parameters(), lr=1e-4, phase_name=category + "_classification_margin"),
                   dict(params=list(head.allocation.parameters()), lr=3e-4, phase_name=category + "_length_allocation")]
    groups.append(dict(params=list(model.attributes.parameters()) + list(model.path_attributes.parameters()),
                       lr=3e-4, phase_name="finite_plate_attributes"))
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    assert not optimizer.state
    metadata = dict(decision=decision,
        prior_adoption=read(HERE / "emission325_Pro_local_adoption_483.json"),
        rows=read(HERE / "actual_legacy_incremental_full65_supervision_413_attempt1/manifest.json"),
        feedback=read(HERE / "corrective260_error_feedback_446_attempt1/manifest.json"),
        earlier=read(HERE / "first2_error_feedback_421_attempt2/manifest.json"),
        baseline=read(HERE / "actual_legacy_original_full65_contract_D3_score_411_attempt1/receipt.json")["results"],
        raw=read(HERE / "owned_emission_free65_466_attempt1/registered_protocol.json")["original_PCDFiles"],
        seals=read(HERE / "actual_emission_free65_resume_478_attempt1/all_predictions_frozen_before_any_GT_scoring.json")["completed_inputs"])
    migration = dict(new_semantic_heads_fresh_state=True, compatible_active_old_state_entries=0,
        old50_optimizer_states_archived_unchanged=True, old_endpoint_moments_not_reshaped=True,
        parent_optimizer_state_archive_path=str((checkpoint.parent / "latest_recovery_state.pt").relative_to(ROOT)),
        parent_optimizer_state_archive_sha256=digest(checkpoint.parent / "latest_recovery_state.pt"),
        frozen325_features_and_classification_prior=True, H_V_independent_increment_parameters=True)
    return model, optimizer, migration, prior, recovery, metadata


def prepare_inputs(index, prior, metadata):
    entry = metadata["rows"][index]
    ref = entry["mapping"]["hybrid"]
    path = ROOT / ref["path"]
    assert digest(path) == ref["sha256"]
    mapping = read(path)
    fresh = load_fresh_training_interfaces((ROOT / mapping["fresh_capture_manifest_path"]).parent,
                                          mapping["fresh_capture_manifest_sha256"])
    sequences, arrays = fresh["sequences"], fresh["arrays"]
    probabilities = [arrays["original_parent_probabilities"][r["start"]:r["stop"]]
                     for r in fresh["manifest"]["sequences"]]
    folder = HERE / "actual_legacy_original_full65_375_attempt1" / ("input_%02d" % index)
    diagnostic = folder / "hybrid/vertical_diagnostic.json"
    receipt = read(folder / "receipt.json")
    frozen = {k.replace("\\", "/"): v for k, v in receipt["routes"]["hybrid"]["frozen_outputs_sha256"].items()}
    assert digest(diagnostic) == frozen["hybrid/vertical_diagnostic.json"]
    config = prior["config"]
    features, base, overflow = prepare_candidate_inputs(sequences, probabilities, arrays["structural"], arrays["process"],
        read(diagnostic), prior["original_hybrid"]["plate_heads"]["split_fractions"],
        config["parent_capacity"], config["child_capacity"])
    assert not overflow
    return sequences, features, add_current_domain(base, sequences), mapping


def make_source(index, prior, metadata, hypothesis_cache=None):
    entry = metadata["rows"][index]
    v, h = entry["mapping"]["hybrid"], entry["mapping"]["legacy115"]
    fb, old = metadata["feedback"][index], metadata["earlier"][index]
    teachers = sorted(set(metadata["baseline"]["legacy115"]["all_old_correct_child_identity_by_source"][index]) |
                      set(metadata["baseline"]["hybrid"]["all_old_correct_child_identity_by_source"][index]))
    score = HERE / "actual_emission_guard_full65_score_467_attempt1" / ("source_%02d.json" % index)
    expected = metadata["prior_adoption"]["training_only_keeping_identity_bindings_sha256"][str(score.relative_to(ROOT))]
    from training.dataio import read_xyz_pcd
    original = ROOT / metadata["raw"][index]["original_pcd_path"]
    assert digest(original) == metadata["raw"][index]["original_pcd_sha256"]
    xyz = read_xyz_pcd(original) * 1000.
    assert len(xyz) == metadata["raw"][index]["original_points"]
    source = pack_source(ROOT, HERE / "actual_legacy_original_full65_375_attempt1" / ("input_%02d" % index),
        ROOT / v["path"], v["sha256"], prior["original_hybrid"]["plate_heads"],
        prior["config"]["parent_capacity"], prior["config"]["child_capacity"], teachers, ROOT / h["path"], h["sha256"],
        feedback_path=ROOT / fb["path"], feedback_sha256=fb["sha256"],
        first2_feedback_path=ROOT / old["path"], first2_feedback_sha256=old["sha256"],
        first2_free_directory=HERE / "actual_frozen_incremental_first2_free65_407_attempt3" / ("input_%02d" % index),
        first2_score_path=HERE / "actual_incremental_first2_full65_independent_score_406_attempt1" / ("source_%02d.json" % index),
        original_pcd_entry=metadata["raw"][index], raw_context=raw_xyz_context(xyz),
        new_H_correct_ids=metadata["prior_adoption"]["E3_new_correct_H_identities_training_only_by_source"][index],
        current325_case_score_path=score, current325_case_score_sha256=expected,
        current325_free_directory=(ROOT / metadata["seals"][index]["receipt_path"]).parent)
    sequences, features, base, mapping = prepare_inputs(index, prior, metadata)
    assert np.array_equal(features, source["features"])
    source["base"] = base
    source["group_ids"] = base["allocation_group_ids"]
    cache = Path(hypothesis_cache) if hypothesis_cache is not None else None
    if cache is not None and cache.exists():
        stored = read(cache)
        assert stored["original_PCD_sha256"] == digest(original)
        assert stored["TRAIN_target_location_queries_only_not_runtime_lookup"]
        source["path_plate_hypotheses"] = stored["path"]
        source["local_plate_hypotheses"] = stored["local"]
    else:
        indexer = build_finite_surface_index(xyz)
        # Existing XYZ normals propose every original candidate's planes.
        for seq in sequences:
            center = (seq["xyz"][0] + seq["xyz"][-1]) / 2.
            for normal in np.asarray(seq["features"])[len(seq["features"]) // 2, 60:66].reshape(2, 3):
                if np.linalg.norm(normal) > 1e-8:
                    indexer.propose(normal, center)
        canonical = read(ROOT / mapping["E16_canonical_payload_path"])["children"]
        by_id = {cid: (row, ci) for row in mapping["mapped_parents"] for ci, cid in enumerate(row["original_child_ids"])}
        path, local = [], []
        for child in canonical:
            attached = by_id.get(child["original_child_id"])
            if attached is None:
                result = {"fields": [[] for _ in range(16)], "unresolved_fields_explicit": True}
            else:
                row, ci = attached; a, z = row["child_parameter_endpoints_mm"][ci]
                result = indexer.ordered_hypotheses(sequences[row["sequence_index"]], a, z)
            path.append(dict(fields=result["fields"], unresolved_fields_explicit=True))
            for sample in child["ordered_auxiliary_material_samples"]:
                if attached is None or not sample["pointwise_payload_valid"]:
                    item = {"fields": [[] for _ in range(16)]}
                else:
                    position = a + float(sample["fraction"]) * (z - a)
                    item = indexer.ordered_hypotheses(sequences[row["sequence_index"]], position, position)
                local.append(dict(fields=item["fields"], unresolved_fields_explicit=True))
        source["path_plate_hypotheses"], source["local_plate_hypotheses"] = path, local
        if cache is not None:
            import json
            cache.parent.mkdir(parents=True, exist_ok=True)
            def plain(value):
                if isinstance(value, np.ndarray): return value.tolist()
                if isinstance(value, np.generic): return value.item()
                raise TypeError(type(value).__name__)
            with cache.open("x", encoding="utf-8") as stream:
                json.dump(dict(original_PCD_sha256=digest(original), path=path, local=local,
                    TRAIN_target_location_queries_only_not_runtime_lookup=True,
                    full_original_rows_retained=indexer.original_rows), stream, default=plain, allow_nan=False)
        del indexer
    assert len(source["path_plate_hypotheses"]) == len(source["path_features"])
    assert len(source["local_plate_hypotheses"]) == source["original_local_points"]
    for key in ("path_metric_context", "local_metric_context", "raw_xyz_metric_context"):
        source.pop(key, None)
    del xyz, sequences
    gc.collect()
    return source
