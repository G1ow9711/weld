"""Require all65 actual normal terminals and a sealed separate score before GT audits."""
from pathlib import Path

from .reports.serial_training_recovery_310_20261004.verify_first_actual_resume_update_310 import read, digest


def require_actual_complete_final600_baseline(root, owner, free, score, expected_tool_session):
    root, owner, free, score = [Path(p).resolve() for p in (root, owner, free, score)]
    if not all(p.is_relative_to(root) for p in (owner, free, score)):
        raise ValueError("Baseline evidence must remain inside the current project")
    required = [owner / "receipt.json", owner / "controller_tool_terminal.json",
        owner / "registered_protocol.json", owner / "registration_receipt.json",
        free / "receipt.json", score / "receipt.json", owner / "strict_all65_actual_terminal.json"]
    if any(not p.is_file() for p in required):
        raise RuntimeError("Complete65_actual_frozen_outputs_original_normal0_tool_and_separate_score_required_before_GT_audit")
    controller = read(owner / "controller_tool_terminal.json")
    if controller["original_tool_session"] != expected_tool_session or controller["actual_tool_exit_code"] != 0:
        raise RuntimeError("The original whole65 owner tool did not complete with actual normal0")
    protocol_path = owner / "registered_protocol.json"
    if digest(protocol_path) != read(owner / "registration_receipt.json")["registered_protocol_sha256"]:
        raise ValueError("Frozen whole65 protocol changed")
    protocol = read(protocol_path)
    if (protocol["source_packages"], protocol["original_XYZ"], protocol["original_G"]) != (65, 48928945, 3593):
        raise ValueError("The whole65 original scope was narrowed")
    owner_receipt, free_receipt, score_receipt = [read(p / "receipt.json") for p in (owner, free, score)]
    assert owner_receipt["actual_natural0_prediction_natives"] == 65
    assert owner_receipt["actual_natural0_independent_score_native"]
    assert owner_receipt["free_receipt_sha256"] == digest(free / "receipt.json")
    assert owner_receipt["score_receipt_sha256"] == digest(score / "receipt.json")
    assert free_receipt["source_packages"] == score_receipt["source_packages"] == 65
    assert free_receipt["all65_native_natural0"] and free_receipt["all_predictions_frozen_before_GT_reads"]
    assert score_receipt["original_G"] == 3593 and score_receipt["all_predictions_frozen_before_this_process_read_any_GT"]
    score_terminal = read(owner / "strict_all65_actual_terminal.json")
    assert score_terminal["native_exit_code"] == score_terminal["launcher_exit_code"] == 0
    assert score_terminal["held_original_Popen_kernel_handle"] and not score_terminal["resource_terminated"]
    freeze_path = (root / free_receipt["freeze_path"]).resolve()
    assert freeze_path.is_relative_to(root)
    assert digest(freeze_path) == free_receipt["freeze_sha256"] == score_receipt["prediction_freeze_sha256"]
    freeze = read(freeze_path)
    assert len(freeze["outputs"]) == 65 and freeze["all65_prediction_natives_natural0"]
    seen, original_points = set(), 0
    for row in freeze["outputs"]:
        index = row["source_index_metadata_only"]
        assert type(index) is int and 0 <= index < 65 and index not in seen
        seen.add(index)
        entry = protocol["original_PCD_input_book"][index]
        assert entry["source_index_metadata_only"] == index
        assert row["original_points"] == entry["original_points"]
        assert row["original_pcd_sha256"] == entry["original_pcd_sha256"]
        assert row["model_content_sha256"] == protocol["final_model_content_sha256"]
        assert row["final_checkpoint_sha256"] == protocol["final_checkpoint_sha256"]
        assert row["no_GT_target_file_read_by_this_native"] and row["actual_native_natural0"]
        prediction_path = (root / row["prediction_path"]).resolve()
        assert prediction_path.is_relative_to(root) and digest(prediction_path) == row["prediction_sha256"]
        terminal_path = owner / f"source_{index:02d}_actual_terminal.json"
        assert digest(terminal_path) == row["actual_terminal_sha256"]
        terminal = read(terminal_path)
        assert terminal["native_exit_code"] == terminal["launcher_exit_code"] == 0
        assert terminal["held_original_Popen_kernel_handle"] and not terminal["resource_terminated"]
        assert terminal["native_identity"] == row["native_identity"]
        reported = terminal["worker_reported_identity"]
        assert (reported["process_id"], reported["creation_filetime_ticks"]) == (
            terminal["native_identity"]["process_id"], terminal["native_identity"]["creation_filetime_ticks"])
        original_points += row["original_points"]
    assert seen == set(range(65)) and original_points == 48928945
    return dict(scope="ACTUAL_FINAL600_ALL65_SEALED_FREE_BASELINE_AND_SEPARATE_GEOMETRY_SCORE",
        source_packages=65, original_XYZ=48928945, original_G=3593,
        original_owner_tool_session=expected_tool_session, actual_owner_tool_exit_code=0,
        protocol_sha256=digest(protocol_path), freeze_path=str(freeze_path.relative_to(root)),
        freeze_sha256=digest(freeze_path), free_receipt_sha256=digest(free / "receipt.json"),
        score_receipt_sha256=digest(score / "receipt.json"),
        same_final600_model_content_sha256=protocol["final_model_content_sha256"],
        geometry_score_pass_not_required_for_causal_diagnosis=True,
        formal_contract_completion_not_proved_by_this_prerequisite=True, full_contract_pass=False)
