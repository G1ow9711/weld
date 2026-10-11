"""Counted all25 B training, atomic complete-source recovery and FIT scoring.

Run budget belongs to the frozen protocol. This is neither a toy update nor
an audit that rolls back parameter changes. Only complete persisted updates
are reported as committed training. Interrupted AdamW intents stay explicit.
"""
import io
import json
import os
from pathlib import Path
import random
import shutil
import time
from datetime import datetime, timezone
from hashlib import sha256

import numpy as np
import torch

from .live_json_snapshot_282 import write_json_snapshot as dump, read_json_snapshot
from .raw_point_groups_285 import RawPointSource
from .source_checkpoint_232 import finite_state, tree_hash, gradient_receipt, parameter_binding, rng_payload
from .world_set_model_285 import WorldSetModel
from .world_set_loss_285 import full_world_set_loss
from .world_set_runtime_285 import decode_world_predictions
from .world_set_evaluation_285 import score_source, aggregate_sources
from .reports.full25_source_geometry_263_20261003.run_fresh_source_geometry_263 import memory, process_identity

VERSION = "COUNTED_ALL25_COMPLETE_SOURCE_WORLD_NEURAL_TRAINING_285_v1"


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_sha(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cursor(completed, updates, protocol):
    if not (type(completed) is int and type(updates) is int and 0 <= updates <= completed <= protocol["maximum_complete_sources"]):
        raise ValueError("Counted B budget/cursor invalid")
    return dict(complete_sources=completed, actual_committed_optimizer_updates=updates,
        epoch=completed//25, slot=completed % 25)


def rng_state(model):
    values = rng_payload(model, {}, {})
    values["source_order_generator_policy"] = "REGISTERED_FIXED_GLOBAL_0_TO_24_ORDER_NO_MUTABLE_GENERATOR"
    return values


def atomic_bytes(path, contents):
    path = Path(path)
    temporary = path.with_name(path.name+".%d.tmp" % os.getpid())
    with temporary.open("wb") as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())
    deadline = time.monotonic()+15
    while True:
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.025)
    return dict(path=str(path), sha256=sha256(contents).hexdigest(), bytes=len(contents))


def commit(directory, model, optimizer, current, protocol, source_receipt=None):
    """Two generations retain the preceding complete checkpoint until commit."""
    state = finite_state(model, optimizer)
    rng = rng_state(model)
    protocol_sha = sha256(json.dumps(protocol, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    payload = dict(version=VERSION, state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
        cursor=current, protocol=protocol, protocol_content_sha256=protocol_sha,
        RNG=rng, RNG_content_sha256=tree_hash(rng), state_receipt=state,
        model_training_mode=model.training, source_receipt=source_receipt)
    output = io.BytesIO()
    torch.save(payload, output)
    contents = output.getvalue()
    completed = current["complete_sources"]
    target = directory/("latest_%s.pt" % ("A" if completed % 2 == 0 else "B"))
    artifact = atomic_bytes(target, contents)
    pinned = None
    if completed in (0, 1) or completed % 25 == 0:
        label = "initial_zero" if completed == 0 else ("first_real_update" if completed == 1 else "epoch_%03d" % (completed//25))
        pinned = atomic_bytes(directory/(label+".pt"), contents)
    descriptor = dict(version=VERSION, utc=utc(), status="COMMITTED_COMPLETE_SOURCE_MODEL_ADAMW_RNG",
        cursor=current, protocol_content_sha256=protocol_sha, checkpoint=artifact,
        pinned_checkpoint=pinned, state_receipt=state, RNG_content_sha256=payload["RNG_content_sha256"],
        source_receipt=source_receipt, full_contract_pass=False)
    dump(directory/"latest_commit.json", descriptor)
    # Immutable historical descriptors record the bytes once committed. The
    # rotating generations are recovery storage, not 2001 available snapshots.
    history = dict(descriptor)
    history["historical_rotating_checkpoint_may_be_replaced"] = True
    history_path = directory/"history"/("source_%04d_commit.json" % completed)
    if history_path.exists():
        raise ValueError("Cannot replace an already committed source history")
    dump(history_path, history)
    return descriptor


def restore(directory, model, optimizer, protocol):
    descriptor = read_json_snapshot(directory/"latest_commit.json")
    artifact = descriptor["checkpoint"]
    if file_sha(artifact["path"]) != artifact["sha256"]:
        raise ValueError("Latest committed B checkpoint hash differs")
    with Path(artifact["path"]).open("rb") as stream:
        saved = torch.load(stream, map_location="cpu")
    if saved["version"] != VERSION or saved["protocol"] != protocol or saved["cursor"] != descriptor["cursor"]:
        raise ValueError("B method/config/cursor changed")
    expected = parameter_binding(model, optimizer)
    if saved["state_receipt"]["parameter_binding"] != expected:
        raise ValueError("AdamW parameter names/order/options/device changed")
    model.load_state_dict(saved["state_dict"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    if finite_state(model, optimizer) != saved["state_receipt"]:
        raise ValueError("Complete restored model/AdamW binding differs")
    rng = saved["RNG"]
    if rng["CUDA"] is not None or rng["independent_numpy"] or rng["independent_torch"]:
        raise ValueError("Registered B CPU RNG registry changed")
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if tree_hash(rng_state(model)) != saved["RNG_content_sha256"]:
        raise ValueError("Actual restored B RNG differs")
    model.train(saved["model_training_mode"])
    c = saved["cursor"]
    if c != cursor(c["complete_sources"], c["actual_committed_optimizer_updates"], protocol):
        raise ValueError("Complete-source recovery cursor invalid")
    intent_path = directory.parent/"optimizer_step_intent.json"
    if intent_path.exists():
        intent = read(intent_path)
        if intent["complete_sources_after"] > c["complete_sources"]:
            raise ValueError("Uncommitted optimizer intent exists; preserve rather than silently replay/reset")
    return c, descriptor


def validate_protocol(root, protocol):
    if (protocol["version"] != VERSION or protocol["device"] != "cpu" or protocol["epochs"] != 80
            or protocol["maximum_complete_sources"] != 2000 or protocol["global_parent_queries"] != 512
            or protocol["global_child_capacity"] != 16 or protocol["formal_E_reviewed_cells"] != 0):
        raise ValueError("Frozen first-round B registration changed")
    for path, expected in protocol["source_code_sha256"].items():
        if file_sha(root/path) != expected:
            raise ValueError("Frozen B source changed: "+path)
    for path, expected in protocol["artifacts_sha256"].items():
        if file_sha(root/path) != expected:
            raise ValueError("Registered B input/loss/evidence artifact changed: "+path)
    inputs = read(root/protocol["input_scope_path"])
    targets = read(root/protocol["supervision_scope_path"])
    original_inputs = read(root/protocol["original_raw_input_scope_path"])
    if len(inputs) != 25 or len(targets) != 25:
        raise ValueError("All25 original sources required")
    if len(original_inputs) != 25 or sum(v["original_points"] for v in original_inputs) != 15387103:
        raise ValueError("All original25 point denominators required")
    for index, original in enumerate(original_inputs):
        if original["source_index_metadata_only"] != index:
            raise ValueError("Original PCD provenance order changed")
        if file_sha(original["original_pcd_path"]) != original["original_pcd_sha256"]:
            raise ValueError("Original input PCD changed")
    for index, (entry, target) in enumerate(zip(inputs, targets)):
        if entry["source_index_metadata_only"] != index or target["source_index_metadata_only"] != index:
            raise ValueError("Registered global source order changed")
        if file_sha(root/entry["group_directory"]/"receipt.json") != entry["receipt_sha256"]:
            raise ValueError("Pure original point input receipt changed")
        if file_sha(root/target["loss_target_path"]) != target["loss_target_sha256"]:
            raise ValueError("Full original supervision changed")
    if sum(t["original_children"] for t in targets) != 1605 or sum(t["original_parents"] for t in targets) != 1461:
        raise ValueError("Original parent/child denominator changed")
    return inputs, targets


def train_registered_round(root, protocol_path, output, resume=False):
    root, protocol_path, output = Path(root), Path(protocol_path), Path(output)
    if resume:
        if not output.is_dir():
            raise ValueError("Cannot resume a missing registered round")
    else:
        output.mkdir(exist_ok=False)
    identity = process_identity()
    identity["scope"] = "ACTUAL_ORIGINAL_ALL25_B_WORLD_NEURAL_CPU_TRAINING"
    dump(output/"actual_compute_process_identity.json", identity)
    protocol = read(protocol_path)
    current = cursor(0, 0, protocol)
    stage, started = "REGISTERED_PROTOCOL_VALIDATION", time.monotonic()

    def progress(**extra):
        dump(output/"progress.json", dict(version=VERSION, utc=utc(), stage=stage,
            elapsed_seconds=time.monotonic()-started, cursor=current, actual_native_identity=identity,
            resources=memory(), full_contract_pass=False, **extra))

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if torch.version.cuda is not None:
            raise ValueError("First B round requires the actual CPU-only runtime")
        progress()
        inputs, target_entries = validate_protocol(root, protocol)
        if shutil.disk_usage(root).free < protocol["minimum_disk_free_bytes"]:
            raise RuntimeError("Project disk headroom below frozen floor")
        random.seed(protocol["seed"])
        np.random.seed(protocol["seed"])
        torch.manual_seed(protocol["seed"])
        model = WorldSetModel(protocol["global_parent_queries"], protocol["global_child_capacity"],
            hidden=protocol["hidden"], point_block_size=protocol["point_block_size"])
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"],
            weight_decay=protocol["weight_decay"], foreach=False)
        checkpoints = output/"checkpoints"
        if resume:
            current, descriptor = restore(checkpoints, model, optimizer, protocol)
            dump(output/("actual_resume_%s.json" % str(identity["creation_filetime_ticks"])),
                dict(utc=utc(), actual_identity=identity, restored_cursor=current,
                    restored_checkpoint=descriptor["checkpoint"], budget_reset=False))
        else:
            checkpoints.mkdir()
            (checkpoints/"history").mkdir()
            (output/"source_receipts").mkdir()
            (output/"optimizer_executions").mkdir()
            descriptor = commit(checkpoints, model, optimizer, current, protocol)
        stage = "REAL_COMPLETE_SOURCE_TRAINING"
        while current["complete_sources"] < protocol["maximum_complete_sources"]:
            source_index = current["slot"]
            entry, target_entry = inputs[source_index], target_entries[source_index]
            source_start = time.monotonic()
            stage = "PURE_ORIGINAL_XYZ_INPUT_OPEN"
            progress(source_index_metadata_only=source_index)
            source = RawPointSource(root, root/entry["group_directory"])
            targets = read(root/target_entry["loss_target_path"])
            optimizer.zero_grad(set_to_none=True)
            stage = "ACTUAL_ALL_ORIGINAL_POINTS_NEURAL_FORWARD"
            progress(source_index_metadata_only=source_index, original_points=len(source.xyz_mm),
                original_parents=targets["original_parents"], original_children=targets["original_children"])
            prediction = model(source)
            stage = "ACTUAL_FULL_ORIGINAL_PARENT_MATCH_AND_LOSS"
            progress(source_index_metadata_only=source_index, original_points_consumed=prediction["input_receipt"]["original_points_consumed"])
            loss, loss_receipt = full_world_set_loss(prediction, targets)
            stage = "ACTUAL_FULL_SOURCE_BACKWARD"
            progress(source_index_metadata_only=source_index, total_loss=float(loss.detach()))
            loss.backward()
            gradients = gradient_receipt(model)
            if not gradients["nonzero"]:
                raise ValueError("Real source has zero complete gradient; no fabricated training update")
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["maximum_gradient_norm"])
            if not torch.isfinite(pre_clip_norm):
                raise ValueError("Nonfinite global gradient norm")
            clipped = gradient_receipt(model)
            before = tree_hash(model.state_dict())
            completed_after = current["complete_sources"]+1
            intent = dict(version=VERSION, utc=utc(), actual_identity=identity,
                complete_sources_before=current["complete_sources"], complete_sources_after=completed_after,
                optimizer_updates_before=current["actual_committed_optimizer_updates"],
                before_model_sha256=before, source_index_metadata_only=source_index,
                gradient_sha256=tree_hash(clipped), note="Crash here requires explicit reconciliation; not silently replayed")
            dump(output/"optimizer_step_intent.json", intent)
            stage = "REAL_ADAMW_CALL"
            progress(source_index_metadata_only=source_index, optimizer_call_intent=intent)
            optimizer.step()
            after = tree_hash(model.state_dict())
            if before == after:
                raise ValueError("Actual AdamW did not change registered model")
            executed = dict(intent, optimizer_step_returned=True, after_model_sha256=after, returned_utc=utc(),
                parameter_change_verified=True, actual_optimizer_invocations_this_source=1)
            dump(output/"optimizer_executions"/("source_%04d.json" % completed_after), executed)
            next_cursor = cursor(completed_after, current["actual_committed_optimizer_updates"]+1, protocol)
            source_receipt = dict(version=VERSION, utc=utc(), source_index_metadata_only=source_index,
                cursor_before=current, cursor_after=next_cursor, input_receipt=prediction["input_receipt"],
                loss_receipt=loss_receipt, gradient_receipt_before_clip=gradients,
                gradient_receipt_used=clipped, pre_clip_global_norm=float(pre_clip_norm),
                optimizer_execution=executed, complete_source_seconds=time.monotonic()-source_start,
                E_official_supervision_masked=True, accuracy_measured=False, full_contract_pass=False)
            stage = "COMPLETE_SOURCE_CHECKPOINT_COMMIT"
            progress(source_index_metadata_only=source_index, executed_optimizer_update_not_yet_committed=True)
            descriptor = commit(checkpoints, model, optimizer, next_cursor, protocol, source_receipt)
            dump(output/"source_receipts"/("source_%04d.json" % completed_after), source_receipt)
            current = next_cursor
            stage = "COUNTED_REAL_SOURCE_UPDATE_COMMITTED"
            progress(source_index_metadata_only=source_index, last_complete_source_seconds=source_receipt["complete_source_seconds"],
                last_loss=loss_receipt["total_loss"], model_content_sha256=descriptor["state_receipt"]["model_content_sha256"],
                latest_checkpoint=descriptor["checkpoint"])
            print("REAL_COMMITTED_B_UPDATE", current["actual_committed_optimizer_updates"], "EPOCH", current["epoch"],
                "SOURCE", source_index, "LOSS", loss_receipt["total_loss"], flush=True)
            del prediction, loss, source, targets
            if shutil.disk_usage(root).free < protocol["minimum_disk_free_bytes"]:
                raise RuntimeError("Project disk headroom below frozen floor at complete boundary")
        stage = "FULL25_NO_GT_RUNTIME_FIT_EVALUATION"
        evaluation = output/"actual_full25_fit_evaluation"
        evaluation.mkdir(exist_ok=False)
        model.eval()
        scores = []
        with torch.no_grad():
            for source_index, (entry, target_entry) in enumerate(zip(inputs, target_entries)):
                progress(evaluation_sources_completed=len(scores), source_index_metadata_only=source_index)
                source = RawPointSource(root, root/entry["group_directory"])
                output_prediction = model(source)
                decoded = decode_world_predictions(output_prediction)
                dump(evaluation/("source_%02d_predictions.json" % source_index), decoded)
                targets = read(root/target_entry["loss_target_path"])
                score = score_source(decoded, targets)
                dump(evaluation/("source_%02d_score.json" % source_index), score)
                scores.append(score)
                del source, output_prediction, decoded, targets
        aggregate = aggregate_sources(scores)
        aggregate.update(actual_committed_optimizer_updates=current["actual_committed_optimizer_updates"],
            actual_complete_sources=current["complete_sources"], original_sources=25,
            checkpoint=descriptor["checkpoint"], Pro_before_further_training_required=not aggregate["geometry_gates_pass"]
                or not aggregate["full_contract_pass"])
        dump(evaluation/"aggregate.json", aggregate)
        stage = "REGISTERED_ROUND_TRAIN_AND_FIT_TEST_COMPLETE"
        progress(actual_fit_result=aggregate, further_training_budget=0)
        dump(output/"receipt.json", dict(status=stage, utc=utc(), version=VERSION,
            cursor=current, aggregate=aggregate, actual_identity=identity, full_contract_pass=False))
        print("REAL_B_ROUND_COMPLETE_FULL25_FIT", current, aggregate["categories"], flush=True)
    except BaseException as exc:
        dump(output/"failed_actual_training_attempt.json", dict(version=VERSION, utc=utc(), stage=stage,
            last_committed_cursor=current, actual_identity=identity, exception=type(exc).__name__, message=str(exc),
            optimizer_intent_must_be_reconciled=True, model_accuracy_not_inferred=True, full_contract_pass=False))
        raise
