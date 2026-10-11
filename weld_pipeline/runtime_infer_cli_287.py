"""Single/batch fresh PCD entrypoint, copied to the exported bundle as infer.py."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import torch
from weld_runtime_287.raw_point_groups_285 import RawPointSource, file_sha
from weld_runtime_287.world_set_model_285 import WorldSetModel
from weld_runtime_287.world_set_runtime_285 import decode_world_predictions
from weld_runtime_287.runtime_csv_export_287 import export_decoded, atomic_json, write_csv


def load_bundle(directory):
    configuration = json.loads((directory/"runtime_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory/"runtime_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["file_sha256"].items():
        path = (directory/relative).resolve()
        if not path.is_relative_to(directory.resolve()) or file_sha(path) != expected:
            raise ValueError("RUNTIME_BUNDLE_CONTENT_CHANGED: "+relative)
    if configuration["schema"] != "FRESH_PCD_ONLY_WORLD_NEURAL_RUNTIME_287_v1":
        raise ValueError("RUNTIME_SCHEMA_UNSUPPORTED")
    if torch.version.cuda is not None:
        raise ValueError("REGISTERED_CPU_ONLY_TORCH_RUNTIME_REQUIRED")
    saved = torch.load(directory/"weights.pt", map_location="cpu")
    if set(saved) != {"schema", "state_dict"} or saved["schema"] != configuration["schema"]:
        raise ValueError("WEIGHTS_RUNTIME_KEYS_INVALID")
    model = WorldSetModel(configuration["parent_queries"], configuration["child_capacity"],
        hidden=configuration["hidden"], point_block_size=configuration["point_block_size"])
    model.load_state_dict(saved["state_dict"], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("NONFINITE_TRAINED_WEIGHTS")
    model.eval()
    return model, configuration, manifest


def main():
    parser = argparse.ArgumentParser(description="Predict from original ASCII XYZ PCD only; material fields stay unknown in this B round")
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workpiece-id", help="Output metadata for a single item only; never a model feature")
    args = parser.parse_args()
    if args.workpiece_id and (len(args.input) != 1 or len(args.workpiece_id) > 128
            or any(character in args.workpiece_id for character in ("/", "\\", "\r", "\n", "\0"))):
        parser.error("--workpiece-id must be a single safe output identifier")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    bundle = Path(__file__).resolve().parent
    startup_error = None
    try:
        model, configuration, manifest = load_bundle(bundle)
    except Exception as exc:
        startup_error = dict(error_code="RUNTIME_BUNDLE_LOAD_FAILED", exception=type(exc).__name__, reason=str(exc))
    receipts = []
    for number, path in enumerate(args.input, 1):
        identifier = args.workpiece_id or "case_%04d" % number
        directory = args.output/("case_%04d" % number)
        directory.mkdir()
        started = time.monotonic()
        try:
            if startup_error:
                raise ValueError(startup_error["reason"])
            source = RawPointSource(path, configuration)
            with torch.no_grad():
                output = model(source)
                decoded = decode_world_predictions(output)
            if decoded["input_receipt"]["original_points_consumed"] != source.fresh_read_receipt["original_points"]:
                raise ValueError("NEURAL_ENCODING_OMITTED_ORIGINAL_ROWS")
            exported = export_decoded(directory, identifier, decoded)
            atomic_json(directory/"predictions.json", decoded)
            status = ("COMPLETED_WITH_INVALID_PREDICTIONS" if exported["invalid_prediction_rows"] else
                ("SUCCESS_WITH_PUBLISHED_WELDS" if exported["published_rows"] else "SUCCESS_NO_PUBLISHED_WELDS"))
            receipt = dict(status=status, workpiece_id=identifier, input_path_metadata_only=str(path.resolve()),
                fresh_input_receipt=source.fresh_read_receipt, neural_input_receipt=decoded["input_receipt"],
                export=exported, runtime_file_manifest_schema=manifest["schema"],
                actual_optimizer_updates=0, GT_or_STEP_reads=0, formal_attribute_accuracy_measured=False,
                model_acceptance_pass=False, full_contract_pass=False)
            del source, output, decoded
        except Exception as exc:
            write_csv(directory/"welds.csv", [])
            receipt = dict(status="FAILED", workpiece_id=identifier, input_path_metadata_only=str(path.resolve()),
                error_code="RUNTIME_BUNDLE_LOAD_FAILED" if startup_error else "INPUT_OR_INFERENCE_FAILED",
                exception=type(exc).__name__, reason=str(exc), all_GT_must_count_FN_on_evaluation=True,
                actual_optimizer_updates=0, full_contract_pass=False)
        receipt.update(utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
            original_input_read_and_export_included_in_timing=True)
        atomic_json(directory/"receipt.json", receipt)
        receipts.append(receipt)
        print("FRESH_PCD_RUNTIME_ITEM", number, receipt["status"], round(receipt["elapsed_seconds"], 3), flush=True)
    failures = sum(item["status"] == "FAILED" for item in receipts)
    invalid = sum(item.get("export", {}).get("invalid_prediction_rows", 0) for item in receipts)
    atomic_json(args.output/"batch_receipt.json", dict(utc=datetime.now(timezone.utc).isoformat(),
        requested_inputs=len(args.input), completed_input_receipts=len(receipts), failed_inputs=failures,
        invalid_prediction_rows=invalid, startup_error=startup_error, per_input=receipts,
        GT_or_STEP_reads=0, actual_optimizer_updates=0, full_contract_pass=False,
        current_runtime_is_final_contract_delivery=False))
    return 2 if failures else (3 if invalid else 0)


if __name__ == "__main__":
    sys.exit(main())
