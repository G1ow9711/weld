"""Opt-in raw AllFaces PCD -> frozen175 -> 195/197/199 -> 193 H-line entry.

This module prepares a runtime interface, not a trained model or a default
predictor. An UNTRAINED registry fails before importing torch, loading a
checkpoint, or opening a PCD. Runtime accepts no truth, STEP, prepared feature
cache, workpiece name, or source-specific coordinate table.
"""
import argparse
import builtins
from contextlib import contextmanager
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import warnings


VERSION = "raw_boundary_incidence_pcd_206_v1"
ROOT = Path(__file__).resolve().parents[1]
REGISTRY_SCHEMA = "frozen_raw_incidence_runtime_registry_206_v1"
RELEASE_SCHEMA = "frozen_raw_incidence_runtime_release_206_v1"
FROZEN175_SHA = "3869ca383cc5edd20a03d318b465f7a5b7a83e604c71df89fc457c7298f341d0"
DEFAULT_REGISTRY = ROOT / "weld_pipeline/reports/raw_incidence_entry_206_20261002/standby_registry.json"
RUNTIME_MODULES = (
    "training.dataio", "weld_pipeline.forest_event_centers",
    "weld_pipeline.bounded_event_phase_v2", "weld_pipeline.canonical_horizontal_patch_heatmaps",
    "weld_pipeline.neural_bounded_events", "weld_pipeline.boundary_query_graph_195",
    "weld_pipeline.boundary_incidence_net_197", "weld_pipeline.structured_graph_solver_v1",
)


class RuntimeRefusal(RuntimeError):
    """A registration/precondition refusal; never silently replaces weights."""


class ForbiddenRuntimeRead(RuntimeError):
    pass


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")


def save_npz(np, path, **arrays):
    # Supplying a write-only stream avoids NumPy/zipfile reopening a filename
    # with w+b, which the closed runtime read guard must otherwise reject.
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def project_file(value):
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        raise RuntimeRefusal("registered source/checkpoint must remain inside project")
    return path


def registration(config_path, release_path=None, execute=False):
    """Check one immutable config/weight pair; no case-name routing exists."""
    config_path = project_file(config_path)
    config = json.loads(config_path.read_text("utf8"))
    if config.get("schema") != REGISTRY_SCHEMA or config.get("entry_version") != VERSION:
        raise RuntimeRefusal("UNKNOWN_REGISTRY_SCHEMA")
    if config.get("status") != "TRAINED_FROZEN" or config.get("weights197") is None:
        raise RuntimeRefusal("UNTRAINED: no frozen trained197 checkpoint registered; no PCD/model read")
    expected_keys = {"schema", "entry_version", "status", "scope", "weights175", "weights197",
                     "source_sha256", "runtime", "training_config_sha256"}
    if set(config) != expected_keys or config["scope"] != "Hline_only":
        raise RuntimeRefusal("UNKNOWN_CONFIG_FIELDS_OR_SCOPE")
    if set(config["weights175"]) != {"path", "sha256"} or config["weights175"]["sha256"] != FROZEN175_SHA:
        raise RuntimeRefusal("frozen175 registration differs")
    if set(config["weights197"]) != {"path", "sha256", "actual_optimizer_steps"}:
        raise RuntimeRefusal("trained197 exact path/SHA/actual steps required")
    steps = config["weights197"]["actual_optimizer_steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise RuntimeRefusal("UNTRAINED: trained197 actual_optimizer_steps must be positive")
    allowed_runtime = {"device", "threads", "raw_chunk_size", "relation_chunk_size", "solver_options"}
    runtime = config["runtime"]
    if set(runtime) != allowed_runtime or runtime["device"] != "cpu" or runtime["threads"] != 2:
        raise RuntimeRefusal("this206 registry requires CPU2 runtime")
    for key in ("raw_chunk_size", "relation_chunk_size"):
        if isinstance(runtime[key], bool) or not isinstance(runtime[key], int) or runtime[key] <= 0:
            raise RuntimeRefusal("positive integer runtime chunk sizes required")
    if set(runtime["solver_options"]) != {"time_limit_seconds", "mip_rel_gap", "max_variables", "certificate_tolerance"}:
        raise RuntimeRefusal("unknown solver option; registered exact options required")
    if runtime["solver_options"]["mip_rel_gap"] != 0.:
        raise RuntimeRefusal("OPTIMAL zero-gap certificate required; nonoptimal remains failure")
    sources = config["source_sha256"]
    if not isinstance(sources, dict) or not sources:
        raise RuntimeRefusal("full project dependency registration required")
    this_path = Path(__file__).resolve()
    registered = {project_file(path): digest for path, digest in sources.items()}
    required = {this_path, *(ROOT / (name.replace(".", "/") + ".py") for name in RUNTIME_MODULES)}
    if not required.issubset(registered):
        raise RuntimeRefusal("entry/provider/runtime source omitted from whitelist")
    for path, expected in registered.items():
        if path.suffix != ".py" or file_sha256(path) != expected:
            raise RuntimeRefusal("registered source SHA mismatch: " + str(path))
    for key in ("weights175", "weights197"):
        path = project_file(config[key]["path"])
        if path.suffix != ".pt" or file_sha256(path) != config[key]["sha256"]:
            raise RuntimeRefusal("registered checkpoint SHA mismatch: " + key)
    if execute:
        if release_path is None:
            raise RuntimeRefusal("NO_RUNTIME_RELEASE: exact trained config has not been released")
        release = json.loads(project_file(release_path).read_text("utf8"))
        if set(release) != {"schema", "authorized", "released_by", "config_sha256"} or (
                release["schema"] != RELEASE_SCHEMA or release["authorized"] is not True or
                release["released_by"] != "root" or release["config_sha256"] != file_sha256(config_path)):
            raise RuntimeRefusal("runtime release does not match exact registered config")
    return config, registered


def prepare_environment():
    temporary = ROOT / ".codex-local/tmp"
    cache = ROOT / ".codex-local/cache/pycache"
    temporary.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.update(TEMP=str(temporary), TMP=str(temporary), PYTHONPYCACHEPREFIX=str(cache),
                      OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
                      CUDA_VISIBLE_DEVICES="")
    sys.pycache_prefix = str(cache)


def runtime_api(registered):
    """Preload the same source functions used by181, before runtime read guard."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    api = {name: importlib.import_module(name) for name in RUNTIME_MODULES}
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # Check actual locally loaded modules, including imports inside frozen code.
    loaded = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix == ".py" and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / ".venv"):
            if path not in registered:
                raise RuntimeRefusal("unregistered local dependency: " + str(path))
            loaded[str(path.relative_to(ROOT))] = file_sha256(path)
    api.update(np=np, torch=torch, loaded_project_source_sha256=loaded)
    return api


@contextmanager
def runtime_read_guard(pcd_path):
    """Closed Python path-read whitelist during inference: exactly one raw PCD.

    Weights/config/source imports are checked and completed before entry. This
    is an in-process Python I/O guard, not an OS sandbox or a proof against
    arbitrary native-extension file I/O. No prepared descriptor is accepted.
    """
    pcd_path = Path(pcd_path).resolve()
    originals = (builtins.open, io.open, Path.open, os.open)
    ledger = dict(allowed_path=str(pcd_path), reads=[], rejected_reads=[])

    def check(path):
        if not isinstance(path, (str, bytes, os.PathLike)):
            raise ForbiddenRuntimeRead("unregistered file descriptor during raw runtime")
        actual = Path(os.fsdecode(path)).resolve()
        if actual != pcd_path:
            ledger["rejected_reads"].append(str(actual))
            raise ForbiddenRuntimeRead("Forbidden raw runtime read: " + str(actual))
        ledger["reads"].append(str(actual))

    def guard(opener):
        def checked(path, mode="r", *args, **kwargs):
            if "r" in str(mode) or "+" in str(mode):
                check(path)
            return opener(path, mode, *args, **kwargs)
        return checked

    def os_guard(path, flags, *args, **kwargs):
        if (flags & getattr(os, "O_ACCMODE", 3)) != os.O_WRONLY:
            check(path)
        return originals[3](path, flags, *args, **kwargs)

    builtins.open, io.open, Path.open = (guard(opener) for opener in originals[:3])
    os.open = os_guard
    try:
        yield ledger
    finally:
        builtins.open, io.open, Path.open, os.open = originals


def install_highs_cpu2(solver):
    """Thread wrapper only;199 mathematical source/graphs/options unchanged."""
    original = solver.milp
    def cpu2(*args, **kwargs):
        options = dict(kwargs.get("options") or {})
        options.update(threads=2, parallel=False)
        kwargs["options"] = options
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected.*")
            return original(*args, **kwargs)
    solver.milp = cpu2
    return original


def canonical_inputs(sequences, data, fields, api):
    """Every generated H/nonarc runtime entry survives; IDs are ordinals only."""
    np = api["np"]
    canonical_frame = api["weld_pipeline.canonical_horizontal_patch_heatmaps"].canonical_frame
    bounded_features = api["weld_pipeline.bounded_event_phase_v2"].bounded_features
    path_type = api["weld_pipeline.boundary_query_graph_195"].CanonicalPathInput
    inputs, ledger, seen = [], [], set()
    if len(fields) != len(sequences):
        raise ValueError("raw field count differs from full candidate count")
    entry_map = {}
    for entry in data["entries"]:
        index = int(entry["sequence"])
        if index in entry_map:
            raise ValueError("duplicate runtime entry for generated candidate")
        entry_map[index] = entry
    for index, sequence in enumerate(sequences):
        row = dict(path_id=index, type=sequence["type"], curve_kind=sequence.get("curve_kind"),
                   query_count=len(sequence["xyz"]), sample_step_mm=float(sequence["sample_step_mm"]))
        ledger.append(row)
        if sequence["type"] != "horizontal" or sequence.get("curve_kind") == "arc":
            row["status"] = "OUT_OF_SCOPE_RETAINED_V_OR_ARC"
            continue
        if index not in entry_map:
            row["status"] = "FAILED_MISSING_RUNTIME_ENTRY"
            raise ValueError("H-line candidate missing runtime entry: " + str(index))
        entry = entry_map[index]
        canonical, _, transform = canonical_frame(sequence)
        if transform != entry["transform"] or entry["type"] != "horizontal":
            row["status"] = "FAILED_CANONICAL_TRANSFORM"
            raise ValueError("pure-PCD canonical transform differs from feature runtime entry")
        xyz = np.asarray(canonical["xyz"])
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0 or not np.isfinite(xyz).all():
            row["status"] = "FAILED_CANONICAL_GEOMETRY"
            raise ValueError("finite nonempty original canonical XYZ required")
        raw = np.asarray(fields[index])
        if len(raw) != len(xyz):
            row["status"] = "FAILED_RAW_QUERY_COUNT"
            raise ValueError("175 raw rows differ from canonical query rows")
        # Closure values bind the actual181 functions and this exact entry.
        def provider(indices, sequence=canonical, runtime_entry=entry):
            return bounded_features(sequence, data, runtime_entry, indices)
        path = path_type(index, xyz[0], canonical["direction"], canonical["sample_step_mm"], raw, provider)
        inputs.append(path); seen.add(index)
        row.update(status="HLINE_CANONICAL_RUNTIME_INPUT", transform=transform,
                   origin_mm=list(path.origin_mm), direction=list(path.direction))
    expected = {i for i, seq in enumerate(sequences) if seq["type"] == "horizontal" and seq.get("curve_kind") != "arc"}
    if seen != expected:
        raise ValueError("NOT_VERIFIED: generated H-line path coverage incomplete")
    return inputs, ledger


def write_path_trace(path, output, api):
    np = api["np"]; graph_module = api["weld_pipeline.boundary_query_graph_195"]
    q, centres, support, admitted, flags = graph_module.frozen_query_fields(path.raw, path.step_mm)
    span = q[-1] if len(q) else 0.
    span_lower = centres >= -graph_module.RESOLUTION_MM
    span_upper = centres <= span + graph_module.RESOLUTION_MM
    if not np.array_equal(flags["span"], span_lower & span_upper):
        raise ValueError("independent span trace differs from frozen195 gate")
    save_npz(np, output, query_index=np.arange(len(q), dtype=np.int64), query_position_mm=q,
                        raw7=path.raw, centre_mm=centres, support=support, finite=np.isfinite(path.raw),
                        support_ok=flags["support"], radius_ok=flags["radius"],
                        span_lower_ok=span_lower, span_upper_ok=span_upper, admitted=admitted)


def path_views(graph, graph_type):
    """All contexts including zero-node paths; no cache/labels or truncation."""
    for path_id, indices in sorted(graph.paths.items()):
        yield path_id, graph_type(tuple(graph.nodes[int(i)] for i in indices),
                                  (graph.path_contexts[path_id],),
                                  {path_id: graph.admission_receipts[path_id]})


def result_receipt(result):
    return dict(status=result.status, optimal_proved=result.optimal_proved,
                complete_graph_solution=result.complete_graph_solution, base_score=result.base_score,
                objective_max=result.objective_max, upper_bound_max=result.upper_bound_max,
                mip_gap=result.mip_gap, counts=result.counts, component_receipts=result.component_receipts,
                message=result.message, graph_fingerprint=result.graph_fingerprint)


def export_chains(graph, result, export_selected):
    """Failed geometry retains every selected child, node, originalq/XYZ/score."""
    valid, invalid, consumed = [], [], []
    selected = [tuple(map(int, pair)) for pair in result.selection.selected_child_pairs.tolist()]
    for ordinal, chain in enumerate(result.selection.chains):
        chain = tuple(map(int, chain)); pairs = list(zip(chain[:-1], chain[1:])); consumed.extend(pairs)
        try:
            welds, _, _ = export_selected(graph, pairs)
            valid.extend(welds)
        except Exception as error:
            invalid.append(dict(chain_index=ordinal, node_ordinals=list(chain), child_pairs=[list(pair) for pair in pairs],
                invalid_child_prediction_count=len(pairs), runtime_ids=[list(graph.nodes[i].runtime_id) for i in chain],
                original_query_position_mm=[graph.nodes[i].query_position_mm for i in chain],
                raw_predicted_q_mm=[graph.nodes[i].predicted_position_mm for i in chain],
                raw_predicted_xyz_mm=[list(graph.nodes[i].predicted_xyz_mm) for i in chain],
                full_path_base_structure_score=result.base_score, error_type=type(error).__name__, error=str(error)))
    valid_count = sum(len(weld["segments"]) for weld in valid)
    invalid_count = sum(row["invalid_child_prediction_count"] for row in invalid)
    if len(set(selected)) != len(selected) or sorted(selected) != sorted(consumed) or len(selected) != valid_count + invalid_count:
        raise ValueError("NOT_VERIFIED: selected child valid/invalid ledger is incomplete")
    return valid, invalid


def load_models(config, api):
    np, torch = api["np"], api["torch"]
    saved175 = torch.load(project_file(config["weights175"]["path"]), map_location="cpu", weights_only=False)
    mean, std = np.asarray(saved175["mean"]), np.asarray(saved175["std"])
    if mean.dtype != np.float32 or std.dtype != np.float32 or mean.shape != (425,) or std.shape != (425,) or (
            not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0)):
        raise RuntimeRefusal("exact frozen175 float32[425] normalizer required")
    model175 = api["weld_pipeline.neural_bounded_events"].BoundedEventNet().cpu().eval()
    model175.load_state_dict(saved175["state_dict"], strict=True)
    saved197 = torch.load(project_file(config["weights197"]["path"]), map_location="cpu", weights_only=False)
    actual_steps = saved197.get("actual_steps")
    if isinstance(actual_steps, bool) or not isinstance(actual_steps, int) or actual_steps != config["weights197"]["actual_optimizer_steps"]:
        raise RuntimeRefusal("trained197 checkpoint actual_steps differs from registration")
    training_config = saved197.get("config")
    if not isinstance(training_config, dict) or training_config.get("normalizer_sha256") != FROZEN175_SHA:
        raise RuntimeRefusal("trained197 config lacks exact frozen175 normalizer provenance")
    training_cores = training_config.get("cores_sha256", {})
    runtime_cores = {project_file(path): digest for path, digest in config["source_sha256"].items()}
    for name in ("boundary_query_graph_195.py", "boundary_incidence_net_197.py", "structured_graph_solver_v1.py"):
        key = "weld_pipeline/" + name
        expected = runtime_cores.get((ROOT / key).resolve())
        if not isinstance(training_cores, dict) or key not in training_cores or not isinstance(expected, str) or training_cores[key] != expected:
            raise RuntimeRefusal("trained197 training/runtime frozen core differs: " + key)
    training_sha = hashlib.sha256(json.dumps(training_config, sort_keys=True,
                                           separators=(",", ":")).encode("utf8")).hexdigest()
    if training_sha != config["training_config_sha256"]:
        raise RuntimeRefusal("trained197 training config digest differs")
    model197 = api["weld_pipeline.boundary_incidence_net_197"].BoundaryIncidenceNet(mean, std).cpu().eval()
    model197.load_state_dict(saved197["state_dict"], strict=True)
    if not np.array_equal(model197.feature_mean.detach().numpy(), mean) or not np.array_equal(model197.feature_std.detach().numpy(), std):
        raise RuntimeRefusal("trained197 normalizer does not equal registered frozen175")
    for model in (model175, model197):
        for parameter in model.parameters():
            if not bool(torch.isfinite(parameter).all()):
                raise RuntimeRefusal("registered model contains nonfinite weights")
            parameter.requires_grad_(False)
    return model175, model197, mean, std


def run_raw(pcd_path, config_path, release_path, output):
    config, registered = registration(config_path, release_path, execute=True)
    pcd = Path(pcd_path).resolve(); output = Path(output).resolve()
    if pcd.suffix.lower() != ".pcd" or not pcd.is_file():
        raise RuntimeRefusal("one raw ASCII AllFaces XYZ PCD required")
    if not output.is_relative_to(ROOT) or output.exists():
        raise RuntimeRefusal("new project-local output directory required; preserve existing evidence")
    prepare_environment(); api = runtime_api(registered)
    model175, model197, mean, std = load_models(config, api)
    original_sha = file_sha256(pcd)
    output.mkdir(parents=True)
    np, torch = api["np"], api["torch"]
    module195 = api["weld_pipeline.boundary_query_graph_195"]
    module197 = api["weld_pipeline.boundary_incidence_net_197"]
    solver = api["weld_pipeline.structured_graph_solver_v1"]
    original_milp = install_highs_cpu2(solver)
    predictions, invalid, failures, path_receipts, candidates, guard_ledger = [], [], [], [], [], {}
    processed = 0; scored_paths = 0; expected_paths = None; graph = None
    try:
        with runtime_read_guard(pcd) as guard_ledger:
            xyz_mm = api["training.dataio"].read_xyz_pcd(pcd) * 1000.
            sequences, data = api["weld_pipeline.forest_event_centers"].build_inputs(xyz_mm)
            # Store every generated candidate before any canonical conversion.
            lengths = np.asarray([len(seq["xyz"]) for seq in sequences], np.int64)
            save_npz(np, output / "all_generated_candidate_geometry.npz", lengths=lengths,
                                xyz=np.concatenate([np.asarray(seq["xyz"]) for seq in sequences]) if len(sequences) else np.empty((0,3)),
                                direction=np.asarray([seq["direction"] for seq in sequences]).reshape(-1,3))
            candidates = [dict(path_id=i, type=seq["type"], curve_kind=seq.get("curve_kind"),
                               query_count=len(seq["xyz"]), sample_step_mm=float(seq["sample_step_mm"])) for i,seq in enumerate(sequences)]
            dump(output / "all_generated_candidates.json", candidates)
            fields = api["weld_pipeline.neural_bounded_events"].predict_fields(
                model175, data, sequences, mean, std, config["runtime"]["raw_chunk_size"])
            save_npz(np, output / "all_frozen175_raw7.npz", lengths=lengths,
                                raw=np.concatenate(fields) if fields else np.empty((0,7),np.float32))
            inputs, candidates = canonical_inputs(sequences, data, fields, api)
            dump(output / "all_generated_candidates.json", candidates)
            for path in inputs:
                write_path_trace(path, output / ("path_" + str(path.path_id) + "_admission.npz"), api)
            graph = module195.build_runtime_graph(inputs)
            expected_paths = len(inputs)
            dump(output / "full_graph_hashes.json", graph.complete_hashes())
            dump(output / "admission_summary.json", graph.admission_receipts)
            save_npz(np, output / "all_runtime_nodes.npz", **graph.node_arrays())
            options = solver.SolveOptions(chunk_size=config["runtime"]["relation_chunk_size"], **config["runtime"]["solver_options"])
            for path_id, local in path_views(graph, module195.QueryGraph):
                result = None
                try:
                    with torch.inference_mode():
                        node_inputs, relations = module197.query_graph_195_inputs(local)
                        scores = model197(node_inputs, relations, chunk_size=config["runtime"]["relation_chunk_size"])
                        detached = module197.solver_199_scores(scores)
                        save_npz(np, output / ("path_" + str(path_id) + "_scores.npz"),
                                            u=detached.node_scores, a=detached.child_scores, d=detached.duplicate_scores)
                        scored_paths += 1
                        result = solver.solve_runtime_graph(local, detached, options)
                        module197.certified_selection_199(result, scores)
                    valid, broken = export_chains(local, result, module195.export_selected)
                    for weld in valid:
                        weld["id"] = "weld_" + str(len(predictions)).zfill(6)
                        predictions.append(weld)
                    invalid.extend(dict(path_id=path_id, **row) for row in broken)
                    path_receipts.append(dict(path_id=path_id, solve=result_receipt(result),
                        score_coverage=scores.coverage, full_graph_hashes=local.complete_hashes(),
                        selected_nodes=np.flatnonzero(result.selection.z).tolist(),
                        selected_child_pairs=result.selection.selected_child_pairs.tolist(), chains=result.selection.chains,
                        geometry_status="FAILED_GEOMETRY_EXPORT" if broken else "PASS"))
                    processed += 1
                except Exception as error:
                    failures.append(dict(path_id=path_id, error_type=type(error).__name__, error=str(error),
                        solve=None if result is None else result_receipt(result), full_path_retained=True,
                        full_invalid_prediction_count="NOT_VERIFIED"))
                dump(output / "runtime_progress.json", dict(completed_paths=processed, expected_paths=expected_paths,
                                                           failed_paths=len(failures), stage="runtime199"))
    except Exception as error:
        failures.append(dict(stage="raw_features_or_canonical_graph", error_type=type(error).__name__,
                             error=str(error), returned_generated_candidates_retained=bool(candidates),
                             candidate_generation_status="RETURNED" if candidates else "FAILED_NOT_RETURNED",
                             full_invalid_prediction_count="NOT_VERIFIED"))
    finally:
        solver.milp = original_milp
    changed = {str(path.relative_to(ROOT)): file_sha256(path) for path, digest in registered.items() if file_sha256(path) != digest}
    input_unchanged = file_sha256(pcd) == original_sha
    weight_hashes_unchanged = all(file_sha256(project_file(config[key]["path"])) == config[key]["sha256"] for key in ("weights175", "weights197"))
    if changed or not input_unchanged or not weight_hashes_unchanged:
        failures.append(dict(stage="provenance", changed_sources=changed, input_unchanged=input_unchanged,
                             weight_hashes_unchanged=weight_hashes_unchanged))
    valid_count = sum(len(weld["segments"]) for weld in predictions)
    invalid_count = sum(row["invalid_child_prediction_count"] for row in invalid)
    complete = not failures and expected_paths is not None and processed == expected_paths
    receipt = dict(schema=VERSION, status="NOT_VERIFIED" if not complete else ("FAILED_GEOMETRY_EXPORT" if invalid else "COMPLETE_RAW_HLINE_RUNTIME"),
        scope="Hline_only", input_pcd=str(pcd), input_sha256=original_sha, coordinates="raw XYZ metres *1000 to mm",
        config_sha256=file_sha256(config_path), weights175=config["weights175"], weights197=config["weights197"],
        loaded_project_source_sha256=api["loaded_project_source_sha256"], input_unchanged=input_unchanged,
        weight_hashes_unchanged=weight_hashes_unchanged, changed_sources=changed,
        generated_candidates=len(candidates), expected_Hline_paths=expected_paths, completed_paths=processed,
        path_receipts=path_receipts, runtime_read_guard=guard_ledger, failures=failures,
        valid_export_child_count=valid_count, invalid_child_prediction_count=invalid_count,
        full_prediction_child_count=valid_count+invalid_count if complete else "NOT_VERIFIED",
        runtime_geometry_complete=complete and not invalid,
        adoption_candidate=False, ground_truth_parameters_used=False,
        runtime_STEP_used=False, runtime_prepared_data_cache_used=False, case_name_or_coordinate_lookup_used=False,
        IDs_are_indices_only=True, all_admitted_candidates_and_all_relations_retained=graph is not None,
        scores_are_trained197_neural_predictions=scored_paths > 0, neural_scored_paths=scored_paths,
        optimizer_steps=0, device="cpu", threads=2,
        highs_threads_engineering_wrapper=True, default_predictor_replaced=False,
        full16_raw_rebuild_verified=False, negative_controls_full16_verified=False,
        accuracy="NOT_MEASURED_NO_GT", E="UNKNOWN_INCOMPLETE", V_and_arc="OUT_OF_SCOPE_INCOMPLETE", full_contract_pass=False)
    dump(output / "predictions.json", predictions); dump(output / "invalid_prediction_ledger.json", invalid)
    dump(output / "all_generated_candidates.json", candidates)
    receipt["output_sha256"] = {path.name:file_sha256(path) for path in sorted(output.iterdir()) if path.is_file()}
    dump(output / "receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--runtime-release")
    parser.add_argument("--pcd", help="one raw ASCII AllFaces XYZ PCD, metres")
    parser.add_argument("--output", help="new project-local output directory")
    parser.add_argument("--check-registration", action="store_true")
    parser.add_argument("--run", action="store_true", help="opt-in only; frozen trained197 registration required")
    args = parser.parse_args(argv)
    try:
        if args.run == args.check_registration:
            raise RuntimeRefusal("NO_RUN: choose --check-registration or explicit --run")
        if args.check_registration:
            registration(args.config)
            print("REGISTERED_TRAINED_WEIGHTS_NO_INFERENCE")
            return 0
        # Registration precedes PCD/output validation, runtime imports/model I/O.
        registration(args.config, args.runtime_release, execute=True)
        if not args.pcd or not args.output:
            raise RuntimeRefusal("--pcd and --output required for released runtime")
        result = run_raw(args.pcd, args.config, args.runtime_release, args.output)
        print(result["status"], "valid_children", result["valid_export_child_count"],
              "invalid_children", result["invalid_child_prediction_count"])
        return 0 if result["status"] == "COMPLETE_RAW_HLINE_RUNTIME" else 1
    except RuntimeRefusal as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
