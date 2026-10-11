"""Fixed source-batch C runner; preparation/preflight never optimizer.step.

Actual optimization requires a separate root release bound to the exact config.
Frozen175 supplies normalization metadata only. Runtime decoding has no labels.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import random
import time
import threading
import warnings

import numpy as np
import torch

from weld_pipeline.boundary_incidence_net_197 import (
    BoundaryIncidenceNet, query_graph_195_inputs, solver_199_scores, certified_selection_199,
    latent_structured_hinge_from_optima,differentiable_structure_score)
from weld_pipeline.boundary_query_graph_195 import export_selected, QueryGraph
from weld_pipeline.structured_graph_artifacts_200 import (
    file_sha256, load_runtime_graph, load_training_truth, complete_path_views)
from weld_pipeline.structured_graph_solver_v1 import (
    GraphScores, GraphTruth, TruthEvent, TruthChild, SolveOptions,
    solve_runtime_graph, solve_best_truth_consistent_graph, solve_loss_augmented_graph)
import weld_pipeline.structured_graph_solver_v1 as solver199

ROOT = Path(__file__).resolve().parents[1]
VERSION = "boundary_incidence_source_batch_C_200_v1"
SEED = 20261002
NORMALIZER_SHA = "3869ca383cc5edd20a03d318b465f7a5b7a83e604c71df89fc457c7298f341d0"
FROZEN_CORES = {
    "weld_pipeline/boundary_query_graph_195.py": "9cee6158b0b44852cfe0bb42e8c949ee07f0f2ead9556098e259f05930c24187",
    "weld_pipeline/boundary_incidence_net_197.py": "72586ce7cd104c23fc427235c4fff766002a6b5875a31cc03b538e7b95bbedda",
    "weld_pipeline/structured_graph_solver_v1.py": "d0d0537aaac722f61a326f7055975485f449e372550c5905e6a2c6afea1c1149",
}

class SolverCertificateError(RuntimeError):
    def __init__(self,message,details):
        super().__init__(message);self.details=details


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), "utf8")


def seed_runtime(device):
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":
        raise ValueError("launch must set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(SEED)
    return dict(seed=SEED, deterministic=True, CUBLAS_WORKSPACE_CONFIG=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                OMP_NUM_THREADS=os.environ["OMP_NUM_THREADS"], OPENBLAS_NUM_THREADS=os.environ["OPENBLAS_NUM_THREADS"],
                matmul_allow_tf32=False,cudnn_allow_tf32=False,cudnn_benchmark=False,cudnn_deterministic=True)


def install_highs_cpu2():
    """Engineering wrapper only: unchanged199 c/bounds/constraints/full graph."""
    if getattr(solver199.milp, "_source200_cpu2", False):
        return
    original = solver199.milp
    def cpu2(*args, **kwargs):
        options = dict(kwargs.get("options") or {})
        options.update(threads=2, parallel=False)
        kwargs["options"] = options
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected.*")
            return original(*args, **kwargs)
    cpu2._source200_cpu2 = True
    solver199.milp = cpu2


def load_frozen175_normalizer(path):
    if file_sha256(path) != NORMALIZER_SHA:
        raise ValueError("frozen175 checkpoint SHA mismatch")
    state = torch.load(path, map_location="cpu")
    mean, std = np.asarray(state["mean"]), np.asarray(state["std"])
    if mean.dtype != np.float32 or std.dtype != np.float32 or mean.shape != (425,) or std.shape != (425,):
        raise ValueError("exact frozen175 float32[425] normalizer required")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("frozen175 normalizer is invalid")
    # Never instantiate/call the frozen175 model or use its learned weights.
    mean, std = mean.copy(), std.copy()
    del state
    gc.collect()
    return mean, std


def model_sha(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def exact_latent_loss(model, graph, truth, chunk_size, solver_options, progress=None, supplied_scores=None):
    progress=progress or (lambda stage, **values: None)
    progress("FORWARD_START" if supplied_scores is None else "REUSE_COMPLETE_FORWARD_SCORES")
    inputs, relations = query_graph_195_inputs(graph)
    scores = supplied_scores if supplied_scores is not None else model(inputs, relations, chunk_size=chunk_size)
    progress("FORWARD_COMPLETE",node_count=len(scores.u),child_count=len(scores.a),duplicate_count=len(scores.d))
    detached = solver_199_scores(scores)
    progress("GOOD_EXACT_SOLVE_START",incumbent_max=None,upper_bound_max=None,live_solver_callback_available=False)
    good = solve_best_truth_consistent_graph(graph, detached, truth, solver_options)
    progress("GOOD_EXACT_SOLVE_COMPLETE",status=good.status,incumbent_max=good.objective_max,upper_bound_max=good.upper_bound_max,mip_gap=good.mip_gap,receipt=solver_receipt(good))
    try:good_certificate=certified_selection_199(good,scores)
    except Exception as error:raise SolverCertificateError(str(error),dict(good=solver_receipt(good),bad="NOT_RUN_GOOD_CERTIFICATE_FAILED")) from error
    progress("BAD_EXACT_SOLVE_START",incumbent_max=None,upper_bound_max=None,live_solver_callback_available=False)
    bad = solve_loss_augmented_graph(graph, detached, truth, solver_options)
    progress("BAD_EXACT_SOLVE_COMPLETE",status=bad.status,incumbent_max=bad.objective_max,upper_bound_max=bad.upper_bound_max,mip_gap=bad.mip_gap,receipt=solver_receipt(bad))
    # These adapters independently reject nonoptimal, incomplete, wrong bounds,
    # gap, products, ordinals and same-sized-but-different candidate graphs.
    try:bad_certificate=certified_selection_199(bad,scores)
    except Exception as error:raise SolverCertificateError(str(error),dict(good=solver_receipt(good),bad=solver_receipt(bad))) from error
    loss = latent_structured_hinge_from_optima(scores, good_certificate, bad_certificate, bad.structural_loss)
    units = len(truth.events)+len(truth.true_children)
    return loss/max(1, units), dict(good=solver_receipt(good), bad=solver_receipt(bad),
                                  loss=float(loss.detach()), normalized_loss=float((loss/max(1, units)).detach()),
                                  truth_units=units, complete_counts=dict(nodes=len(graph.nodes), children=graph.pair_count("child"),
                                  duplicate=graph.pair_count("duplicate"), message=graph.pair_count("message")))


def solver_receipt(result):
    return dict(status=result.status, optimal_proved=result.optimal_proved,
                complete_graph_solution=result.complete_graph_solution, base_score=result.base_score,
                objective_max=result.objective_max, upper_bound_max=result.upper_bound_max, mip_gap=result.mip_gap,
                structural_loss=result.structural_loss, loss_details=result.loss_details,
                counts=result.counts, components=result.component_receipts, graph_fingerprint=result.graph_fingerprint,
                selected_nodes=np.flatnonzero(result.selection.z).tolist() if result.selection is not None else None,
                selected_children=np.flatnonzero(result.selection.y).tolist() if result.selection is not None else None,
                selected_pairs=result.selection.selected_child_pairs.tolist() if result.selection is not None else None,
                chains=result.selection.chains if result.selection is not None else None)


def gradient_state(model):
    out = {}
    nonzero = False
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise ValueError("nonfinite accumulated source gradient: "+name)
            nonzero = nonzero or bool(torch.any(parameter.grad != 0))
            out[name] = parameter.grad.detach().cpu().clone()
    return out, nonzero


def assert_core_hashes():
    for path, expected in FROZEN_CORES.items():
        if file_sha256(ROOT/path) != expected:
            raise ValueError("frozen core changed: "+path)


def prepare_config(output, artifact_directory):
    """Metadata/hash-only preparation; no model forward or optimizer creation."""
    assert_core_hashes()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    artifact_directory = Path(artifact_directory).resolve()
    complete_path = artifact_directory/"complete.json"
    complete = json.loads(complete_path.read_text("utf8"))
    if not complete["real_query_representation_hard_training_start_gate"] or complete["totals"]["GT_children"] != 1221:
        raise ValueError("195 full1221 representation hard gate required")
    sources = []
    for source, receipt_hash in complete["source_receipts"].items():
        receipt_path = artifact_directory/(source+".json")
        if file_sha256(receipt_path) != receipt_hash:
            raise ValueError("source195 receipt binding changed")
        receipt = json.loads(receipt_path.read_text("utf8"))
        graph_meta_path = artifact_directory/(source+"_candidate_graph.json")
        graph_meta = json.loads(graph_meta_path.read_text("utf8"))
        suffixes = ("_candidate_graph.json", "_all_runtime_nodes.npz", "_runtime_FG_context.npz", "_offline_labels.npz", "_offline_lineage.json")
        hashes = {source+suffix: file_sha256(artifact_directory/(source+suffix)) for suffix in suffixes}
        if hashes[source+"_candidate_graph.json"] != receipt["candidate_graph_sha256"] or hashes[source+"_offline_lineage.json"] != receipt["offline_lineage_sha256"]:
            raise ValueError("195 graph/lineage original receipt mismatch")
        # Evaluation GT is copied verbatim from the two original frozen
        # manifests, separated from runtime features and train-event labels.
        relation_manifest = ROOT/"weld_pipeline/data/joint_relation_v1"/source/"manifest.json"
        expected_inputs = complete["frozen_inputs_sha256"]
        if file_sha256(relation_manifest) != expected_inputs[str(relation_manifest)]:
            raise ValueError("original source relation manifest changed")
        relation = json.loads(relation_manifest.read_text("utf8"))
        cache_root = Path(relation["source_cache_root"])
        if not cache_root.is_absolute():
            cache_root = ROOT/cache_root
        truth_manifest = cache_root/"manifest.json"
        if file_sha256(truth_manifest) != expected_inputs[str(truth_manifest)]:
            raise ValueError("original source truth manifest changed")
        cache = json.loads(truth_manifest.read_text("utf8"))
        truth = [w for w in cache["truth"] if w["type"] == "horizontal" and all(s["kind"] == "line" for s in w["segments"])]
        denominator = sum(len(w["segments"]) for w in truth)
        if denominator != receipt["counts"]["GT_children"]:
            raise ValueError("full source GTchild denominator mismatch")
        truth_path = output/(source+"_evaluation_truth.json")
        dump(truth_path, truth)
        sources.append(dict(source=source, artifact_directory=str(artifact_directory), expected_sha256=hashes,
            path_count=len(graph_meta["paths"]), truth_path=str(truth_path.resolve()), truth_sha256=file_sha256(truth_path),
            denominator=denominator, original_truth_manifest_sha256=file_sha256(truth_manifest),
            original_relation_manifest_sha256=file_sha256(relation_manifest)))
    if len(sources) != 16 or sum(s["denominator"] for s in sources) != 1221 or sum(s["path_count"] for s in sources) != 2011:
        raise ValueError("fixed full16/G1221/2011 complete paths required")
    config = dict(version=VERSION, training_status="PENDING_ROOT_RELEASE", actual_optimizer_steps=0,
        cores_sha256=FROZEN_CORES, runner_sha256=file_sha256(__file__),
        loader_sha256=file_sha256(ROOT/"weld_pipeline/structured_graph_artifacts_200.py"),
        normalizer_path=str(ROOT/"weld_pipeline/runs/deep_bounded_events_175/final.pt"), normalizer_sha256=NORMALIZER_SHA,
        source_batch_schedule="C", seed=SEED, learning_rate=3e-4, weight_decay=1e-4,
        maximum_complete_epochs=80, maximum_actual_updates=12000,
        maximum_source_slots=16*80, maximum_possible_actual_updates=16*80,
        fit_every_actual_updates=1000, fit_final=True, sources=sources,
        source_path_denominator=sum(s["path_count"] for s in sources), empty_paths_count=1659,
        nonempty_paths_count=352, path_loss="latent_hinge/max(1,GTchild+GTevent)/P_source",
        strict_deterministic=True, CUBLAS_WORKSPACE_CONFIG=":4096:8",matmul_allow_tf32=False,cudnn_allow_tf32=False,
        cudnn_benchmark=False,cudnn_deterministic=True, highs_threads=2, highs_parallel=False,
        chunk_size=4096, solver_options=dict(time_limit_seconds=None,mip_rel_gap=0.,chunk_size=65536,max_variables=None),
        full_denominator=1221, full_sources=16, full_contract_pass=False,
        invalid_prediction_policy="fullFP=official_valid_export_FP+invalid_selected_child_count; failed chains kept inledger",
        gates_required=["195_full_oracle", "196_full_review", "197_numerics", "199_solver", "200_sourcegradient", "201_review", "203_CUDA", "200_real_maxpath"])
    dump(output/"config.json", config)
    return config


def preflight_real_path(config, output, source="161-LB18A", path_id=86, device="cuda"):
    """Authorized new197 initialization engineering check; no optimizer.step."""
    assert_core_hashes(); install_highs_cpu2()
    environment = seed_runtime(device)
    spec = next(s for s in config["sources"] if s["source"] == source)
    graph = load_runtime_graph(spec["artifact_directory"], source, spec["expected_sha256"])
    truth = load_training_truth(spec["artifact_directory"], source, graph, spec["expected_sha256"])
    views = complete_path_views(graph, truth)
    _, path_graph, path_truth, original_ids = next(v for v in views if v[0] == path_id)
    mean, std = load_frozen175_normalizer(config["normalizer_path"])
    model = BoundaryIncidenceNet(mean, std).to(device).train()
    before = model_sha(model)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    runner_before=file_sha256(__file__);loader_before=file_sha256(ROOT/"weld_pipeline/structured_graph_artifacts_200.py")
    phase={"stage":"INITIALIZED","started_wall_time_UTC":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"optimizer_steps":0}
    phase_lock=threading.Lock();stop_heartbeat=threading.Event()
    progress_path=Path(output)/"real_maxpath_progress.json"
    def progress(stage,**values):
        with phase_lock:
            phase.update(stage=stage,stage_started_elapsed_seconds=time.monotonic()-started,**values)
            dump(progress_path,dict(phase,elapsed_seconds=time.monotonic()-started))
        print("200_MAXPATH_STAGE",stage,json.dumps(values,ensure_ascii=False),flush=True)
    def heartbeat():
        while not stop_heartbeat.wait(15):
            with phase_lock:
                dump(progress_path,dict(phase,elapsed_seconds=time.monotonic()-started,heartbeat=True))
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    receipt = dict(status="RUNNING_INITIALIZATION_ENGINEERING_PREFLIGHT", source=source, path=path_id,
                   optimizer_steps=0, frozen175_forward=False, environment=environment, actual_model_precision_claim=False)
    dump(Path(output)/"real_maxpath_preflight.json", receipt)
    try:
        progress("FORWARD_START")
        inputs,relations=query_graph_195_inputs(path_graph)
        scores=model(inputs,relations,chunk_size=config["chunk_size"])
        progress("FORWARD_COMPLETE",nodes=len(scores.u),children=len(scores.a),duplicate_pairs=len(scores.d),message_pairs=path_graph.pair_count("message"),score_dtype=str(scores.u.dtype))
        # Engineering resource evidence before a potentially long exact MILP:
        # one role-legal runtime chain, no GT labels or geometric pruning.
        s=int(path_graph.roles[path_id][0][0]);e=int(path_graph.roles[path_id][1][0])
        proxy_z=np.zeros(len(path_graph.nodes),bool);proxy_z[[s,e]]=True
        proxy_y=np.zeros(path_graph.pair_count("child"),bool);proxy_y[path_graph.child_edge_id(s,e)]=True
        progress("ENGINEERING_SCORE_BACKWARD_START",not_identity_loss=True)
        differentiable_structure_score(scores,proxy_z,proxy_y).backward(retain_graph=True)
        proxy_gradients,proxy_nonzero=gradient_state(model)
        if model_sha(model)!=before:raise ValueError("engineering backward altered weights")
        dump(Path(output)/"real_maxpath_resource.json",dict(status="PASS_COMPLETE_FORWARD_RESOURCE_BACKWARD_NO_UPDATES",
            model_precision_claim=False,actual_optimizer_steps=0,frozen175_forward=False,environment=environment,
            runner_sha256=runner_before,loader_sha256=loader_before,cores_sha256=FROZEN_CORES,
            complete_counts=dict(nodes=len(path_graph.nodes),child=path_graph.pair_count("child"),message=path_graph.pair_count("message"),duplicate=path_graph.pair_count("duplicate")),
            gradient_finite=True,gradient_nonzero=proxy_nonzero,weights_unchanged=True,
            GPU_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()) if str(device).startswith("cuda") else 0,
            GPU_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()) if str(device).startswith("cuda") else 0,
            message_pass_gradients={str(p):dict(finite=True,nonzero=any(bool(torch.any(v!=0)) for k,v in proxy_gradients.items() if k.startswith("messages."+str(p)+"."))) for p in (0,1)}))
        model.zero_grad(set_to_none=True)
        progress("ENGINEERING_SCORE_BACKWARD_COMPLETE",gradient_finite=True,identity_hinge_proof="PENDING_EXACT_GOOD_BAD")
        progress("RUNTIME_EXACT_SOLVE_START",incumbent_max=None,upper_bound_max=None,live_solver_callback_available=False)
        runtime=solve_runtime_graph(path_graph,solver_199_scores(scores),SolveOptions(**config["solver_options"]))
        progress("RUNTIME_EXACT_SOLVE_COMPLETE",status=runtime.status,incumbent_max=runtime.objective_max,upper_bound_max=runtime.upper_bound_max,mip_gap=runtime.mip_gap,receipt=solver_receipt(runtime))
        try:certified_selection_199(runtime,scores)
        except Exception as error:raise SolverCertificateError(str(error),dict(runtime=solver_receipt(runtime))) from error
        loss, details = exact_latent_loss(model, path_graph, path_truth, config["chunk_size"], SolveOptions(**config["solver_options"]),progress,scores)
        print("200_REAL_MAXPATH_EXACT_OPTIMA", details["good"]["status"], details["bad"]["status"], flush=True)
        progress("BACKWARD_START")
        loss.backward()
        gradients, nonzero = gradient_state(model)
        if model_sha(model) != before:
            raise ValueError("initialization preflight changed learned weights")
        if file_sha256(__file__)!=runner_before or file_sha256(ROOT/"weld_pipeline/structured_graph_artifacts_200.py")!=loader_before:
            raise ValueError("preflight runner/loader source changed during execution")
        progress("BACKWARD_COMPLETE",finite_gradients=True,gradient_nonzero=nonzero)
        receipt.update(status="PASS_INITIALIZATION_ENGINEERING_NO_UPDATES", details=details,runtime_solve=solver_receipt(runtime),
            finite_gradients=True, gradient_nonzero=nonzero, before_after_weights_identical=True,
            message_pass_gradients={str(p):dict(finite=True,nonzero=any(bool(torch.any(v!=0)) for k,v in gradients.items() if k.startswith("messages."+str(p)+"."))) for p in (0,1)},
            original_global_node_ordinals=original_ids.tolist(), elapsed_seconds=time.monotonic()-started,
            GPU_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()) if str(device).startswith("cuda") else 0,
            GPU_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()) if str(device).startswith("cuda") else 0,
            runner_sha256=file_sha256(__file__), loader_sha256=file_sha256(ROOT/"weld_pipeline/structured_graph_artifacts_200.py"))
    except Exception as error:
        receipt.update(status="FAILED_ENGINEERING_PREFLIGHT_NO_UPDATES", error_type=type(error).__name__, error=str(error),
                       elapsed_seconds=time.monotonic()-started, full_candidate_graph_retained=True,
                       solver_failure_details=getattr(error,"details",None))
        dump(Path(output)/"real_maxpath_preflight.json", receipt)
        stop_heartbeat.set()
        raise
    stop_heartbeat.set()
    dump(Path(output)/"real_maxpath_preflight.json", receipt)
    return receipt


def _export_individual_chains(graph, result):
    valid, invalid = [], []
    all_pairs = [tuple(pair) for pair in result.selection.selected_child_pairs.tolist()]
    consumed = set()
    for chain_index, chain in enumerate(result.selection.chains):
        pairs = list(zip(chain[:-1], chain[1:]))
        consumed.update(pairs)
        try:
            welds, _, _ = export_selected(graph, pairs)
            for weld in welds:
                valid.append(weld)
        except Exception as error:
            invalid.append(dict(chain_index=chain_index, node_ordinals=list(chain), child_pairs=[list(pair) for pair in pairs],
                invalid_selected_child_count=len(pairs), runtime_ids=[list(graph.nodes[i].runtime_id) for i in chain],
                raw_predicted_q_mm=[graph.nodes[i].predicted_position_mm for i in chain],
                raw_predicted_xyz_mm=[list(graph.nodes[i].predicted_xyz_mm) for i in chain],
                error_type=type(error).__name__, error=str(error), base_structure_score=result.base_score))
    if set(all_pairs) != consumed or len(all_pairs) != sum(len(w["segments"]) for w in valid)+sum(v["invalid_selected_child_count"] for v in invalid):
        raise ValueError("NOT_VERIFIED: valid/invalid selected-child partition is incomplete")
    return valid, invalid


def evaluate_full16(model, config, output, step):
    from weld_pipeline.contract_child_evaluation import score_children
    output = Path(output)/("evaluation_step_"+str(step))
    output.mkdir(parents=True, exist_ok=True)
    totals = dict(TP=0,FP=0,FN=0,G=0,official_valid_export_FP=0,invalid_selected_child_count=0)
    receipts = []
    was_training = model.training
    model.eval()
    for spec in config["sources"]:
        source = spec["source"]
        graph = load_runtime_graph(spec["artifact_directory"], source, spec["expected_sha256"])
        predictions, failures, solver_failures, path_receipts = [], [], [], []
        for path, local, _, _ in complete_path_views(graph):
            try:
                with torch.no_grad():
                    inputs, relations = query_graph_195_inputs(local)
                    scores = model(inputs, relations, chunk_size=config["chunk_size"])
                    result = solve_runtime_graph(local, solver_199_scores(scores), SolveOptions(**config["solver_options"]))
                    certified_selection_199(result, scores)
                valid, invalid = _export_individual_chains(local, result)
                for weld in valid:
                    weld["id"] = "weld_"+str(len(predictions)).zfill(6)
                    predictions.append(weld)
                failures.extend(dict(path=path, **error) for error in invalid)
                path_receipts.append(dict(path=path, solve=solver_receipt(result), selected_nodes=np.flatnonzero(result.selection.z).tolist(),
                    selected_child_pairs=result.selection.selected_child_pairs.tolist(), chains=result.selection.chains,
                    geometry_export_status="FAILED_GEOMETRY_EXPORT" if invalid else "PASS"))
            except Exception as error:
                solver_failures.append(dict(path=path,error_type=type(error).__name__,error=str(error), full_path_retained=True))
        if file_sha256(spec["truth_path"]) != spec["truth_sha256"]:
            raise ValueError("evaluation truth changed")
        truth = json.loads(Path(spec["truth_path"]).read_text("utf8"))
        score = score_children(predictions, truth)
        invalid_count = sum(v["invalid_selected_child_count"] for v in failures)
        # Official score fields are normalized below after source inspection;
        # fullFP always includes failed geometry children, no deleted G.
        tp, fp, fn = score["valid_children"], score["false_predictions"], score["missed_children"]
        if tp+fn != spec["denominator"]:
            raise ValueError("official scorer omitted source G")
        status = "NOT_VERIFIED" if solver_failures else ("FAILED_GEOMETRY_EXPORT" if failures else "COMPLETE_RUNTIME_EVALUATION")
        case = dict(source=source, scientific_status=status, G=spec["denominator"], official_valid_export_score=score,
                    TP=tp, FN=fn, official_valid_export_FP=fp, invalid_selected_child_count=invalid_count, full_FP=fp+invalid_count,
                    failures=failures, solver_failures=solver_failures, path_receipts=path_receipts,full_contract_pass=False)
        dump(output/(source+".json"), case); dump(output/(source+"_predictions.json"), predictions)
        receipts.append(case)
        for key, value in (("TP",tp),("FP",fp+invalid_count),("FN",fn),("G",spec["denominator"]),("official_valid_export_FP",fp),("invalid_selected_child_count",invalid_count)):
            totals[key] += value
        del graph
        gc.collect()
    if totals["G"] != 1221 or len(receipts) != 16:
        raise ValueError("full16/G1221 evaluation incomplete")
    verified = not any(c["scientific_status"] == "NOT_VERIFIED" for c in receipts)
    report = dict(step=step,totals=totals,complete_G=True,all_solvers_optimal=verified,
        scientific_status="NOT_VERIFIED" if not verified else ("FAILED_GEOMETRY_EXPORT" if totals["invalid_selected_child_count"] else "COMPLETE_RUNTIME_EVALUATION"),
        capacity_candidate=verified and totals["invalid_selected_child_count"]==0 and totals["TP"]>=1160 and totals["FP"]<=61,
        v4_replacement_candidate=verified and totals["invalid_selected_child_count"]==0 and totals["TP"]>=1188 and totals["FP"]<=46,
        full_contract_pass=False,source_order=[s["source"] for s in config["sources"]])
    dump(output/"complete.json", report)
    model.train(was_training)
    return report


def train_released(config_path, release_path, output, device):
    """Called only by --run with a root-issued exact-config release."""
    config_path=Path(config_path); output=Path(output)
    config=json.loads(config_path.read_text("utf8")); release=json.loads(Path(release_path).read_text("utf8"))
    if release.get("authorized") is not True or release.get("config_sha256") != file_sha256(config_path) or release.get("released_by") != "root":
        raise PermissionError("actual updates require root exact-config release")
    if set(config["gates_required"])-set(release.get("passed_gates", [])):
        raise PermissionError("all prerequisite gates must pass before optimizer exists")
    assert_core_hashes()
    if file_sha256(__file__)!=config["runner_sha256"] or file_sha256(ROOT/"weld_pipeline/structured_graph_artifacts_200.py")!=config["loader_sha256"]:
        raise ValueError("prepared runner/loader source changed")
    output.mkdir(parents=True,exist_ok=True); seed_runtime(device); install_highs_cpu2()
    mean,std=load_frozen175_normalizer(config["normalizer_path"])
    model=BoundaryIncidenceNet(mean,std).to(device).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    steps=0; traversals=0; path_attempts=0; source_slots=0; completed_sources=0; zero_gradient_sources=0; epoch_done=0; history=[]; failure=None
    try:
        for epoch in range(80):
            order=list(range(16)); random.Random(SEED+epoch).shuffle(order)
            for index in order:
                spec=config["sources"][index]; source=spec["source"]
                optimizer.zero_grad(set_to_none=True); source_slots+=1
                graph=load_runtime_graph(spec["artifact_directory"],source,spec["expected_sha256"])
                truth=load_training_truth(spec["artifact_directory"],source,graph,spec["expected_sha256"])
                views=complete_path_views(graph,truth)
                if len(views)!=spec["path_count"]:
                    raise ValueError("complete source path denominator changed")
                source_loss=0.;path_log=output/("epoch_"+str(epoch)+"_source_"+str(index)+"_paths.jsonl")
                path_handle=path_log.open("w",encoding="utf8")
                for path,local,local_truth,_ in views:
                    path_attempts+=1
                    def train_progress(stage,**values):
                        dump(output/"training_stage.json",dict(stage=stage,epoch=epoch,source=source,path=path,
                            actual_optimizer_steps=steps,path_attempts=path_attempts,completed_path_traversals=traversals,
                            full_source_path_denominator=len(views),**values))
                    loss,detail=exact_latent_loss(model,local,local_truth,config["chunk_size"],SolveOptions(**config["solver_options"]),train_progress)
                    (loss/len(views)).backward()
                    traversals+=1
                    source_loss+=float(loss.detach())/len(views)
                    path_handle.write(json.dumps(dict(path=path,**detail),ensure_ascii=False)+"\n");path_handle.flush()
                    del loss
                _,nonzero=gradient_state(model)
                path_handle.close()
                if nonzero:
                    optimizer.step();steps+=1
                else:zero_gradient_sources+=1
                completed_sources+=1
                history.append(dict(epoch=epoch,source=source,path_count=len(views),source_loss=source_loss,
                                    optimizer_step_executed=nonzero,actual_steps=steps,traversals=traversals,
                                    path_receipt_log=str(path_log),path_receipt_log_sha256=file_sha256(path_log)))
                dump(output/"training_progress.json",dict(status="ACTUAL_TRAINING",actual_optimizer_steps=steps,source_slots=source_slots,
                    completed_sources=completed_sources,zero_gradient_sources=zero_gradient_sources,attempted_path_traversals=path_attempts,
                    complete_path_traversals=traversals,epoch_completed=epoch_done,last_source=source,empty_paths_in_denominator=True))
                del graph,truth,views;gc.collect()
                if steps and nonzero and steps%1000==0:
                    torch.save(dict(state_dict=model.state_dict(),actual_steps=steps,config=config,optimizer=optimizer.state_dict()),output/("step_"+str(steps)+".pt"))
                    evaluate_full16(model,config,output,steps)
                if steps>=12000:
                    break
            epoch_done=epoch+1
            if steps>=12000:
                break
    except Exception as error:
        # Discard this source's accumulated gradients. No failed/nonoptimal
        # path is removed or used for a partial-source update.
        optimizer.zero_grad(set_to_none=True)
        if "path_handle" in locals() and not path_handle.closed:path_handle.close()
        failure=dict(source=locals().get("source"),path=locals().get("path"),epoch=locals().get("epoch"),
                     error_type=type(error).__name__,error=str(error),full_G_retained=1221,round_stopped=True,solver_failure_details=getattr(error,"details",None))
    torch.save(dict(state_dict=model.state_dict(),actual_steps=steps,config=config,optimizer=optimizer.state_dict()),output/"final.pt")
    final=evaluate_full16(model,config,output,"final_"+str(steps))
    receipt=dict(status="FAILED_ROUND_STOPPED" if failure else "BUDGET_COMPLETE",actual_optimizer_steps=steps,
        source_slots=source_slots,epochs_completed=epoch_done,maximum1280_source_steps=True,complete_path_traversals=traversals,
        attempted_path_traversals=path_attempts,completed_sources=completed_sources,zero_gradient_no_update_sources=zero_gradient_sources,
        failure=failure,final_evaluation=final,full_contract_pass=False,frozen197_prepare_metadata_left_unchanged=True)
    dump(output/"training_receipt.json",receipt);dump(output/"source_path_history.json",history)
    return receipt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare-config",action="store_true")
    parser.add_argument("--artifact-directory",default=str(ROOT/"weld_pipeline/reports/runtime_query_graph_oracle_195_20261002"))
    parser.add_argument("--config")
    parser.add_argument("--preflight-real-maxpath",action="store_true")
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--root-release")
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    if sum((args.prepare_config,args.preflight_real_maxpath,args.run))!=1:
        parser.error("exactly one prepare-config/preflight-real-maxpath/run action required")
    if args.prepare_config:
        prepare_config(args.output,args.artifact_directory)
    elif args.preflight_real_maxpath:
        if not args.config:parser.error("--config required")
        preflight_real_path(json.loads(Path(args.config).read_text("utf8")),args.output,device=args.device)
    else:
        if not args.config or not args.root_release:parser.error("--config and --root-release required")
        train_released(args.config,args.root_release,args.output,args.device)


if __name__=="__main__":
    main()
