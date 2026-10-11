"""Recover original initial bytes and revalidate a historical native certificate.

No optimizer exists here. Certificate replay is explicitly a historical replay,
not a new native call, a fresh dual bound, or a trained accuracy measurement.
"""
import argparse
import ctypes
import json
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import torch
from scipy.optimize import OptimizeResult

from weld_pipeline import train_boundary_incidence_f1_214 as original
from weld_pipeline.durable_artifacts_223 import AtomicWriter

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'restore_original_initial_bytes_and_runtime_certificate_224_v1'


def read_json(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def verify_original_termination(snapshot):
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    api.OpenProcess.restype = ctypes.c_void_p
    api.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_uint64)] * 4
    api.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    api.CloseHandle.argtypes = [ctypes.c_void_p]
    observations = []
    for expected in snapshot['process_observations']:
        handle = api.OpenProcess(0x1000, False, expected['pid'])
        if not handle:
            error = ctypes.get_last_error()
            if error != 87:
                raise ctypes.WinError(error)
            observations.append({'pid': expected['pid'], 'current': 'ABSENT',
                                 'original_bound_exit_snapshot_verified': True})
            continue
        try:
            times = [ctypes.c_uint64() for _ in range(4)]
            code = ctypes.c_uint32()
            if not api.GetProcessTimes(handle, *[ctypes.byref(value) for value in times]):
                raise ctypes.WinError(ctypes.get_last_error())
            if not api.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            if times[0].value != expected['creation_filetime_ticks']:
                # A reused PID is a different process; the original bound
                # termination snapshot still proves the original is terminal.
                observations.append({'pid': expected['pid'], 'current': 'PID_REUSED',
                                     'original_bound_exit_snapshot_verified': True})
                continue
            if code.value == 259 or not times[1].value:
                raise RuntimeError('Original process still active; do not duplicate a maximum job')
            if code.value != expected['exit_code'] or times[1].value != expected['exit_filetime_ticks']:
                raise RuntimeError('Original kernel termination evidence differs')
            observations.append({'pid': expected['pid'], 'current': 'ORIGINAL_TERMINATED',
                                 'creation_filetime_ticks': times[0].value,
                                 'exit_filetime_ticks': times[1].value, 'exit_code': code.value})
        finally:
            api.CloseHandle(handle)
    return observations


def same_bytes(name, actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.dtype != expected.dtype or actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
        raise ValueError('Original initial bytes differ: ' + name)
    if not np.isfinite(actual).all():
        raise ValueError('Nonfinite initial array: ' + name)
    return {'field': name, 'dtype': str(actual.dtype), 'shape': list(actual.shape), 'byte_identical': True}


def restore(config_path, original_attempt, device='cuda'):
    config_path, original_attempt = Path(config_path), Path(original_attempt)
    original.assert_core_hashes()
    config = read_json(config_path)
    if original.file_sha256(config_path) != 'fe86560d3c9273188978e49d04583064ae4d8b63fe3f8f5d3c65b98262aff378':
        raise ValueError('Original214 config changed')
    if original.file_sha256(original.__file__) != config['runner_sha256']:
        raise ValueError('Original214 runner changed')
    if original.file_sha256(ROOT / 'weld_pipeline/structured_graph_artifacts_200.py') != config['loader_sha256']:
        raise ValueError('Original200 loader changed')
    binding = read_json(original_attempt / 'initial_snapshot_binding.json')
    for filename, field in [('initial_state.pt', 'state_sha256'), ('initial_complete_scores.npz', 'score_sha256')]:
        if original.file_sha256(original_attempt / filename) != binding[field]:
            raise ValueError('Initial artifact changed: ' + filename)
    kernel = verify_original_termination(read_json(original_attempt.parent / 'root_interruption_observation_20261002.json'))
    environment = original.seed_runtime(device)
    spec = next(source for source in config['sources'] if source['source'] == '161-LB18A')
    graph = original.load_runtime_graph(spec['artifact_directory'], spec['source'], spec['expected_sha256'])
    truth = original.load_training_truth(spec['artifact_directory'], spec['source'], graph, spec['expected_sha256'])
    path_id, path_graph, path_truth, ordinals = next(view for view in original.complete_path_views(graph, truth) if view[0] == 86)
    del graph, truth
    with (original_attempt / 'initial_state.pt').open('rb') as stream:
        checkpoint = torch.load(stream, map_location='cpu')
    if checkpoint['optimizer_steps'] != 0 or checkpoint['seed'] != config['seed'] or checkpoint['config'] != config:
        raise ValueError('Checkpoint state, seed or embedded config differs')
    mean, std = original.load_frozen175_normalizer(config['normalizer_path'])
    model = original.BoundaryIncidenceNet(mean, std)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    if original.model_sha(model) != binding['model_state_hash'] or checkpoint['model_state_hash'] != binding['model_state_hash']:
        raise ValueError('Original model state hash differs')
    model = model.to(device).train()
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    inputs, relations = original.query_graph_195_inputs(path_graph)
    scores = model(inputs, relations, chunk_size=config['chunk_size'])
    detached = original.solver_199_scores(scores)
    checks = []
    with np.load(original_attempt / 'initial_complete_scores.npz', allow_pickle=False) as saved:
        for name, value in [('u_native', scores.u), ('a_native', scores.a), ('d_native', scores.d)]:
            checks.append(same_bytes(name, value.detach().cpu().numpy(), saved[name]))
        for name, value in [('u_solve_float64', detached.node_scores),
                            ('a_solve_float64', detached.child_scores),
                            ('d_solve_float64', detached.duplicate_scores),
                            ('node_event_index', np.asarray(path_truth.node_event_index, '<i8')),
                            ('original_global_node_ordinals', np.asarray(ordinals, '<i8'))]:
            checks.append(same_bytes(name, np.asarray(value), saved[name]))
    if detached.graph_fingerprint != binding['graph_content_hash']:
        raise ValueError('Restored complete graph fingerprint differs')
    hashes = {mode: original.expected_objective_snapshot(path_graph, detached, path_truth, mode)
              for mode in ('runtime', 'good', 'bad')}
    if hashes != binding['objective_snapshot_hashes']:
        raise ValueError('Restored mode objective snapshot differs')
    if original.model_sha(model) != binding['model_state_hash']:
        raise ValueError('Restore or forward changed model state')
    receipt = {'version': VERSION, 'status': 'PASS_ORIGINAL_INITIAL_BYTES_RESTORED',
               'UTC': datetime.now(timezone.utc).isoformat(), 'source': spec['source'], 'path_id': path_id,
               'kernel_original_termination': kernel, 'environment': environment,
               'initial_binding_sha256': original.file_sha256(original_attempt / 'initial_snapshot_binding.json'),
               'config_sha256': original.file_sha256(config_path), 'model_state_hash': binding['model_state_hash'],
               'graph_content_hash': detached.graph_fingerprint, 'objective_snapshot_hashes': hashes,
               'arrays_byte_identical': checks, 'complete_counts': {'nodes': len(path_graph.nodes),
                   'children': path_graph.pair_count('child'), 'duplicate': path_graph.pair_count('duplicate'),
                   'message': path_graph.pair_count('message')},
               'GPU_peak_allocated_bytes': int(torch.cuda.max_memory_allocated()) if str(device).startswith('cuda') else 0,
               'GPU_peak_reserved_bytes': int(torch.cuda.max_memory_reserved()) if str(device).startswith('cuda') else 0,
               'actual_optimizer_steps': 0, 'new_native_solve_calls': 0,
               'actual_model_precision_claim': False, 'full_contract_pass': False}
    return model, path_graph, path_truth, scores, config, binding, receipt


def revalidate_original_runtime(path_graph, scores, config, binding, original_attempt):
    original_attempt = Path(original_attempt)
    raw = read_json(original_attempt / 'real_maxpath_runtime_component_0.json')
    prepared = read_json(original_attempt / 'real_maxpath_runtime_component_0_native_prepared.json')
    mode_receipt = read_json(original_attempt / 'real_maxpath_mode_runtime.json')
    if raw['status'] != 'ACTUAL_NATIVE_MILP_CALL_RETURNED' or raw['mode'] != 'runtime':
        raise ValueError('Original native Runtime call never returned')
    if (raw['raw_status'] != 0 or raw['raw_mip_gap'] != 0 or raw['native_model_verified'] is not True
            or raw['raw_min_objective'] != raw['raw_min_dual_bound']):
        raise ValueError('Original Runtime native optimum unproved')
    if mode_receipt['status'] != 'OPTIMAL' or not mode_receipt['optimal_proved'] or not mode_receipt['complete_graph_solution']:
        raise ValueError('Original full Runtime graph was not certified')
    if (raw['graph_content_hash'] != binding['graph_content_hash']
            or raw['objective_snapshot_hash'] != binding['objective_snapshot_hashes']['runtime']
            or mode_receipt['counts']['objective_snapshot_hash'] != binding['objective_snapshot_hashes']['runtime']):
        raise ValueError('Historical Runtime graph or objective binding differs')
    expected_options = original.backend215._options(raw['solver_options'])
    if raw['native_options_verified'] != expected_options or prepared['native_options_verified'] != expected_options:
        raise ValueError('Historical actual option readback differs')
    witness_path = original_attempt / 'real_maxpath_runtime_component_0_native_witness.npz'
    if original.file_sha256(witness_path) != raw['native_witness_sha256']:
        raise ValueError('Original native primal witness changed')
    with np.load(witness_path, allow_pickle=False) as saved:
        x = saved['native_x'].copy()
    replay = {'native_solver_ran_now': False, 'historical_components_replayed': 0,
              'original_native_return_sha256': original.file_sha256(original_attempt / 'real_maxpath_runtime_component_0.json'),
              'original_native_witness_sha256': raw['native_witness_sha256'],
              'original_native_model_readback_sha256': original.file_sha256(original_attempt / 'real_maxpath_runtime_component_0_native_prepared.json')}
    def replay_native(*, c, integrality, bounds, constraints, options):
        if replay['historical_components_replayed'] != 0:
            raise ValueError('Exactly one historical full-path Runtime component is bound')
        formulation = original.solver213._formulation_digest(c, integrality, bounds, constraints)
        model_hash = original.backend215.model_digest(c, integrality, bounds.lb, bounds.ub,
                                                     constraints.lb, constraints.ub, constraints.A)
        if formulation != raw['formulation_hash']:
            raise ValueError('Rebuilt historical Runtime full formulation differs')
        if model_hash != raw['native_model_hash'] or model_hash != prepared['original_full_model_hash'] or model_hash != prepared['native_full_model_hash']:
            raise ValueError('Rebuilt historical Runtime full native model differs')
        if options != raw['solver_options']:
            raise ValueError('Rebuilt Runtime solver options differ')
        if x.shape != np.asarray(c).shape:
            raise ValueError('Original native primal shape differs')
        replay.update(historical_components_replayed=1, formulation_hash=formulation,
                      complete_native_model_hash=model_hash,
                      witness_source='HISTORICAL_NATIVE_RETURN_NOT_NEW_NATIVE_CALL')
        return OptimizeResult(status=raw['raw_status'], x=x.copy(), fun=raw['raw_min_objective'],
                              mip_dual_bound=raw['raw_min_dual_bound'], mip_gap=raw['raw_mip_gap'],
                              message='Historical HiGHS 1.10.0 certificate revalidation; no native execution')
    original_milp = original.solver213.milp
    try:
        original.solver213.milp = replay_native
        result = original.solver213.solve_runtime_graph(path_graph, original.solver_199_scores(scores),
                                                       original.SolveOptions(**config['solver_options']))
    finally:
        original.solver213.milp = original_milp
    original.certified_selection_199(result, scores)
    if (replay['historical_components_replayed'] != 1
            or np.flatnonzero(result.selection.z).tolist() != mode_receipt['selected_nodes']
            or np.flatnonzero(result.selection.y).tolist() != mode_receipt['selected_children']
            or result.selection.selected_child_pairs.tolist() != mode_receipt['selected_pairs']):
        raise ValueError('Historical Runtime complete selections differ')
    replay.update(status='PASS_REVALIDATED_HISTORICAL_NATIVE_RUNTIME_CERTIFICATE',
                  graph_content_hash=result.graph_fingerprint,
                  objective_snapshot_hash=result.counts['objective_snapshot_hash'],
                  raw_native_primal_min=raw['raw_min_objective'], raw_native_dual_min=raw['raw_min_dual_bound'],
                  raw_native_gap=raw['raw_mip_gap'], full_graph_certificate=original.solver_receipt(result),
                  actual_optimizer_steps=0, actual_model_precision_claim=False)
    return result, replay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-attempt', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Preserve original restoration attempt')
    args.output.mkdir(parents=True)
    writer = AtomicWriter()
    started = time.monotonic()
    try:
        model, graph, truth, scores, config, binding, receipt = restore(args.config, args.original_attempt, args.device)
        writer.write_json(args.output / 'restored_forward_receipt.json', receipt)
        print('224_SAME_INITIAL_BYTES_PASS', receipt['complete_counts'], flush=True)
        result, replay = revalidate_original_runtime(graph, scores, config, binding, args.original_attempt)
        writer.write_json(args.output / 'runtime_historical_certificate_revalidation.json', replay)
        receipt.update(status='PASS_RESTORATION_AND_HISTORICAL_RUNTIME_REVALIDATION_NO_UPDATES',
                       elapsed_seconds=time.monotonic() - started, new_native_solve_calls=0,
                       restore_source_sha256=original.file_sha256(__file__),
                       writer_source_sha256=original.file_sha256(ROOT / 'weld_pipeline/durable_artifacts_223.py'),
                       Good_Bad='NOT_EXECUTED_BY_RESTORE_VERIFIER', identity_hinge_backward='NOT_EXECUTED',
                       original_artifacts_modified=False)
        writer.write_json(args.output / 'verification_receipt.json', receipt)
        print('224_RESTORATION_COMPLETE_NO_NATIVE_NO_UPDATES', flush=True)
    except Exception as error:
        writer.write_json(args.output / 'verification_failure.json', {'status': 'RESTORE_FAILED_NO_UPDATES',
            'error_type': type(error).__name__, 'error': str(error), 'actual_optimizer_steps': 0,
            'full_contract_pass': False})
        raise


if __name__ == '__main__':
    main()
