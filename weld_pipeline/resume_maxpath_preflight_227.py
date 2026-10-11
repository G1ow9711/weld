"""One same-instance recovery: historical Runtime, compiled Good, full Bad.

No optimizer step exists. Native exact results must be expanded and pass the
unchanged213/197 certificates before a separate training release is possible.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import threading
import time

import numpy as np
import torch

from weld_pipeline import train_boundary_incidence_f1_214 as original
from weld_pipeline import highs_observed_backend_225 as observed
from weld_pipeline.fixed_variable_compiler_226 import compile_problem, solve_compiled
from weld_pipeline.restore_incidence_snapshot_224 import restore, revalidate_original_runtime
from weld_pipeline.durable_artifacts_223 import AtomicWriter, ArtifactRecorder

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'SAME_INSTANCE_MAXPATH_RECOVERY_WITH_EXACT_FIXED_COMPILATION_227_v1'


def read_json(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def pack_npz(**values):
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    return stream.getvalue()


class Telemetry:
    def __init__(self, output, binding):
        self.output, self.binding = Path(output), binding
        self.recorder = ArtifactRecorder(output)
        self.pending = queue.Queue(maxsize=128)
        self.stop = threading.Event()
        self.failures, self.dropped = [], 0
        self.lock = threading.RLock()
        self.phase = {'stage': 'RECOVERY_START', 'stage_revision': 0}
        self.started = time.monotonic()
        self.writer = threading.Thread(target=self._writer, daemon=True, name='phase-journal-writer-227')
        self.writer.start()
        self.heartbeat = threading.Thread(target=self._heartbeat, daemon=True, name='phase-heartbeat-227')
        self.heartbeat.start()

    def send(self, target, event):
        event = {**event, 'process_id': os.getpid(), 'attempt_id': self.output.name,
                 'UTC': datetime.now(timezone.utc).isoformat(),
                 'elapsed_seconds': time.monotonic() - self.started,
                 'graph_content_hash': self.binding['graph_content_hash'],
                 'optimizer_steps': 0}
        try:
            self.pending.put_nowait((target, event))
        except queue.Full:
            self.dropped += 1

    def phase_event(self, stage, **values):
        with self.lock:
            self.phase = {'stage': stage, 'stage_revision': self.phase['stage_revision'] + 1, **values}
            event = dict(self.phase)
        self.send('real_maxpath_progress.json', event)
        print('227_MAXPATH_STAGE', stage, flush=True)

    def _heartbeat(self):
        while not self.stop.wait(15):
            with self.lock:
                event = dict(self.phase, heartbeat=True)
            self.send('real_maxpath_progress.json', event)

    def _writer(self):
        while not self.stop.is_set() or not self.pending.empty():
            try:
                target, event = self.pending.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.recorder.dump(self.output / target, event)
            except Exception as error:
                self.failures.append({'target': target, 'error_type': type(error).__name__, 'error': str(error)})
                print('227_TELEMETRY_DEGRADED', type(error).__name__, flush=True)
            finally:
                self.pending.task_done()

    def close(self):
        self.stop.set()
        self.heartbeat.join(timeout=2)
        self.writer.join(timeout=10)
        return {**self.recorder.receipt(), 'queue_events_dropped': self.dropped,
                'writer_errors': self.failures, 'writer_still_running': self.writer.is_alive()}


def install_recovery_solver(output, binding, telemetry, writer):
    solver = original.solver213
    old_build, old_milp = solver._build_component, solver.milp
    context = {}
    def build(*args, **kwargs):
        built = old_build(*args, **kwargs)
        ids, edge_ids, pairs, a, dup_ids, dup_pairs, d, u, roles, truth, mode, paths = args
        c, integer, bounds, rows, size = built
        registry = size['_witness_registry']
        local = {int(identity): ordinal for ordinal, identity in enumerate(ids)}
        rules = [(registry['product_start'] + ordinal, local[int(left)], local[int(right)])
                 for ordinal, (left, right) in enumerate(dup_pairs)]
        context.clear()
        context.update(mode=mode, ids=ids, duplicate_pairs=dup_pairs, registry=registry,
                       rules=rules, preserve=np.arange(registry['product_start'] + registry['product_count'], len(c)))
        return built

    def native_call(*, c, integrality, bounds, constraints, options):
        mode = context['mode']
        if mode not in ('good', 'bad'):
            raise ValueError('Runtime must reuse the already revalidated historical certificate')
        prefix = Path(output) / (mode + '_component_0')
        full_hash = original.backend215.model_digest(c, integrality, bounds.lb, bounds.ub,
                                                    constraints.lb, constraints.ub, constraints.A)
        formulation = solver._formulation_digest(c, integrality, bounds, constraints)
        problem = None
        if mode == 'good':
            old_start = read_json(ROOT / 'weld_pipeline/reports/structured_training_runner_214_20261002/real_maxpath_attempt1/real_maxpath_good_component_0.json')
            if formulation != old_start['formulation_hash']:
                raise ValueError('Recovered full Good formulation differs from the original attempt')
            problem = compile_problem(c, integrality, bounds, constraints,
                zero_product_rules=context['rules'], preserve_columns=context['preserve'])
            expected = read_json(ROOT / 'weld_pipeline/reports/fixed_variable_compiler_226_20261002/actual_max_good_compilation_receipt.json')
            if problem.original_model_hash != expected['problem']['original_full_model_hash'] or problem.compiled_model_hash != expected['problem']['compiled_native_model_hash']:
                raise ValueError('Compiled real Good differs from the verified compilation')
            mapping = writer.commit_bytes(prefix.with_name(prefix.name + '_elimination_map.npz'), problem.mapping_bytes())
            writer.commit_json(prefix.with_name(prefix.name + '_compilation.json'), {**problem.receipt(), 'mapping': mapping})
        prepared = {'mode': mode, 'status': 'ACTUAL_NATIVE_CALL_START',
                    'graph_content_hash': binding['graph_content_hash'],
                    'objective_snapshot_hash': binding['objective_snapshot_hashes'][mode],
                    'original_full_formulation_hash': formulation, 'original_full_model_hash': full_hash,
                    'native_problem_scope': 'EXACT_FIXED_COMPILED' if problem is not None else 'ORIGINAL_FULL_F1',
                    'expected_native_model_hash': problem.compiled_model_hash if problem is not None else full_hash,
                    'solver_options': options, 'optimizer_steps': 0}
        writer.commit_json(prefix.with_name(prefix.name + '_start.json'), prepared)
        native_count, incumbent_sequence = [0], [0]
        local_context = dict(context)
        def model_observer(value):
            if value['native_full_model_hash'] != prepared['expected_native_model_hash']:
                raise ValueError('Actual native matrix/readback differs from compiled expectation')
            writer.commit_json(prefix.with_name(prefix.name + '_native_prepared.json'), {
                **value, 'original_full_model_hash': full_hash,
                'expected_native_model_hash': prepared['expected_native_model_hash'],
                'native_problem_scope': prepared['native_problem_scope'],
                'graph_content_hash': binding['graph_content_hash'],
                'objective_snapshot_hash': binding['objective_snapshot_hashes'][mode]})
        def progress_observer(value):
            telemetry.send(mode + '_live_native_bounds.json', {
                **value, 'mode': mode, 'objective_snapshot_hash': binding['objective_snapshot_hashes'][mode],
                'native_problem_scope': prepared['native_problem_scope'],
                'compiled_objective_constant': problem.objective_constant if problem is not None else 0.})
        def incumbent_observer(value):
            compiled_x = value.pop('native_x')
            full_x = problem.lift(compiled_x) if problem is not None else compiled_x
            full_x, checks = solver._rebuild_auxiliary_witness(full_x, local_context['ids'],
                local_context['duplicate_pairs'], local_context['registry'], c, bounds, constraints, 1e-9)
            ordinal = incumbent_sequence[0]
            incumbent_sequence[0] += 1
            target = Path(output) / 'incumbents' / (mode + '_' + str(ordinal).zfill(6) + '.npz')
            artifact = writer.commit_bytes(target, pack_npz(native_compiled_x=compiled_x, original_full_x=full_x))
            writer.commit_json(target.with_suffix('.json'), {**value, 'artifact': artifact, 'mode': mode,
                'native_problem_scope': prepared['native_problem_scope'], 'original_full_model_hash': full_hash,
                'objective_snapshot_hash': binding['objective_snapshot_hashes'][mode],
                'original_full_auxiliary_checks': checks, 'original_full_objective_min': float(np.dot(c, full_x)),
                'optimality_certificate': False, 'branch_and_bound_tree_checkpoint': False})
            native_count[0] += 1
        original.backend215.MODEL_OBSERVER = model_observer
        original.backend215.PROGRESS_OBSERVER = progress_observer
        observed.INCUMBENT_OBSERVER = incumbent_observer
        started = time.monotonic()
        try:
            result = solve_compiled(problem, options) if problem is not None else observed.milp(
                c, integrality=integrality, bounds=bounds, constraints=constraints, options=options)
            receipt = {**prepared, 'status': 'ACTUAL_NATIVE_CALL_RETURNED',
                'elapsed_seconds': time.monotonic() - started, 'raw_status': int(result.status),
                'raw_message': str(result.message), 'full_min_objective': original.solver213._number(result.fun),
                'full_min_dual_bound': original.solver213._number(result.mip_dual_bound),
                'raw_mip_gap': original.solver213._number(result.mip_gap),
                'native_callback_actual_event_counts': result.native_callback_actual_event_counts,
                'native_callback_errors': result.native_callback_errors,
                'native_unsaved_superseded_incumbents': result.native_unsaved_superseded_incumbents,
                'verified_incumbent_artifacts': native_count[0],
                'native_model_readback_hash': result.native_full_model_hash,
                'native_options_readback': result.native_options_verified}
            if result.x is not None:
                artifact = writer.commit_bytes(prefix.with_name(prefix.name + '_final_full_witness.npz'),
                                               pack_npz(original_full_x=result.x))
                receipt['final_witness'] = artifact
            writer.commit_json(prefix.with_name(prefix.name + '_return.json'), receipt)
            return result
        finally:
            original.backend215.MODEL_OBSERVER = None
            original.backend215.PROGRESS_OBSERVER = None
            observed.INCUMBENT_OBSERVER = None
    solver._build_component, solver.milp = build, native_call
    def uninstall():
        solver._build_component, solver.milp = old_build, old_milp
    return uninstall


def run(args):
    if args.output.exists():
        raise FileExistsError('Preserve original recovery attempt; no duplicate maximum run')
    args.output.mkdir(parents=True)
    writer, telemetry, uninstall = AtomicWriter(), None, None
    import ctypes
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.GetCurrentProcess.restype = ctypes.c_void_p
    api.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_uint64)] * 4
    times = [ctypes.c_uint64() for _ in range(4)]
    if not api.GetProcessTimes(api.GetCurrentProcess(), *[ctypes.byref(value) for value in times]):
        raise ctypes.WinError(ctypes.get_last_error())
    writer.commit_json(args.output / 'actual_compute_process_identity.json', {
        'process_id': os.getpid(), 'creation_filetime_ticks': times[0].value,
        'UTC': datetime.now(timezone.utc).isoformat(), 'actual_optimizer_steps': 0,
        'version': VERSION, 'scope': 'ACTUAL_PYTHON_COMPUTE_PROCESS_NOT_VENV_LAUNCHER'})
    try:
        release = read_json(args.preflight_release)
        if release['status'] != 'AUTHORIZED_SINGLE_ENGINEERING_RECOVERY_NO_OPTIMIZER':
            raise ValueError('Engineering recovery release required')
        for relative, expected in release['files_sha256'].items():
            if original.file_sha256(ROOT / relative) != expected:
                raise ValueError('Recovery code or prerequisite changed: ' + relative)
        model, graph, truth, scores, config, binding, restored = restore(args.config, args.original_attempt, args.device)
        writer.commit_json(args.output / 'restored_initial_bytes.json', restored)
        runtime, historical = revalidate_original_runtime(graph, scores, config, binding, args.original_attempt)
        writer.commit_json(args.output / 'historical_runtime_certificate.json', historical)
        telemetry = Telemetry(args.output, binding)
        telemetry.phase_event('HISTORICAL_RUNTIME_REVALIDATED', new_native_call=False)
        uninstall = install_recovery_solver(args.output, binding, telemetry, writer)
        detached = original.solver_199_scores(scores)
        options = original.SolveOptions(**config['solver_options'])
        telemetry.phase_event('GOOD_EXACT_SOLVE_START', mode='good')
        good = original.solver213.solve_best_truth_consistent_graph(graph, detached, truth, options)
        good_certificate = original.certified_selection_199(good, scores)
        writer.commit_json(args.output / 'good_full_optimal_certificate.json', original.solver_receipt(good))
        telemetry.phase_event('GOOD_EXACT_SOLVE_COMPLETE', mode='good', status=good.status)
        telemetry.phase_event('BAD_EXACT_SOLVE_START', mode='bad')
        bad = original.solver213.solve_loss_augmented_graph(graph, detached, truth, options)
        bad_certificate = original.certified_selection_199(bad, scores)
        writer.commit_json(args.output / 'bad_full_optimal_certificate.json', original.solver_receipt(bad))
        telemetry.phase_event('BAD_EXACT_SOLVE_COMPLETE', mode='bad', status=bad.status)
        units = len(truth.events) + len(truth.true_children)
        source_paths = next(spec['path_count'] for spec in config['sources'] if spec['source'] == '161-LB18A')
        for value in (scores.u, scores.a, scores.d):
            value.retain_grad()
        raw_loss = original.latent_structured_hinge_from_optima(scores, good_certificate, bad_certificate, bad.structural_loss)
        loss = raw_loss / max(1, units) / source_paths
        telemetry.phase_event('IDENTITY_HINGE_BACKWARD_START')
        loss.backward()
        gradients, nonzero = original.gradient_state(model)
        scale = 1. / max(1, units) / source_paths
        expected = [bad.selection.z.astype(float) - good.selection.z,
                    bad.selection.y.astype(float) - good.selection.y,
                    good.selection.t.astype(float) - bad.selection.t]
        score_gradients = []
        for name, value, difference in zip(('u', 'a', 'd'), (scores.u, scores.a, scores.d), expected):
            actual = value.grad.detach().cpu().numpy() if value.grad is not None else np.zeros(len(value))
            target = difference * scale
            if not np.allclose(actual, target, rtol=1e-6, atol=1e-10):
                raise ValueError('Identity hinge score derivative differs: ' + name)
            score_gradients.append({'field': name, 'max_abs_error': float(np.max(np.abs(actual - target), initial=0.))})
        if original.model_sha(model) != binding['model_state_hash']:
            raise ValueError('Preflight backward changed weights or buffers')
        original.assert_core_hashes()
        telemetry.phase_event('IDENTITY_HINGE_BACKWARD_COMPLETE', finite_gradients=True)
        final = {'status': 'PASS_SAME_INSTANCE_MAXPATH_RUNTIME_GOOD_BAD_AND_IDENTITY_BACKWARD',
                 'version': VERSION, 'runtime': 'HISTORICAL_NATIVE_CERTIFICATE_REVALIDATED',
                 'good': 'EXACT_FIXED_COMPILED_FULL_WITNESS_OPTIMAL', 'bad': 'FULL_F1_WITNESS_OPTIMAL',
                 'graph_content_hash': binding['graph_content_hash'], 'objective_snapshot_hashes': binding['objective_snapshot_hashes'],
                 'raw_identity_hinge': float(raw_loss.detach()), 'source_C_path_normalized_loss': float(loss.detach()),
                 'truth_units': units, 'source_path_denominator': source_paths,
                 'finite_gradients': True, 'gradient_nonzero': nonzero, 'score_gradient_checks': score_gradients,
                 'before_after_model_hash': binding['model_state_hash'], 'actual_optimizer_steps': 0,
                 'training_release': False, 'full_contract_pass': False}
        writer.commit_json(args.output / 'maxpath_preflight_complete.json', final)
        print('227_MAXPATH_PREFLIGHT_PASS_NO_OPTIMIZER', flush=True)
    except Exception as error:
        writer.commit_json(args.output / 'preflight_failure.json', {'status': 'FAILED_ENGINEERING_RECOVERY_NO_UPDATES',
            'error_type': type(error).__name__, 'error': str(error), 'actual_optimizer_steps': 0, 'full_contract_pass': False})
        raise
    finally:
        if uninstall is not None:
            uninstall()
        if telemetry is not None:
            writer.commit_json(args.output / 'telemetry_terminal_receipt.json', telemetry.close())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--original-attempt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-release', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
