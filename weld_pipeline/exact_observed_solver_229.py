"""Per-graph exact Good compilation and full Runtime/Bad native observation.

No neural framework is imported. Every scope binds the current complete graph,
scores and train-only supervision. Integer objectives/constraints and consumer
certificates stay those of frozen213; compilation only substitutes proven fixed
variables. No GT information is accepted for a Runtime call.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import ctypes
from dataclasses import asdict
from hashlib import sha256
import inspect
import io
import json
import os
from pathlib import Path
import queue
import threading
import time

import numpy as np

from . import structured_graph_solver_f1_213 as solver
from . import highs_backend_215 as base
from . import highs_observed_backend_225 as observed
from .fixed_variable_compiler_226 import compile_problem, solve_compiled
from .durable_artifacts_223 import AtomicWriter, ArtifactRecorder

VERSION = 'EXACT_PER_GRAPH_OBSERVED_NATIVE_SOURCE_C_229_v1'


def plain(value):
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def pack_npz(**values):
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    return stream.getvalue()


def actual_process_identity():
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.GetCurrentProcess.restype = ctypes.c_void_p
    api.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_uint64)] * 4
    times = [ctypes.c_uint64() for _ in range(4)]
    if not api.GetProcessTimes(api.GetCurrentProcess(), *[ctypes.byref(value) for value in times]):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(process_id=os.getpid(), creation_filetime_ticks=times[0].value,
                scope='ACTUAL_NATIVE_PYTHON_PROCESS_NOT_VENV_LAUNCHER', version=VERSION)


class TrainingTelemetry:
    """One optional-progress writer. Required mathematical files bypass it."""
    def __init__(self, output, owner, *, heartbeat_seconds=15.):
        self.output, self.owner = Path(output), dict(owner)
        self.recorder = ArtifactRecorder(output,
            telemetry_names=('training_stage.json', 'training_progress.json'),
            telemetry_suffixes=('_live_native_bounds.json',))
        self.pending = queue.Queue(maxsize=128)
        self.stop = threading.Event()
        self.guard = threading.RLock()
        self.phase = {'stage': 'ENGINE_START'}
        self.failures, self.dropped = [], 0
        self.started = time.monotonic()
        self.heartbeat_seconds = heartbeat_seconds
        self.writer = threading.Thread(target=self._write, daemon=True, name='training-journal-229')
        self.writer.start()
        self.heartbeat = threading.Thread(target=self._beat, daemon=True, name='training-heartbeat-229')
        self.heartbeat.start()

    def send(self, target, event):
        value = plain({**event, **self.owner, 'UTC': datetime.now(timezone.utc).isoformat(),
                       'elapsed_seconds': time.monotonic()-self.started})
        try:
            self.pending.put_nowait((target, value))
        except queue.Full:
            self.dropped += 1

    def phase_event(self, stage, **values):
        with self.guard:
            self.phase = plain(dict(stage=stage, **values))
            event = dict(self.phase)
        self.send('training_stage.json', event)

    def _beat(self):
        while not self.stop.wait(self.heartbeat_seconds):
            with self.guard:
                event = dict(self.phase, heartbeat=True)
            self.send('training_stage.json', event)

    def _write(self):
        while not self.stop.is_set() or not self.pending.empty():
            try:
                target, value = self.pending.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                self.recorder.dump(self.output/target, value)
            except Exception as error:
                self.failures.append(dict(target=target, error_type=type(error).__name__, error=str(error)))
            finally:
                self.pending.task_done()

    def close(self):
        self.stop.set()
        self.heartbeat.join(timeout=2)
        self.writer.join(timeout=30)
        if self.heartbeat.is_alive() or self.writer.is_alive():
            raise RuntimeError('Training telemetry failed to reach terminal state')
        return dict(owner=self.owner, journal=self.recorder.receipt(), errors=self.failures,
                    dropped_optional_progress_events=self.dropped, version=VERSION)


def objective_snapshot(graph, scores, truth, mode):
    """Byte-identical to214 snapshot definition, without importing Torch."""
    _, _, roles, _, children, duplicates = solver._topology(graph)
    labels = solver._truth(truth, roles) if mode != 'runtime' else None
    digest = solver._objective_digest(scores.node_scores, labels, mode, children, duplicates)
    for kind, values in (('child', scores.child_scores), ('duplicate', scores.duplicate_scores)):
        if callable(values):
            raise ValueError('Current complete score arrays required')
        digest.update(kind.encode('ascii'))
        digest.update(np.ascontiguousarray(values, dtype='<f8').tobytes())
    return digest.hexdigest()


class ExactObservedAdapter:
    def __init__(self, output, owner, telemetry, *, writer=None):
        self.output, self.owner, self.telemetry = Path(output), dict(owner), telemetry
        self.writer = writer or AtomicWriter()
        self.metadata = {}
        self.scope, self.component_context = None, None
        self.scope_sequence, self.native_calls = 0, 0
        self._originals = None

    def set_metadata(self, **values):
        if self.scope is not None:
            raise RuntimeError('Cannot change training binding during an active solve')
        self.metadata = plain(values)

    def _emit(self, target, values):
        self.telemetry.send(target, values)

    def install(self):
        if self._originals is not None:
            raise RuntimeError('Exact adapter already installed')
        names = ('_build_component', 'milp', 'solve_runtime_graph',
                 'solve_best_truth_consistent_graph', 'solve_loss_augmented_graph')
        self._originals = {name: getattr(solver, name) for name in names}
        self._build_signature = inspect.signature(self._originals['_build_component'])
        solver._build_component, solver.milp = self._build, self._native
        solver.solve_runtime_graph = self._runtime
        solver.solve_best_truth_consistent_graph = self._good
        solver.solve_loss_augmented_graph = self._bad
        return self

    def uninstall(self):
        if self.scope is not None:
            raise RuntimeError('Cannot remove adapter during an active solve')
        if self._originals is not None:
            for name, value in self._originals.items():
                setattr(solver, name, value)
            self._originals = None

    @contextmanager
    def _bound_scope(self, graph, scores, truth, mode):
        if self.scope is not None:
            raise RuntimeError('Nested/parallel native scopes refused')
        graph_hash = solver.query_graph_195_fingerprint(graph)
        if scores.graph_fingerprint != graph_hash:
            raise ValueError('Current graph/score binding differs')
        if mode == 'runtime' and truth is not None:
            raise ValueError('Runtime accepts no supervision')
        score_hash = objective_snapshot(graph, scores, truth, mode)
        ordinal = self.scope_sequence
        self.scope_sequence += 1
        folder = self.output/'solver_calls'/f'scope_{ordinal:09d}_{mode}'
        folder.mkdir(parents=True, exist_ok=False)
        binding = plain({**self.metadata, **self.owner, 'mode': mode,
            'scope_ordinal': ordinal, 'graph_content_hash': graph_hash,
            'objective_snapshot_hash': score_hash, 'version': VERSION})
        score_file = self.writer.commit_bytes(folder/'complete_current_scores.npz', pack_npz(
            u=np.asarray(scores.node_scores), a=np.asarray(scores.child_scores), d=np.asarray(scores.duplicate_scores),
            **({} if truth is None else {'node_event_index':np.asarray(truth.node_event_index)})))
        self.writer.commit_json(folder/'input_binding.json', {**binding, 'score_file':score_file})
        self.scope = dict(folder=folder, binding=binding, components=0)
        try:
            yield
        finally:
            self.scope, self.component_context = None, None

    def _solve(self, name, mode, graph, scores, truth, options):
        with self._bound_scope(graph, scores, truth, mode):
            call = self._originals[name]
            result = call(graph, scores, options) if truth is None else call(graph, scores, truth, options)
            selection = result.selection
            self.writer.commit_json(self.scope['folder']/'full_f1_result.json', plain(dict(
                binding=self.scope['binding'], status=result.status,
                optimal_proved=result.optimal_proved, complete_graph_solution=result.complete_graph_solution,
                graph_fingerprint=result.graph_fingerprint, base_score=result.base_score,
                objective_max=result.objective_max, upper_bound_max=result.upper_bound_max, mip_gap=result.mip_gap,
                structural_loss=result.structural_loss, counts=result.counts, components=result.component_receipts,
                selected_nodes=np.flatnonzero(selection.z) if selection is not None else None,
                selected_children=np.flatnonzero(selection.y) if selection is not None else None,
                neural_score_and_gradient_check='CONSUMER_REQUIRED',
                native_component_calls=self.scope['components'])))
            return result

    def _runtime(self, graph, scores, options=None):
        return self._solve('solve_runtime_graph', 'runtime', graph, scores, None, options)

    def _good(self, graph, scores, labels, options=None):
        return self._solve('solve_best_truth_consistent_graph', 'good', graph, scores, labels, options)

    def _bad(self, graph, scores, labels, options=None):
        return self._solve('solve_loss_augmented_graph', 'bad', graph, scores, labels, options)

    def _build(self, *args, **kwargs):
        if self.scope is None:
            raise RuntimeError('Full graph/score scope must precede component construction')
        parameters = self._build_signature.bind(*args, **kwargs).arguments
        if parameters['mode'] != self.scope['binding']['mode']:
            raise ValueError('Component mode differs from graph binding')
        built = self._originals['_build_component'](*args, **kwargs)
        c, integer, bounds, constraints, size = built
        registry = size['_witness_registry']
        ids, pairs = parameters['ids'], parameters['dup_pairs']
        local = {int(identity): ordinal for ordinal, identity in enumerate(ids)}
        self.component_context = dict(ids=ids, duplicate_pairs=pairs, registry=registry,
            rules=[(registry['product_start']+ordinal, local[int(a)], local[int(b)])
                   for ordinal,(a,b) in enumerate(pairs)],
            preserve=np.arange(registry['product_start']+registry['product_count'],len(c)))
        return built

    def _native(self, *, c, integrality, bounds, constraints, options):
        if self.scope is None or self.component_context is None:
            raise RuntimeError('Unbound native component refused')
        binding = dict(self.scope['binding'])
        mode, ordinal = binding['mode'], self.scope['components']
        self.scope['components'] += 1
        binding['component_ordinal'] = ordinal
        folder = self.scope['folder']/f'component_{ordinal:06d}'
        folder.mkdir(exist_ok=False)
        context = self.component_context
        full_hash = base.model_digest(c, integrality, bounds.lb, bounds.ub,
                                     constraints.lb, constraints.ub, constraints.A)
        formulation = solver._formulation_digest(c, integrality, bounds, constraints)
        problem = compile_problem(c, integrality, bounds, constraints,
            zero_product_rules=context['rules'], preserve_columns=context['preserve']) if mode == 'good' else None
        mip_anchor = None
        if problem is not None and not np.count_nonzero(problem.integer):
            original_binary = np.flatnonzero(np.asarray(integrality) == 1)
            if len(original_binary):
                anchor = int(original_binary[0])
                if bounds.lb[anchor] != bounds.ub[anchor]:
                    raise ValueError('No compiled binaries but original anchor is not fixed')
                # Keep an existing fixed variable. Do not invent a binary or
                # fill undefined native LP MIP fields with synthetic zeros.
                problem = compile_problem(c, integrality, bounds, constraints,
                    zero_product_rules=context['rules'],
                    preserve_columns=np.unique(np.r_[context['preserve'],anchor]))
                mip_anchor = dict(original_column=anchor,fixed_value=float(bounds.lb[anchor]),
                                  reason='PRESERVE_EXISTING_FIXED_INTEGER_FOR_NATIVE_MIP_CERTIFICATE')
        if problem is not None:
            mapping = self.writer.commit_bytes(folder/'elimination_map.npz', problem.mapping_bytes())
            self.writer.commit_json(folder/'compilation.json', {**problem.receipt(), 'mapping':mapping})
        prepared = {**binding, 'status':'ACTUAL_NATIVE_CALL_START', 'original_full_model_hash':full_hash,
            'original_full_formulation_hash':formulation,
            'native_problem_scope':'EXACT_FIXED_COMPILED' if problem is not None else 'ORIGINAL_FULL_F1',
            'expected_native_model_hash':problem.compiled_model_hash if problem is not None else full_hash,
            'compiled_objective_constant':problem.objective_constant if problem is not None else 0.,
            'native_mip_metadata_anchor':mip_anchor,
            'solver_options':dict(options)}
        self.writer.commit_json(folder/'start.json', prepared)
        archive_count, archive_sequence = [0], [0]
        previous = (base.MODEL_OBSERVER, base.PROGRESS_OBSERVER, observed.INCUMBENT_OBSERVER)

        def model_observer(value):
            if value['native_full_model_hash'] != prepared['expected_native_model_hash']:
                raise ValueError('Native model readback differs from expected current component')
            self.writer.commit_json(folder/'native_prepared.json', {**value, 'binding':prepared})

        def progress_observer(value):
            # Only enqueue while in the C callback; no I/O or phase lock.
            self._emit('current_live_native_bounds.json', {**value, **prepared})

        def incumbent_observer(value):
            value = dict(value)
            native_x = value.pop('native_x')
            full_x = problem.lift(native_x) if problem is not None else native_x
            full_x, checks = solver._rebuild_auxiliary_witness(full_x, context['ids'],
                context['duplicate_pairs'], context['registry'], c, bounds, constraints, 1e-9)
            number = archive_sequence[0]
            archive_sequence[0] += 1
            target = folder/'incumbents'/f'primal_{number:06d}.npz'
            artifact = self.writer.commit_bytes(target, pack_npz(native_x=native_x, original_full_x=full_x))
            self.writer.commit_json(target.with_suffix('.json'), plain(dict(
                **value, binding=prepared, artifact=artifact, original_full_auxiliary_checks=checks,
                original_full_min_objective=float(np.dot(c,full_x)),
                optimality_certificate=False, branch_tree_checkpoint=False)))
            archive_count[0] += 1

        base.MODEL_OBSERVER, base.PROGRESS_OBSERVER = model_observer, progress_observer
        observed.INCUMBENT_OBSERVER = incumbent_observer
        started = time.monotonic()
        try:
            result = solve_compiled(problem, options) if problem is not None else observed.milp(
                c, integrality=integrality, bounds=bounds, constraints=constraints, options=options)
            self.native_calls += 1
            if getattr(result,'native_callback_errors',[]):
                raise RuntimeError('Native observer/complete primal archive errors: '+repr(result.native_callback_errors))
            witness = None
            if result.x is not None:
                witness = self.writer.commit_bytes(folder/'final_original_full_witness.npz',
                    pack_npz(original_full_x=np.asarray(result.x,np.float64)))
            self.writer.commit_json(folder/'return.json', plain(dict(prepared, status='ACTUAL_NATIVE_CALL_RETURNED',
                elapsed_seconds=time.monotonic()-started, raw_status=int(result.status),raw_message=str(result.message),
                full_min_objective=solver._number(result.fun), full_min_dual_bound=solver._number(result.mip_dual_bound),
                raw_native_gap=solver._number(result.mip_gap), final_full_witness=witness,
                verified_feasible_primal_archives=archive_count[0],
                native_options_readback=getattr(result,'native_options_verified',None),
                native_model_readback_hash=getattr(result,'native_full_model_hash',None),
                native_callback_actual_event_counts=getattr(result,'native_callback_actual_event_counts',{}),
                native_unsaved_superseded_incumbents=getattr(result,'native_unsaved_superseded_incumbents',0))))
            return result
        finally:
            base.MODEL_OBSERVER, base.PROGRESS_OBSERVER, observed.INCUMBENT_OBSERVER = previous
