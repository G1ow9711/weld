"""Pinned native backend with non-logging progress and verified incumbent events.

The full mathematical model and native options use the frozen215 validators.
An incumbent is a feasible primal checkpoint, never an optimality certificate
or a saved branch-and-bound search tree. No callback supplies a user solution.
"""
import time
import queue
import threading

import numpy as np
from scipy.optimize import OptimizeResult

from weld_pipeline import highs_backend_215 as base

highspy = base.highspy
VERSION = 'HIGHS_1100_FULL_MODEL_OBSERVED_CALLBACKS_225_v3'
INCUMBENT_OBSERVER = None


def milp(c, *, integrality, bounds, constraints, options=None):
    cost = np.ascontiguousarray(c, np.float64)
    raw_integer = np.asarray(integrality)
    if cost.ndim != 1 or raw_integer.shape != cost.shape or np.any((raw_integer != 0) & (raw_integer != 1)):
        raise ValueError('Complete binary/continuous integrality required')
    integer = np.asarray(raw_integer, np.uint8)
    matrix = constraints.A.tocsc()
    lower, upper = np.asarray(bounds.lb, np.float64), np.asarray(bounds.ub, np.float64)
    row_lower, row_upper = np.asarray(constraints.lb, np.float64), np.asarray(constraints.ub, np.float64)
    expected = base.model_digest(cost, integer, lower, upper, row_lower, row_upper, matrix)
    solver = highspy.Highs()
    native_options = base._readback_options(solver, base._options(options))
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = len(cost), matrix.shape[0]
    lp.sense_, lp.offset_ = highspy.ObjSense.kMinimize, 0.0
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = cost, lower, upper
    lp.row_lower_, lp.row_upper_ = row_lower, row_upper
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = matrix.shape[1], matrix.shape[0]
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = (
        np.asarray(matrix.indptr, np.int64), np.asarray(matrix.indices, np.int64), np.asarray(matrix.data, np.float64))
    lp.integrality_ = [highspy.HighsVarType.kInteger if value else highspy.HighsVarType.kContinuous for value in integer]
    if solver.passModel(lp) != highspy.HighsStatus.kOk:
        raise ValueError('Complete native model rejected')
    actual = base._readback_model(solver, expected)
    del lp
    if base.MODEL_OBSERVER is not None:
        base.MODEL_OBSERVER({'status': 'NATIVE_OPTIONS_AND_FULL_MODEL_READBACK_PASS',
                            'native_version': solver.version(), 'native_options_verified': native_options,
                            'original_full_model_hash': expected, 'native_full_model_hash': actual,
                            'observer_version': VERSION})
    progress, incumbent = base.PROGRESS_OBSERVER, INCUMBENT_OBSERVER
    event_counts, callback_errors, registered = {}, [], []
    last_progress = [-float('inf')]
    tolerance = native_options['mip_feasibility_tolerance']
    pending = queue.Queue(maxsize=1)
    worker_stop = threading.Event()
    dropped = [0]

    def process_incumbents():
        while not worker_stop.is_set() or not pending.empty():
            try:
                candidate = pending.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                x = candidate.pop('native_x')
                if x.shape != cost.shape or not np.isfinite(x).all():
                    raise ValueError('Callback primal vector incomplete or nonfinite')
                binary = integer.astype(bool)
                if np.max(np.abs(x[binary] - np.rint(x[binary])), initial=0.) > tolerance:
                    raise ValueError('Callback primal integrality unverified')
                lhs = np.asarray(matrix @ x)
                row_tolerance = tolerance * (1 + np.abs(lhs))
                if (np.any(lhs < row_lower - row_tolerance) or np.any(lhs > row_upper + row_tolerance)
                        or np.any(x < lower - tolerance) or np.any(x > upper + tolerance)):
                    raise ValueError('Callback primal full matrix feasibility unverified')
                objective = float(np.dot(cost, x))
                if abs(objective - candidate['native_objective_min']) > tolerance:
                    raise ValueError('Callback primal objective unverified')
                incumbent({**candidate, 'status': 'VERIFIED_FEASIBLE_PRIMAL_NOT_OPTIMAL_CERTIFICATE',
                           'objective_min': objective, 'native_x': x,
                           'full_matrix_feasibility_checked': True})
            except Exception as error:
                callback_errors.append({'kind': 'INCUMBENT_WORKER', 'error_type': type(error).__name__, 'error': str(error)})
            finally:
                pending.task_done()
    worker = None
    if incumbent is not None:
        worker = threading.Thread(target=process_incumbents, name='native-incumbent-writer-225', daemon=True)
        worker.start()

    def callback(kind, message, data_out, data_in, user):
        # Do not modify data_in: observing does not request interrupts or feed
        # external solutions. Exceptions are retained outside the C callback.
        key = str(kind)
        event_counts[key] = event_counts.get(key, 0) + 1
        try:
            now = time.monotonic()
            improving = int(kind) == int(highspy.cb.HighsCallbackType.kCallbackMipImprovingSolution)
            if progress is not None and (improving or now - last_progress[0] >= 5):
                last_progress[0] = now
                progress({'scope': 'NATIVE_MIN_OBJECTIVE_COMPONENT_WITHOUT_BAD_CONSTANT',
                          'observer_version': VERSION, 'native_callback_kind': key,
                          'actual_native_callback_observed': True, 'native_version': solver.version(),
                          'native_primal_min': base._finite(data_out.mip_primal_bound),
                          'native_dual_min': base._finite(data_out.mip_dual_bound),
                          'native_relative_gap': base._finite(data_out.mip_gap),
                          'native_node_count': int(data_out.mip_node_count),
                          'native_elapsed_seconds': float(data_out.running_time)})
            if improving and incumbent is not None:
                pointer = data_out.mip_solution
                if not pointer:
                    raise ValueError('Improving solution callback has no native pointer')
                x = np.fromiter((float(pointer[index]) for index in range(len(cost))),
                                dtype=np.float64, count=len(cost))
                candidate = {'status': 'COPIED_NATIVE_PRIMAL_PENDING_WORKER_VERIFICATION',
                           'native_version': solver.version(), 'observer_version': VERSION,
                           'complete_native_model_hash': expected,
                           'native_objective_min': float(data_out.objective_function_value),
                           'native_dual_min_at_observation': base._finite(data_out.mip_dual_bound),
                           'native_relative_gap_at_observation': base._finite(data_out.mip_gap),
                           'native_elapsed_seconds': float(data_out.running_time),
                           'native_x': x, 'branch_and_bound_tree_checkpoint': False}
                try:
                    pending.put_nowait(candidate)
                except queue.Full:
                    try:
                        pending.get_nowait(); pending.task_done(); dropped[0] += 1
                    except queue.Empty:
                        pass
                    pending.put_nowait(candidate)
        except Exception as error:
            callback_errors.append({'kind': key, 'error_type': type(error).__name__, 'error': str(error)})

    if progress is not None or incumbent is not None:
        if solver.setCallback(callback, None) != highspy.HighsStatus.kOk:
            raise ValueError('Native observer callback rejected')
        kinds = [highspy.cb.HighsCallbackType.kCallbackMipInterrupt,
                 highspy.cb.HighsCallbackType.kCallbackMipImprovingSolution]
        if native_options['output_flag']:
            kinds.append(highspy.cb.HighsCallbackType.kCallbackMipLogging)
        for kind in kinds:
            if solver.startCallback(kind) != highspy.HighsStatus.kOk:
                raise ValueError('Native callback registration rejected: ' + str(kind))
            registered.append(str(kind))
    try:
        run_status = solver.run()
    finally:
        worker_stop.set()
        if worker is not None:
            worker.join(timeout=30)
            if worker.is_alive():
                raise RuntimeError('Incumbent writer did not reach a safe terminal state')
    native_status = solver.getModelStatus()
    info, solution = solver.getInfo(), solver.getSolution()
    if native_status == highspy.HighsModelStatus.kOptimal:
        status = 0
    elif native_status == highspy.HighsModelStatus.kInfeasible:
        status = 2
    elif native_status == highspy.HighsModelStatus.kUnbounded:
        status = 3
    elif any(native_status == getattr(highspy.HighsModelStatus, name, None) for name in
             ('kTimeLimit', 'kIterationLimit', 'kSolutionLimit', 'kInterrupt', 'kMemoryLimit')):
        status = 1
    else:
        status = 4
    valid = bool(solution.value_valid) and int(info.primal_solution_status) == int(highspy.SolutionStatus.kSolutionStatusFeasible)
    x = np.asarray(solution.col_value, np.float64) if valid else None
    if x is not None and x.shape != cost.shape:
        raise ValueError('Native witness length differs')
    result = OptimizeResult(status=status, success=status == 0, x=x,
        message='HiGHS ' + solver.version() + ': ' + solver.modelStatusToString(native_status),
        fun=base._finite(info.objective_function_value) if valid else None,
        mip_dual_bound=base._finite(info.mip_dual_bound), mip_gap=base._finite(info.mip_gap),
        mip_node_count=int(info.mip_node_count), native_run_status=str(run_status),
        native_model_status=str(native_status), native_backend_version=solver.version(),
        native_options_verified=native_options, native_model_verified=True,
        original_full_model_hash=expected, native_full_model_hash=actual,
        native_progress_callback_enabled=bool(registered), native_observer_version=VERSION,
        native_callback_registered_kinds=registered, native_callback_actual_event_counts=event_counts,
        native_callback_errors=callback_errors, native_unsaved_superseded_incumbents=dropped[0],
        native_solution_is_branch_tree_checkpoint=False)
    solver.clear()
    return result
