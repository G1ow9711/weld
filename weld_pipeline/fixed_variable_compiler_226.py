"""Exact fixed-variable substitution; full model and witness remain authoritative."""
from dataclasses import dataclass
import io

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult

from weld_pipeline import highs_backend_215 as original
from weld_pipeline import highs_observed_backend_225 as native

VERSION = 'EXACT_EXPLICIT_FIXED_AND_ZERO_PRODUCT_SUBSTITUTION_226_v1'


@dataclass
class CompiledProblem:
    cost: np.ndarray
    integer: np.ndarray
    bounds: Bounds
    rows: LinearConstraint
    retained_columns: np.ndarray
    fixed_columns: np.ndarray
    fixed_values: np.ndarray
    retained_rows: np.ndarray
    constant_rows: np.ndarray
    row_shift: np.ndarray
    objective_constant: float
    original_column_count: int
    original_model_hash: str
    compiled_model_hash: str
    zero_product_proof_rows: np.ndarray

    def lift(self, value):
        value = np.asarray(value, np.float64)
        if value.shape != self.cost.shape or not np.isfinite(value).all():
            raise ValueError('Complete finite compiled witness required')
        full = np.empty(self.original_column_count, np.float64)
        full[self.retained_columns] = value
        full[self.fixed_columns] = self.fixed_values
        return full

    def mapping_bytes(self):
        stream = io.BytesIO()
        np.savez_compressed(stream, retained_columns=self.retained_columns, fixed_columns=self.fixed_columns,
            fixed_values=self.fixed_values, retained_rows=self.retained_rows, constant_rows=self.constant_rows,
            row_shift=self.row_shift, zero_product_proof_rows=self.zero_product_proof_rows,
            objective_constant=np.asarray(self.objective_constant, np.float64))
        return stream.getvalue()

    def receipt(self):
        return {'version': VERSION, 'relation': 'EXACT_FIXED_VARIABLE_SUBSTITUTION',
                'original_full_model_hash': self.original_model_hash,
                'compiled_native_model_hash': self.compiled_model_hash,
                'full_and_compiled_model_byte_identical': self.original_model_hash == self.compiled_model_hash,
                'original_variables': self.original_column_count, 'compiled_variables': len(self.cost),
                'compiled_binary_variables': int(np.count_nonzero(self.integer)),
                'fixed_columns': len(self.fixed_columns), 'constant_rows_removed': len(self.constant_rows),
                'compiled_rows': self.rows.A.shape[0], 'compiled_nonzeros': int(self.rows.A.nnz),
                'zero_product_proofs': len(self.zero_product_proof_rows),
                'objective_constant_added_to_primal_and_dual_once': self.objective_constant,
                'candidate_graph_pruned': False, 'neural_scores_pruned': False}


def compile_problem(cost, integer, bounds, rows, *, zero_product_rules=(), preserve_columns=()):
    cost = np.ascontiguousarray(cost, np.float64)
    integer = np.asarray(integer, np.uint8)
    lower, upper = np.asarray(bounds.lb, np.float64), np.asarray(bounds.ub, np.float64)
    matrix = rows.A.tocsc(copy=True)
    matrix.sum_duplicates(); matrix.eliminate_zeros(); matrix.sort_indices()
    if (cost.ndim != 1 or integer.shape != cost.shape or lower.shape != cost.shape
            or upper.shape != cost.shape or not np.isfinite(cost).all() or not np.isfinite(matrix.data).all()):
        raise ValueError('Complete finite mathematical instance required')
    if np.any(lower > upper) or np.any((integer != 0) & (integer != 1)):
        raise ValueError('Invalid bounds or variable types')
    fixed = (lower == upper) & np.isfinite(lower)
    fixed[np.asarray(preserve_columns, np.int64)] = False
    values = lower.copy()
    csr = matrix.tocsr()
    proof_rows = []
    for product, left, right in zero_product_rules:
        product, left, right = int(product), int(left), int(right)
        if any(index < 0 or index >= len(cost) for index in (product, left, right)):
            raise ValueError('Zero-product rule has invalid original columns')
        zero_nodes = [node for node in (left, right) if fixed[node] and values[node] == 0.]
        if not zero_nodes:
            continue
        if lower[product] != 0 or upper[product] < 0:
            raise ValueError('Zero-product proof requires the original nonnegative product bound')
        proof = None
        # Verify an actual original McCormick row t-z<=0. A rule name alone
        # cannot fix a variable; no approximate equality or singleton pruning.
        candidates = matrix.indices[matrix.indptr[product]:matrix.indptr[product + 1]]
        for row in candidates:
            start, end = csr.indptr[row], csr.indptr[row + 1]
            if end - start != 2 or rows.lb[row] != -np.inf or rows.ub[row] != 0.:
                continue
            coefficients = dict(zip(csr.indices[start:end].tolist(), csr.data[start:end].tolist()))
            if coefficients.get(product) == 1. and any(coefficients.get(node) == -1. for node in zero_nodes):
                proof = int(row)
                break
        if proof is None:
            raise ValueError('Original McCormick zero-product implication is missing')
        fixed[product], values[product] = True, 0.
        proof_rows.append((product, proof))
    fixed_columns, retained = np.flatnonzero(fixed), np.flatnonzero(~fixed)
    fixed_values = values[fixed_columns]
    if np.any(integer[fixed_columns].astype(bool) & (fixed_values != np.rint(fixed_values))):
        raise ValueError('Explicit fixed binary value is nonintegral')
    shift = np.asarray(matrix[:, fixed_columns] @ fixed_values).reshape(-1)
    if not np.isfinite(shift).all():
        raise ValueError('Nonfinite fixed substitution shift')
    reduced = matrix[:, retained].tocsr()
    reduced.sum_duplicates(); reduced.eliminate_zeros(); reduced.sort_indices()
    zero_rows = np.diff(reduced.indptr) == 0
    row_lower, row_upper = np.asarray(rows.lb) - shift, np.asarray(rows.ub) - shift
    if np.any(zero_rows & ((row_lower > 0.) | (row_upper < 0.))):
        raise ValueError('INFEASIBLE_CONSTANT_ROW_AFTER_EXACT_FIXED_SUBSTITUTION')
    retained_rows, constant_rows = np.flatnonzero(~zero_rows), np.flatnonzero(zero_rows)
    reduced_rows = LinearConstraint(reduced[retained_rows].tocsc(), row_lower[retained_rows], row_upper[retained_rows])
    reduced_bounds = Bounds(lower[retained], upper[retained])
    constant = float(np.dot(cost[fixed_columns], fixed_values))
    original_hash = original.model_digest(cost, integer, lower, upper, rows.lb, rows.ub, matrix)
    compiled_hash = original.model_digest(cost[retained], integer[retained], reduced_bounds.lb,
        reduced_bounds.ub, reduced_rows.lb, reduced_rows.ub, reduced_rows.A)
    return CompiledProblem(cost[retained], integer[retained], reduced_bounds, reduced_rows,
        retained, fixed_columns, fixed_values, retained_rows, constant_rows, shift, constant,
        len(cost), original_hash, compiled_hash, np.asarray(proof_rows, np.int64).reshape(-1, 2))


def solve_compiled(problem, options=None):
    if not isinstance(problem, CompiledProblem):
        raise TypeError('Verified compiled problem required')
    if len(problem.cost) == 0:
        return OptimizeResult(status=0, success=True, x=problem.lift(np.empty(0)),
            fun=problem.objective_constant, mip_dual_bound=problem.objective_constant, mip_gap=0.,
            message='All variables explicitly fixed; constant rows verified, no native call',
            compiled_problem=problem, native_solver_called=False)
    options = dict(options) if options is not None else {
        'mip_rel_gap': 0., 'mip_abs_gap': 0., 'mip_feasibility_tolerance': 1e-9,
        'presolve': True, 'threads': 2, 'parallel': False}
    result = native.milp(problem.cost, integrality=problem.integer, bounds=problem.bounds,
                         constraints=problem.rows, options=options)
    result.native_solver_called = True
    result.native_original_full_model_hash = problem.original_model_hash
    result.native_compiled_model_hash = problem.compiled_model_hash
    result.native_raw_primal_compiled = result.fun
    result.native_raw_dual_compiled = result.mip_dual_bound
    if result.x is not None:
        result.x = problem.lift(result.x)
    if result.fun is not None:
        result.fun += problem.objective_constant
    if result.mip_dual_bound is not None:
        result.mip_dual_bound += problem.objective_constant
    result.compiled_problem = problem
    return result
