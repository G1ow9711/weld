"""F1 exact sparse formulation. Frozen 199 supplies unchanged graph semantics.

Every original candidate, coefficient, identity cost and graph constraint stays.
Only redundant integrality and proven per-path counting constraints change.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import hstack, vstack, csc_matrix
from . import structured_graph_solver_v1 as baseline
from .structured_graph_solver_v1 import (
    GraphScores, GraphTruth, TruthEvent, TruthChild, GraphSelection, SolveResult,
    GraphSolverError, UnknownSupervisionNotImplemented, _ordinal, _vector,
    _topology, _truth, _components, _full_stream, _score_block, _Rows,
    _chains, _loss, _number, _failure, query_graph_195_fingerprint)

VERSION = 'F1_EXACT_SPARSE_213_v1'
EXPECTED_BASE_SHA = 'd0d0537aaac722f61a326f7055975485f449e372550c5905e6a2c6afea1c1149'


@dataclass(frozen=True)
class SolveOptions(baseline.SolveOptions):
    mip_abs_gap: float = 0.0
    threads: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.mip_rel_gap != 0.0 or self.mip_abs_gap != 0.0:
            raise GraphSolverError('F1 requires relative and absolute MIP gaps exactly zero')
        if _ordinal(self.threads, 'threads') == 0:
            raise GraphSolverError('positive solver thread count required')
        if self.certificate_tolerance < 1e-10:
            raise GraphSolverError('HiGHS feasibility tolerance below supported minimum refused')


def _guard_base_source():
    if sha256(Path(baseline.__file__).read_bytes()).hexdigest() != EXPECTED_BASE_SHA:
        raise GraphSolverError('Frozen 199 source binding changed')


def _array_digest(digest, name, value):
    array = np.ascontiguousarray(value)
    digest.update(name.encode('ascii'))
    digest.update(json.dumps([array.dtype.str, list(array.shape)], separators=(',', ':')).encode('ascii'))
    digest.update(array.tobytes())


def _objective_digest(u, truth, mode, child_count, duplicate_count):
    digest = sha256(b'S=uz+ay-dt;DELTA=FNchild+FPchild+duplicateevent;known-v1;')
    digest.update(mode.encode('ascii'))
    _array_digest(digest, 'u_float64', np.asarray(u, '<f8'))
    digest.update(json.dumps([child_count, duplicate_count], separators=(',', ':')).encode('ascii'))
    if truth is not None:
        indexes, events, children, _ = truth
        _array_digest(digest, 'node_event_index', np.asarray(indexes, '<i8'))
        payload = {'events':[(e.id,e.role,e.parent_id,e.boundary_order) for e in events.values()],
                   'children':[(c.id,c.start_event,c.end_event) for c in children.values()],
                   'bad_objective_constant':len(children) if mode == 'bad' else 0}
        digest.update(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8'))
    return digest


def _formulation_digest(c, integer, bounds, constraint):
    digest = sha256(VERSION.encode('ascii'))
    matrix = constraint.A.tocsc()
    for name, values in (('objective', c), ('integrality', integer), ('lower', bounds.lb),
                         ('upper', bounds.ub), ('row_lower', constraint.lb), ('row_upper', constraint.ub),
                         ('CSC_values', matrix.data), ('CSC_indices', matrix.indices), ('CSC_indptr', matrix.indptr)):
        _array_digest(digest, name, values)
    return digest.hexdigest()


def _build_component(ids, edge_ids, edge_pairs, a, dup_ids, dup_pairs, d, node_u, roles, truth, mode, paths):
    c0, integer0, bounds0, original, size0 = baseline._build_component(
        ids, edge_ids, edge_pairs, a, dup_ids, dup_pairs, d, node_u, roles, truth, mode)
    n, m, pair_count = len(ids), len(edge_ids), len(dup_ids)
    local_index = {int(identity):ordinal for ordinal, identity in enumerate(ids)}
    by_path = {int(path):np.flatnonzero(paths[ids] == path) for path in sorted(set(paths[ids].tolist()))}
    variable_count = len(c0) + 2*len(by_path)
    c = np.r_[c0, np.zeros(2*len(by_path))]
    integer = np.r_[integer0, np.zeros(2*len(by_path), np.uint8)].astype(np.uint8)
    integer[n+m:n+m+pair_count] = 0
    lo = np.r_[bounds0.lb, np.zeros(2*len(by_path))]
    hi = np.r_[bounds0.ub, np.zeros(2*len(by_path))]
    extras = _Rows()
    product_start = n+m
    registry = {'product_start':product_start, 'product_count':pair_count, 'OR':[], 'KT':[]}
    guards = []

    if mode == 'bad':
        indexes, events, _, child_pairs = truth
        column = n+m+pair_count+size0['continuous_MTZ_variables']
        for identity in events:
            members = np.flatnonzero(indexes[ids] == identity)
            if len(members):
                integer[column] = 0
                registry['OR'].append((column, members))
                column += 1
        children = {}
        for edge, (s, target) in enumerate(edge_pairs):
            identity = child_pairs.get((int(indexes[s]), int(indexes[target])))
            if identity is not None:
                children.setdefault(identity, []).append(n+edge)
        for members in children.values():
            integer[column] = 0
            registry['OR'].append((column, np.asarray(members, np.int64)))
            column += 1
        if column != len(c0):
            raise GraphSolverError('Original OR registry no longer matches frozen construction')

    full_pairs = np.asarray(dup_pairs, np.int64).reshape(-1, 2)
    covered_pairs = 0
    for ordinal, (path, members) in enumerate(by_path.items()):
        global_members = ids[members]
        selected_pair_ordinals = np.flatnonzero(paths[full_pairs[:,0]] == path) if pair_count else np.empty(0, np.int64)
        pairs = full_pairs[selected_pair_ordinals]
        expected = len(members)*(len(members)-1)//2
        if len(pairs) != expected or (len(pairs) and (np.any(paths[pairs[:,1]] != path) or np.any(pairs[:,0] >= pairs[:,1]))):
            raise GraphSolverError('K/T guard: incomplete, cross-path or noncanonical all-pair pool')
        if len(pairs):
            keys = pairs[:,0]*len(paths)+pairs[:,1]
            if len(np.unique(keys)) != expected:
                raise GraphSolverError('K/T guard: repeated unordered pair or missing pair')
            if not np.all(np.isin(pairs, global_members)):
                raise GraphSolverError('K/T guard: pair endpoint outside complete path')
        covered_pairs += len(pairs)
        K, T = len(c0)+2*ordinal, len(c0)+2*ordinal+1
        hi[K], hi[T] = len(members), expected
        extras.add([K]+members.tolist(), [1]+[-1]*len(members), 0, 0)
        extras.add([T]+(product_start+selected_pair_ordinals).tolist(), [1]+[-1]*len(pairs), 0, 0)
        registry['KT'].append((K, T, members))
        fixed = False
        event_count = None
        reason = 'runtime_or_bad_no_GT_fixed_count'
        if mode == 'good':
            indexes, events, _, _ = truth
            local_labels = indexes[global_members]
            event_ids = sorted(set(int(label) for label in local_labels if label >= 0))
            covered = set()
            complete = not np.any(local_labels < -1)
            for identity in event_ids:
                alternatives = np.flatnonzero(indexes == identity)
                if identity not in events or not len(alternatives) or not np.all(paths[alternatives] == path):
                    complete = False
                    break
                if covered.intersection(alternatives.tolist()):
                    complete = False
                    break
                covered.update(alternatives.tolist())
            allowed = set(int(ids[member]) for member in members if bounds0.ub[member] > 0)
            complete = complete and covered == allowed
            if complete:
                event_count = len(event_ids)
                lo[K] = hi[K] = event_count
                lo[T] = hi[T] = event_count*(event_count-1)//2
                fixed = True
                reason = 'complete_disjoint_covering_same_path_one_of_groups'
            else:
                reason = 'guard_not_met_keep_coupled_solver_and_generic_cuts'
        if not fixed:
            # k=0 is retained without an explicit zero K coefficient.
            if len(members):
                extras.add([T], [-1], upper=0)
            for tangent in range(1, len(members)):
                extras.add([K,T], [tangent,-1], upper=tangent*(tangent+1)//2)
        guards.append({'path_id':path, 'nodes':len(members), 'duplicate_pairs':len(pairs),
                       'complete_all_pair_guard':True, 'Good_fixed_KT':fixed,
                       'fixed_event_count':event_count, 'reason':reason})
    if covered_pairs != pair_count:
        raise GraphSolverError('K/T guard: unassigned or extra pair')
    extra_constraint = extras.constraint(variable_count)
    original_extended = hstack([original.A,csc_matrix((original.A.shape[0],2*len(by_path)))],format='csc')
    matrix = vstack([original_extended,extra_constraint.A],format='csc')
    constraint = LinearConstraint(matrix, np.r_[original.lb,extra_constraint.lb], np.r_[original.ub,extra_constraint.ub])
    size = dict(size0, variables=variable_count, binary_variables=n+m,
                continuous_product_variables=pair_count,
                continuous_OR_variables=len(registry['OR']), continuous_KT_variables=2*len(by_path),
                constraints=matrix.shape[0], original_constraints=original.A.shape[0],
                per_path_KT_guards=guards, formulation_version=VERSION,
                formulation_hash=_formulation_digest(c,integer,Bounds(lo,hi),constraint))
    if np.count_nonzero(integer) != n+m:
        raise GraphSolverError('Only original z/y may remain binary in F1')
    size['_witness_registry'] = registry
    return c, integer, Bounds(lo,hi), constraint, size


def _rebuild_auxiliary_witness(x, ids, duplicate_pairs, registry, c, bounds, constraint, tolerance):
    rebuilt = x.copy()
    n = len(ids)
    product_start, product_count = registry['product_start'], registry['product_count']
    rebuilt[:product_start] = np.rint(x[:product_start])
    index = {int(identity):ordinal for ordinal, identity in enumerate(ids)}
    z = rebuilt[:n].astype(bool)
    products = np.asarray([z[index[int(s)]] and z[index[int(t)]] for s,t in duplicate_pairs], float)
    rebuilt[product_start:product_start+product_count] = products
    for column, members in registry['OR']:
        rebuilt[column] = int(np.any(rebuilt[members]))
    for K,T,members in registry['KT']:
        count = int(z[members].sum())
        rebuilt[K], rebuilt[T] = count, count*(count-1)//2
    auxiliary = np.r_[np.arange(product_start,product_start+product_count),
                      [column for column,_ in registry['OR']],
                      [column for K,T,_ in registry['KT'] for column in (K,T)]].astype(np.int64)
    residual = float(np.max(np.abs(x[auxiliary]-rebuilt[auxiliary]),initial=0))
    objective_error = abs(float(c@x)-float(c@rebuilt))
    lhs = np.asarray(constraint.A@rebuilt)
    row_tolerance = tolerance*(1+np.abs(lhs))
    if residual > tolerance or objective_error > tolerance:
        raise GraphSolverError('Continuous auxiliary witness/objective residual exceeds strict certificate tolerance')
    if np.any(lhs < constraint.lb-row_tolerance) or np.any(lhs > constraint.ub+row_tolerance) or np.any(rebuilt < bounds.lb-tolerance) or np.any(rebuilt > bounds.ub+tolerance):
        raise GraphSolverError('Rebuilt discrete witness violates original or added constraints')
    return rebuilt, {'maximum_auxiliary_residual':residual,'complete_objective_residual':objective_error,
                     'original_constraints_rechecked':True,'auxiliary_rebuilt_from_checked_binary_z_y':True}


def _solve(graph, scores, options, labels, mode):
    _guard_base_source()
    started = time.monotonic()
    options = options or SolveOptions()
    if not isinstance(options, SolveOptions) or not isinstance(scores, GraphScores):
        raise GraphSolverError("GraphScores and SolveOptions required")
    nodes, paths, roles, groups, m, k = _topology(graph)
    n = len(nodes)
    u = _vector(scores.node_scores, n, "node_scores")
    a_provider = scores.child_scores if callable(scores.child_scores) else _vector(scores.child_scores, m, "child_scores")
    d_provider = scores.duplicate_scores if callable(scores.duplicate_scores) else _vector(scores.duplicate_scores, k, "duplicate_scores")
    if not callable(d_provider):
        if np.any(d_provider < 0):
            raise GraphSolverError("learned duplicate penalty d must be nonnegative")
    truth = _truth(labels, roles) if mode != "runtime" else None
    counts = dict(nodes=n, children=m, duplicate_pairs=k, true_children=len(truth[2]) if truth else None,
                  full_candidate_space_defined=True, full_candidate_enumeration=False, truncated=False, mode=mode)
    fingerprint = query_graph_195_fingerprint(graph) if hasattr(graph, "complete_hashes") and hasattr(graph, "path_contexts") else None
    if scores.graph_fingerprint:
        if len(scores.graph_fingerprint) != 64 or any(c not in "0123456789abcdef" for c in scores.graph_fingerprint):
            raise GraphSolverError("nonempty graph fingerprint must be lowercase SHA256")
        if fingerprint is None or scores.graph_fingerprint != fingerprint:
            raise GraphSolverError("score/candidate full graph fingerprint differs")
    counts["graph_fingerprint"] = fingerprint
    base_variables = n+m+k+sum(role == "cut" for role in roles)+2*len(groups)
    if mode == "bad":
        base_variables += len(truth[1])+len(truth[2])
    counts["preflight_variable_upper_count"] = base_variables
    if options.max_variables is not None and base_variables > options.max_variables:
        return _failure("NOT_SOLVED_RESOURCE_LIMIT", counts, "Explicit variable budget exceeded; full graph not truncated")
    if mode == "good":
        missing_events = [identity for identity in truth[1] if not np.any(truth[0] == identity)]
        if missing_events:
            return _failure("INFEASIBLE", counts, "Full truth events have no runtime alternative: " + str(missing_events))
    components, membership = _components(groups, n, truth)
    storage = [dict(children=[], duplicate=[]) for _ in components]
    objective_digest = _objective_digest(u, truth, mode, m, k)
    try:
        for kind, total, provider in (("child", m, a_provider), ("duplicate", k, d_provider)):
            objective_digest.update(kind.encode('ascii'))
            seen = 0
            for pair_ids, pairs in _full_stream(graph, groups, roles, kind, options.chunk_size):
                values = _score_block(provider, pair_ids, pairs, total, "child_scores" if kind == "child" else "duplicate_scores")
                objective_digest.update(np.ascontiguousarray(values, dtype='<f8').tobytes())
                if not np.all(membership[pairs[:, 0]] == membership[pairs[:, 1]]):
                    raise GraphSolverError("component partition dropped coupling")
                component = int(membership[pairs[0, 0]])
                if not np.all(membership[pairs[:, 0]] == component):
                    raise GraphSolverError("canonical chunk crosses component unexpectedly")
                storage[component]["children" if kind == "child" else "duplicate"].append((pair_ids, pairs, values))
                seen += len(pairs)
            if seen != total:
                raise GraphSolverError("full candidate stream count mismatch")
        counts["full_candidate_enumeration"] = True
        counts["objective_snapshot_hash"] = objective_digest.hexdigest()
        counts["graph_content_hash"] = fingerprint or sha256(json.dumps({"paths":paths.tolist(),"roles":roles,"full_children":m,"full_pairs":k},separators=(",",":")).encode()).hexdigest()
        counts["graph_hash_scope"] = "graph195_complete" if fingerprint else "canonical_role_path_topology_fixture"
        counts["formulation_version"] = VERSION
    except MemoryError:
        return _failure("NOT_SOLVED_RESOURCE_ERROR", counts, "Memory allocation failed while enumerating complete graph; no partial graph result")
    z = np.zeros(n, bool)
    y = np.zeros(m, bool)
    t = np.zeros(k, bool)
    selected_pairs = []
    receipts = []
    base_score = 0.
    bound_sum = 0.
    all_bounds = True
    all_optimal = True
    observed_limit = False
    for component, ids in enumerate(components):
        def unpack(key):
            blocks = storage[component][key]
            if not blocks:
                return np.empty(0, np.int64), np.empty((0, 2), np.int64), np.empty(0, float)
            out = tuple(np.concatenate([block[j] for block in blocks], axis=0) for j in range(3))
            storage[component][key] = []
            return out
        edge_ids, pairs, a = unpack("children")
        dup_ids, duplicate_pairs, d = unpack("duplicate")
        try:
            c, integer, bounds, constraint, size = _build_component(ids, edge_ids, pairs, a, dup_ids, duplicate_pairs, d, u, roles, truth, mode, paths)
        except MemoryError:
            return _failure("NOT_SOLVED_RESOURCE_ERROR", counts, "Sparse full component allocation failed; no truncation", receipts)
        witness_registry = size.pop("_witness_registry")
        remaining = None if options.time_limit_seconds is None else max(0., float(options.time_limit_seconds)-(time.monotonic()-started))
        opt = dict(mip_rel_gap=0.0, mip_abs_gap=0.0, mip_feasibility_tolerance=float(options.certificate_tolerance),
                   presolve=True, threads=int(options.threads), parallel=False)
        if remaining is not None:
            opt["time_limit"] = remaining
        receipt = dict(component=component, node_ordinals=ids.tolist(), path_ids=sorted(set(paths[ids].tolist())),
                       **size, time_limit_remaining_seconds=remaining, GT_parameters_used=(mode != "runtime"),
                       sparse_constraint_nonzeros=int(constraint.A.nnz), mode=mode,
                       graph_content_hash=counts["graph_content_hash"],objective_snapshot_hash=counts["objective_snapshot_hash"],
                       solver_options=opt.copy(),bad_objective_constant_full_graph=(len(truth[2]) if mode=="bad" else 0))
        if remaining == 0.:
            result = None
            raw_status = 1
            x = None
            receipt.update(raw_status=1, raw_message="Global time budget exhausted before this component", raw_mip_gap=None, raw_min_objective=None, raw_min_dual_bound=None)
        else:
            try:
                result = milp(c=c, integrality=integer, bounds=bounds, constraints=constraint, options=opt)
            except MemoryError:
                return _failure("NOT_SOLVED_RESOURCE_ERROR", counts, "HiGHS allocation failed; complete graph retained", receipts)
            except (ValueError, RuntimeError, OverflowError) as error:
                receipt.update(raw_status=None, raw_message=type(error).__name__+": "+str(error))
                receipts.append(receipt)
                return _failure("SOLVER_ERROR", counts, "HiGHS failed before providing a certificate", receipts)
            raw_status = int(result.status)
            x = getattr(result, "x", None)
            receipt.update(raw_status=raw_status, raw_message=str(result.message), raw_mip_gap=_number(getattr(result, "mip_gap", None)),
                           raw_min_objective=_number(getattr(result, "fun", None)), raw_min_dual_bound=_number(getattr(result, "mip_dual_bound", None)))
        if raw_status == 2:
            receipts.append(receipt)
            return _failure("INFEASIBLE", counts, "Component proved infeasible under full graph constraints", receipts)
        if x is None:
            if mode == "good":
                receipts.append(receipt)
                return _failure("NO_SOLUTION_LIMIT" if raw_status == 1 else "SOLVER_ERROR", counts, "No feasible good graph incumbent", receipts)
            # Runtime/bad empty graph is an explicit feasible fallback, never optimal.
            x = np.zeros(len(c))
            receipt["fallback_empty_graph"] = True
            all_optimal = False
            all_bounds = False
            observed_limit = observed_limit or raw_status == 1
        else:
            x = np.asarray(x, float)
            if x.shape != c.shape or not np.isfinite(x).all():
                raise GraphSolverError("invalid solver primal vector")
            binary = integer.astype(bool)
            if np.max(np.abs(x[binary]-np.rint(x[binary])), initial=0.) > options.certificate_tolerance:
                raise GraphSolverError("solver binary incumbent exceeds integrality tolerance")
            lhs = np.asarray(constraint.A @ x)
            tol = options.certificate_tolerance * (1+np.abs(lhs))
            if np.any(lhs < constraint.lb-tol) or np.any(lhs > constraint.ub+tol) or np.any(x < bounds.lb-options.certificate_tolerance) or np.any(x > bounds.ub+options.certificate_tolerance):
                raise GraphSolverError("solver incumbent fails sparse constraint/bound verification")
            x, auxiliary_checks = _rebuild_auxiliary_witness(x,ids,duplicate_pairs,witness_registry,c,bounds,constraint,options.certificate_tolerance)
            receipt.update(auxiliary_checks)
            primal_min = float(np.dot(c, x))
            dual = receipt["raw_min_dual_bound"]
            gap = receipt["raw_mip_gap"]
            certificate = (raw_status == 0 and dual is not None and gap is not None
                           and gap <= options.certificate_tolerance
                           and abs(primal_min-dual) <= options.certificate_tolerance
                           and receipt["raw_min_objective"] is not None
                           and abs(receipt["raw_min_objective"]-dual) <= options.certificate_tolerance)
            receipt["raw_absolute_gap"] = abs(receipt["raw_min_objective"]-dual) if receipt["raw_min_objective"] is not None and dual is not None else None
            receipt["recomputed_primal_min_objective"] = primal_min
            receipt["exact_zero_gap_certificate"] = bool(certificate)
            all_optimal = all_optimal and certificate
            observed_limit = observed_limit or raw_status == 1
            if dual is None:
                all_bounds = False
            else:
                bound_sum += -dual
        local_z = np.rint(x[:len(ids)]).astype(bool)
        local_y = np.rint(x[len(ids):len(ids)+len(edge_ids)]).astype(bool)
        local_t = np.rint(x[len(ids)+len(edge_ids):len(ids)+len(edge_ids)+len(dup_ids)]).astype(bool)
        z[ids] = local_z
        if not np.array_equal(local_t, z[duplicate_pairs[:, 0]] & z[duplicate_pairs[:, 1]]):
            raise GraphSolverError("selected product differs from z_i*z_j")
        y[edge_ids] = local_y
        t[dup_ids] = local_t
        selected_pairs.extend((int(edge_id), int(s), int(target)) for edge_id, (s, target), selected in zip(edge_ids, pairs, local_y) if selected)
        base_score += float(np.dot(u[ids], local_z)+np.dot(a, local_y)-np.dot(d, local_t))
        receipt["selected_node_count"] = int(local_z.sum())
        receipt["selected_child_count"] = int(local_y.sum())
        receipts.append(receipt)
    selected_pairs.sort()
    pair_array = np.asarray([(s, target) for _, s, target in selected_pairs], np.int64).reshape(-1, 2)
    chains = _chains(roles, pair_array, z)
    selection = GraphSelection(z, y, t, chains, pair_array)
    structural, details = _loss(truth, selection) if truth is not None else (None, None)
    if mode == "good" and structural != 0:
        raise GraphSolverError("good graph failed full identity-consistency verification")
    objective = base_score+(structural if mode == "bad" else 0)
    upper = bound_sum+(len(truth[2]) if mode == "bad" else 0) if all_bounds else None
    if n == 0:
        upper = objective
    if upper is not None and upper < objective-options.certificate_tolerance*(1+abs(objective)):
        raise GraphSolverError("maximization bound below feasible objective")
    gap = max(0., (upper-objective)/max(1., abs(objective))) if upper is not None else None
    all_optimal = bool(all_optimal and upper is not None and gap <= options.certificate_tolerance
                       and abs(upper-objective) <= options.certificate_tolerance)
    status = "OPTIMAL" if all_optimal else ("FEASIBLE_LIMIT" if observed_limit else "FEASIBLE_NOT_PROVED_OPTIMAL")
    return SolveResult(status, bool(all_optimal), True, selection, base_score, objective, upper, gap, structural,
                       tuple(receipts), counts, details,
                       "Complete role-legal runtime graph; selection geometry must be validated after solve, without silent deletion", fingerprint)

def solve_runtime_graph(graph, scores, options=None):
    """Runtime interface: no labels, GT event IDs or parent groups accepted."""
    return _solve(graph, scores, options, None, "runtime")

def solve_best_truth_consistent_graph(graph, scores, labels, options=None):
    """Train only: exactly one alternative query/event and edge/true child."""
    return _solve(graph, scores, options, labels, "good")

def solve_loss_augmented_graph(graph, scores, labels, options=None):
    """Train only: maximize score plus fixed identity FN/FP/duplicate cost."""
    return _solve(graph, scores, options, labels, "bad")
