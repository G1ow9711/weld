"""Label-free runtime MILP and separate train-only latent graph optima.

Only role, path and canonical proposal ordinals define runtime feasibility.
Coordinates, labels, WIS, inside-cut rules and distance gates are absent.
SciPy/HiGHS solves sparse connected components; no candidate is truncated.
"""
from dataclasses import dataclass
from hashlib import sha256
from itertools import zip_longest
import json
import math
from numbers import Integral
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

VERSION = "boundary_incidence_sparse_milp_199_v1"
ROLES = ("start", "end", "cut")
ROLE_PAIRS = (("start", "end"), ("start", "cut"), ("cut", "cut"), ("cut", "end"))


class GraphSolverError(ValueError):
    pass


class UnknownSupervisionNotImplemented(NotImplementedError):
    pass


def _ordinal(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise GraphSolverError(name + ": nonnegative integer required")
    return int(value)


def _vector(value, length, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise GraphSolverError(name + ": finite detached vector with exact full count required")
    return array


@dataclass(frozen=True)
class GraphScores:
    node_scores: object
    child_scores: object
    duplicate_scores: object
    graph_fingerprint: str = ""


@dataclass(frozen=True)
class SolveOptions:
    time_limit_seconds: object = None
    mip_rel_gap: float = 0.0
    chunk_size: int = 65536
    max_variables: object = None
    certificate_tolerance: float = 1e-9

    def __post_init__(self):
        if self.time_limit_seconds is not None and (not math.isfinite(float(self.time_limit_seconds)) or self.time_limit_seconds < 0):
            raise GraphSolverError("nonnegative finite time limit required")
        if not math.isfinite(float(self.mip_rel_gap)) or self.mip_rel_gap < 0:
            raise GraphSolverError("nonnegative finite mip_rel_gap required")
        if _ordinal(self.chunk_size, "chunk_size") == 0:
            raise GraphSolverError("chunk_size must be positive")
        if self.max_variables is not None:
            _ordinal(self.max_variables, "max_variables")
        if not math.isfinite(float(self.certificate_tolerance)) or not 0 < self.certificate_tolerance <= 1e-4:
            raise GraphSolverError("certificate tolerance must be in(0,1e-4]")


@dataclass(frozen=True)
class TruthEvent:
    id: int
    role: str
    parent_id: int
    boundary_order: int


@dataclass(frozen=True)
class TruthChild:
    id: int
    start_event: int
    end_event: int


@dataclass(frozen=True)
class GraphTruth:
    node_event_index: object
    events: tuple
    true_children: tuple


@dataclass(frozen=True)
class GraphSelection:
    z: object
    y: object
    t: object
    chains: tuple
    selected_child_pairs: object


@dataclass(frozen=True)
class SolveResult:
    status: str
    optimal_proved: bool
    complete_graph_solution: bool
    selection: object
    base_score: object
    objective_max: object
    upper_bound_max: object
    mip_gap: object
    structural_loss: object
    component_receipts: tuple
    counts: dict
    loss_details: object = None
    message: str = ""
    graph_fingerprint: object = None


def query_graph_195_fingerprint(graph):
    """Match197 canonical binding, without importing torch or its network."""
    paths = []
    for path_id, context in sorted(graph.path_contexts.items()):
        paths.append(dict(path_id=int(path_id), query_count=int(context.query_count),
                          step_mm=float(context.step_mm), origin_mm=list(context.origin_mm),
                          direction=list(context.direction), raw_sha256=context.raw_sha256))
    payload = dict(schema="graph195_complete_runtime_v1", paths=paths, complete_hashes=graph.complete_hashes())
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf8")).hexdigest()


def _topology(graph):
    nodes = tuple(graph.nodes)
    paths = []
    roles = []
    for node in nodes:
        paths.append(_ordinal(node.path_id, "path_id"))
        if node.role not in ROLES:
            raise GraphSolverError("start/end/cut runtime role required")
        roles.append(node.role)
    path_groups = {path: np.asarray([i for i, p in enumerate(paths) if p == path], np.int64) for path in sorted(set(paths))}
    child_count = sum(sum(np.count_nonzero(np.asarray(roles, object)[ids] == a) * np.count_nonzero(np.asarray(roles, object)[ids] == b)
                          - (np.count_nonzero(np.asarray(roles, object)[ids] == a) if a == b else 0)
                          for a, b in ROLE_PAIRS) for ids in path_groups.values())
    dup_count = sum(len(ids) * (len(ids)-1)//2 for ids in path_groups.values())
    if graph.pair_count("child") != child_count or graph.pair_count("duplicate") != dup_count:
        raise GraphSolverError("full role-legal child/duplicate candidate count mismatch; truncation refused")
    return nodes, np.asarray(paths, np.int64), tuple(roles), path_groups, int(child_count), int(dup_count)


def _expected_pairs(groups, roles, kind, chunk_size):
    for ids in groups.values():
        role_ids = {r: np.asarray([i for i in ids if roles[i] == r], np.int64) for r in ROLES}
        blocks = [(role_ids[a], role_ids[b], a == b) for a, b in ROLE_PAIRS] if kind == "child" else [(ids, ids, True)]
        for source, target, may_self in blocks:
            if not len(source) or not len(target):
                continue
            for start in range(0, len(source)*len(target), chunk_size):
                linear = np.arange(start, min(start+chunk_size, len(source)*len(target)), dtype=np.int64)
                pairs = np.c_[source[linear//len(target)], target[linear % len(target)]]
                if kind == "duplicate":
                    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
                elif may_self:
                    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
                if len(pairs):
                    yield pairs


def _full_stream(graph, groups, roles, kind, chunk_size):
    actual = graph.iter_child_pairs(chunk_size) if kind == "child" else graph.iter_duplicate_pairs(chunk_size)
    count = 0
    for expected, provided in zip_longest(_expected_pairs(groups, roles, kind, chunk_size), actual):
        if expected is None or provided is None:
            raise GraphSolverError("incomplete or extra runtime " + kind + " blocks; no truncation allowed")
        provided = np.asarray(provided)
        if provided.dtype.kind not in "iu" or provided.shape != expected.shape or not np.array_equal(provided, expected):
            raise GraphSolverError("runtime " + kind + " iterator must follow full195 canonical path/role/source/target order")
        ids = np.arange(count, count+len(expected), dtype=np.int64)
        count += len(expected)
        yield ids, expected


def _truth(labels, roles):
    if not isinstance(labels, GraphTruth):
        raise GraphSolverError("train-only GraphTruth required")
    indexes = np.asarray(labels.node_event_index)
    if indexes.shape != (len(roles),) or indexes.dtype.kind not in "iu":
        raise GraphSolverError("full int node_event_index[N] required")
    if np.any(indexes == -2):
        raise UnknownSupervisionNotImplemented("PARTIAL_UNKNOWN_MODE_NOT_IMPLEMENTED: event unknown cannot become O/negative")
    if np.any(indexes < -1):
        raise GraphSolverError("event labels use >=0 or -1 known negative; -2 unknown rejected")
    events = {}
    parents = {}
    for event in labels.events:
        if not isinstance(event, TruthEvent):
            raise GraphSolverError("TruthEvent entries required")
        identity = _ordinal(event.id, "event id")
        _ordinal(event.parent_id, "truth parent id")
        _ordinal(event.boundary_order, "boundary order")
        if event.role not in ROLES or identity in events:
            raise GraphSolverError("unique truth event id with runtime role required")
        events[identity] = event
        parents.setdefault(event.parent_id, []).append(event)
    for i, identity in enumerate(indexes):
        if identity >= 0 and (int(identity) not in events or events[int(identity)].role != roles[i]):
            raise GraphSolverError("positive node event id absent or original event role differs")
    expected = set()
    for parent in parents.values():
        parent.sort(key=lambda e: e.boundary_order)
        if len(parent) < 2 or [e.boundary_order for e in parent] != list(range(len(parent))):
            raise GraphSolverError("truth parent needs contiguous original boundary order")
        if [e.role for e in parent] != ["start"] + ["cut"]*(len(parent)-2) + ["end"]:
            raise GraphSolverError("truth parent role syntax S,C*,E required")
        expected.update((a.id, b.id) for a, b in zip(parent[:-1], parent[1:]))
    children = {}
    child_pairs = {}
    for child in labels.true_children:
        if not isinstance(child, TruthChild):
            raise GraphSolverError("TruthChild entries required")
        identity = _ordinal(child.id, "truth child id")
        pair = (_ordinal(child.start_event, "truth start event"), _ordinal(child.end_event, "truth end event"))
        if identity in children or pair in child_pairs or pair not in expected:
            raise GraphSolverError("true children must be unique original same-parent adjacent event pairs")
        children[identity] = child
        child_pairs[pair] = identity
    if set(child_pairs) != expected:
        raise GraphSolverError("full truechild denominator required; missing adjacent truth children refused")
    return indexes.astype(np.int64, copy=False), events, children, child_pairs


class _Union:
    def __init__(self, count):
        self.parent = list(range(count))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def _components(groups, count, truth):
    union = _Union(count)
    for ids in groups.values():
        for i in ids[1:]:
            union.join(int(ids[0]), int(i))
    if truth is not None:
        indexes, events, children, child_pairs = truth
        representatives = {}
        for identity in events:
            ids = np.flatnonzero(indexes == identity)
            for i in ids[1:]:
                union.join(int(ids[0]), int(i))
            if len(ids):
                representatives[identity] = int(ids[0])
        for child in children.values():
            if child.start_event in representatives and child.end_event in representatives:
                union.join(representatives[child.start_event], representatives[child.end_event])
    output = {}
    for i in range(count):
        output.setdefault(union.find(i), []).append(i)
    components = [np.asarray(ids, np.int64) for ids in output.values()]
    components.sort(key=lambda ids: int(ids[0]))
    membership = np.empty(count, np.int64)
    for c, ids in enumerate(components):
        membership[ids] = c
    return components, membership


def _score_block(value, ids, pairs, total, name):
    if callable(value):
        out = _vector(value(ids.copy(), pairs.copy()), len(ids), name+" callback block")
    else:
        # Complete vectors were validated once at the interface. Rechecking an
        # entire O(M) vector for every block would accidentally become O(M^2).
        out = np.asarray(value, np.float64)[ids]
    if name == "duplicate_scores" and np.any(out < 0):
        raise GraphSolverError("learned duplicate penalty d must be nonnegative")
    return out


class _Rows:
    def __init__(self):
        self.row = []
        self.column = []
        self.value = []
        self.lower = []
        self.upper = []
        self.count = 0

    def add(self, columns, values, lower=-np.inf, upper=np.inf):
        columns = np.asarray(columns, np.int64).reshape(-1)
        values = np.asarray(values, np.float64).reshape(-1)
        if len(columns) != len(values):
            raise GraphSolverError("constraint coefficient length mismatch")
        self.row.append(np.full(len(columns), self.count, np.int64))
        self.column.append(columns)
        self.value.append(values)
        self.lower.append(float(lower))
        self.upper.append(float(upper))
        self.count += 1

    def batch(self, columns, values, lower=-np.inf, upper=np.inf):
        columns = np.asarray(columns, np.int64)
        if columns.ndim != 2:
            raise GraphSolverError("batched constraint columns must be [rows,width]")
        n, width = columns.shape
        if not n:
            return
        value = np.broadcast_to(np.asarray(values, np.float64), columns.shape)
        self.row.append(np.repeat(np.arange(self.count, self.count+n, dtype=np.int64), width))
        self.column.append(columns.reshape(-1))
        self.value.append(value.reshape(-1))
        self.lower.extend(np.broadcast_to(lower, (n,)).astype(float).tolist())
        self.upper.extend(np.broadcast_to(upper, (n,)).astype(float).tolist())
        self.count += n

    def constraint(self, variables):
        row = np.concatenate(self.row) if self.row else np.empty(0, np.int64)
        col = np.concatenate(self.column) if self.column else np.empty(0, np.int64)
        value = np.concatenate(self.value) if self.value else np.empty(0, float)
        matrix = coo_matrix((value, (row, col)), shape=(self.count, variables)).tocsc()
        return LinearConstraint(matrix, np.asarray(self.lower), np.asarray(self.upper))


def _build_component(ids, edge_ids, edge_pairs, a, dup_ids, dup_pairs, d, node_u, roles, truth, mode):
    n, m, k = len(ids), len(edge_ids), len(dup_ids)
    index = {int(global_id): local for local, global_id in enumerate(ids)}
    child = np.asarray([[index[int(s)], index[int(t)]] for s, t in edge_pairs], np.int64).reshape(-1, 2)
    duplicate = np.asarray([[index[int(s)], index[int(t)]] for s, t in dup_pairs], np.int64).reshape(-1, 2)
    cut_ids = [i for i, identity in enumerate(ids) if roles[identity] == "cut"]
    rank_index = {i: n+m+k+j for j, i in enumerate(cut_ids)}
    variables = n+m+k+len(cut_ids)
    group_events = {}
    group_children = {}
    if truth is not None:
        indexes, events, children, child_pairs = truth
        for identity in events:
            local = np.flatnonzero(indexes[ids] == identity)
            if len(local):
                group_events[identity] = local
        for e, (s, t) in enumerate(edge_pairs):
            identity = child_pairs.get((int(indexes[s]), int(indexes[t])))
            if identity is not None:
                group_children.setdefault(identity, []).append(e)
    event_or = {}
    child_or = {}
    if mode == "bad":
        for identity in group_events:
            event_or[identity] = variables
            variables += 1
        for identity in group_children:
            child_or[identity] = variables
            variables += 1
    c = np.zeros(variables, np.float64)
    c[:n] = -node_u[ids]
    c[n:n+m] = -a
    c[n+m:n+m+k] = d
    lower = np.zeros(variables)
    upper = np.ones(variables)
    integrality = np.ones(variables, np.uint8)
    for column in rank_index.values():
        upper[column] = max(0, len(cut_ids)-1)
        integrality[column] = 0
    rows = _Rows()
    outgoing = [[] for _ in range(n)]
    incoming = [[] for _ in range(n)]
    for e, (s, t) in enumerate(child):
        outgoing[s].append(n+e)
        incoming[t].append(n+e)
    for local, identity in enumerate(ids):
        if roles[identity] in ("start", "cut"):
            rows.add([local]+outgoing[local], [-1]+[1]*len(outgoing[local]), 0, 0)
        if roles[identity] in ("end", "cut"):
            rows.add([local]+incoming[local], [-1]+[1]*len(incoming[local]), 0, 0)
    y = n+np.arange(m, dtype=np.int64)
    rows.batch(np.c_[y, child[:, 0]], [1, -1], upper=0)
    rows.batch(np.c_[y, child[:, 1]], [1, -1], upper=0)
    t = n+m+np.arange(k, dtype=np.int64)
    rows.batch(np.c_[t, duplicate[:, 0]], [1, -1], upper=0)
    rows.batch(np.c_[t, duplicate[:, 1]], [1, -1], upper=0)
    rows.batch(np.c_[t, duplicate], [-1, 1, 1], upper=1)
    cc = np.asarray([e for e, (s, target) in enumerate(child) if s in rank_index and target in rank_index], np.int64)
    if len(cc):
        columns = np.asarray([[rank_index[int(child[e, 0])], rank_index[int(child[e, 1])], n+int(e)] for e in cc], np.int64)
        rows.batch(columns, [1, -1, len(cut_ids)], upper=len(cut_ids)-1)
    if mode == "good":
        indexes, events, children, child_pairs = truth
        upper[:n][indexes[ids] == -1] = 0
        good_edges = {e for group in group_children.values() for e in group}
        for e in range(m):
            if e not in good_edges:
                upper[n+e] = 0
        for group in group_events.values():
            rows.add(group, np.ones(len(group)), 1, 1)
        for group in group_children.values():
            rows.add(n+np.asarray(group), np.ones(len(group)), 1, 1)
    elif mode == "bad":
        c[n:n+m] -= 1  # selected child count; FN/FP uses identity OR below.
        for identity, group in group_children.items():
            v = child_or[identity]
            columns = n+np.asarray(group, np.int64)
            c[v] += 2
            rows.batch(np.c_[np.full(len(group), v), columns], [1, -1], lower=0)
            rows.add([v]+columns.tolist(), [1]+[-1]*len(group), upper=0)
        for identity, group in group_events.items():
            w = event_or[identity]
            c[group] -= 1
            c[w] += 1
            rows.batch(np.c_[np.full(len(group), w), group], [1, -1], lower=0)
            rows.add([w]+group.tolist(), [1]+[-1]*len(group), upper=0)
    return c, integrality, Bounds(lower, upper), rows.constraint(variables), dict(
        binary_variables=int(np.count_nonzero(integrality)), continuous_MTZ_variables=len(cut_ids),
        variables=variables, constraints=rows.count, children=m, duplicate_pairs=k,
        event_OR_variables=len(event_or), child_OR_variables=len(child_or))


def _chains(roles, selected_pairs, selected_z):
    incoming = {}
    outgoing = {}
    for source, target in selected_pairs:
        source, target = int(source), int(target)
        if source in outgoing or target in incoming:
            raise GraphSolverError("solver selected branching/reused boundary")
        outgoing[source] = target
        incoming[target] = source
    used = set(incoming) | set(outgoing)
    if used != set(np.flatnonzero(selected_z).tolist()):
        raise GraphSolverError("selected z differs from child endpoints")
    for identity in used:
        pair = (identity in incoming, identity in outgoing)
        required = (False, True) if roles[identity] == "start" else ((True, False) if roles[identity] == "end" else (True, True))
        if pair != required:
            raise GraphSolverError("solver role degree violation")
    chains = []
    visited = set()
    for start in sorted(i for i in used if roles[i] == "start"):
        chain = [start]
        while roles[chain[-1]] != "end":
            target = outgoing[chain[-1]]
            if target in chain or target in visited:
                raise GraphSolverError("selected graph cycle or reused vertex")
            chain.append(target)
        visited.update(chain)
        chains.append(tuple(chain))
    if visited != used:
        raise GraphSolverError("selected detached cut cycle")
    return tuple(chains)


def _loss(truth, selection):
    indexes, events, children, child_pairs = truth
    represented = set()
    for source, target in selection.selected_child_pairs:
        identity = child_pairs.get((int(indexes[source]), int(indexes[target])))
        if identity is not None:
            represented.add(identity)
    selected_event_ids = [int(indexes[i]) for i in np.flatnonzero(selection.z) if indexes[i] >= 0]
    fn = len(children)-len(represented)
    fp = int(np.count_nonzero(selection.y))-len(represented)
    duplicate = len(selected_event_ids)-len(set(selected_event_ids))
    if min(fn, fp, duplicate) < 0:
        raise GraphSolverError("invalid negative identity cost")
    return fn+fp+duplicate, dict(FN_true_child=fn, FP_child=fp, duplicate_true_event=duplicate,
        truechild_denominator=len(children), coefficients=dict(FN=1, FP=1, duplicate=1),
        correct_truechild_ids=sorted(represented), selected_positive_event_nodes=len(selected_event_ids),
        distinct_selected_true_events=len(set(selected_event_ids)))


def _number(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _failure(status, counts, message, components=()):
    return SolveResult(status, False, False, None, None, None, None, None, None, tuple(components), counts,
                       message=message, graph_fingerprint=counts.get("graph_fingerprint"))


def _solve(graph, scores, options, labels, mode):
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
    base_variables = n+m+k+sum(role == "cut" for role in roles)
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
    try:
        for kind, total, provider in (("child", m, a_provider), ("duplicate", k, d_provider)):
            seen = 0
            for pair_ids, pairs in _full_stream(graph, groups, roles, kind, options.chunk_size):
                values = _score_block(provider, pair_ids, pairs, total, "child_scores" if kind == "child" else "duplicate_scores")
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
            c, integer, bounds, constraint, size = _build_component(ids, edge_ids, pairs, a, dup_ids, duplicate_pairs, d, u, roles, truth, mode)
        except MemoryError:
            return _failure("NOT_SOLVED_RESOURCE_ERROR", counts, "Sparse full component allocation failed; no truncation", receipts)
        remaining = None if options.time_limit_seconds is None else max(0., float(options.time_limit_seconds)-(time.monotonic()-started))
        opt = dict(mip_rel_gap=float(options.mip_rel_gap), presolve=True)
        if remaining is not None:
            opt["time_limit"] = remaining
        receipt = dict(component=component, node_ordinals=ids.tolist(), path_ids=sorted(set(paths[ids].tolist())),
                       **size, time_limit_remaining_seconds=remaining, GT_parameters_used=(mode != "runtime"),
                       sparse_constraint_nonzeros=int(constraint.A.nnz))
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
            primal_min = float(np.dot(c, x))
            dual = receipt["raw_min_dual_bound"]
            gap = receipt["raw_mip_gap"]
            certificate = (raw_status == 0 and dual is not None and gap is not None
                           and gap <= options.certificate_tolerance
                           and abs(primal_min-dual) <= options.certificate_tolerance*(1+abs(primal_min)))
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
                       and abs(upper-objective) <= options.certificate_tolerance*(1+abs(objective)))
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
