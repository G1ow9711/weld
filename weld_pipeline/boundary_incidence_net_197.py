"""Standby boundary-incidence network and differentiable graph score.

No checkpoint/data loader, optimizer, teacher-node pretraining or decoder lives
here. Graph topology comes only from complete same-path runtime relations.
Runtime IDs group/index nodes; they never enter neural feature vectors.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


VERSION = "boundary_incidence_prepare_197_v1"
STATUS = "STANDBY_NO_TRAINING"
TRAINING_ALLOWED = False
FEATURE_COUNT = 425
INPUT_DIM = 442
NODE_WIDTH = 128
PAIR_DIM = 16
ROLE_NAMES = ("start", "end", "cut")
CHILD_ROLE_PAIRS = ((0, 1), (0, 2), (2, 2), (2, 1))
NODE_GEOMETRY_NAMES = (
    "query_position_mm", "predicted_position_mm", "centre_minus_query_mm",
    "predicted_x_relative_to_path_origin_mm",
    "predicted_y_relative_to_path_origin_mm",
    "predicted_z_relative_to_path_origin_mm", "support",
)
PAIR_FEATURE_NAMES = (
    "query_delta_mm", "centre_delta_mm", "xyz_delta_x_mm",
    "xyz_delta_y_mm", "xyz_delta_z_mm", "xyz_distance_mm",
    "abs_query_delta_mm", "interval_FG_mean", "interval_FG_std",
    "interval_FG_trapezoid_integral_mm", "source_start", "source_end",
    "source_cut", "target_start", "target_end", "target_cut",
)
NORMALIZATION = {
    "feature425": "(x - frozen175_mean) / frozen175_std; no refitting or clipping",
    "raw7": "asinh(raw / 4); 4 logits then 3 frozen cell outputs",
    "geometry7": "q/L,c/L,(c-q)/16,relativeXYZ3/L,log1p(span/16); L=max(span,16mm)",
    "pair16": "asinh(first7/16),FGmean,FGstd,asinh(FGintegral/16),role6",
    "geometry_calculation": "float64 before conversion to network dtype",
    "span": "max(query_count-1,0)*step_mm; scale only, never a pruning rule",
    "zero_width_FG": "one inclusive sample: mean=FG, std=0, integral=0",
}


@dataclass(frozen=True)
class NodeInputs:
    node_features: torch.Tensor
    raw7: torch.Tensor
    roles: torch.Tensor
    node_geometry: torch.Tensor
    path_span_mm: torch.Tensor


class CompletePathRelations:
    """Enumerate every relation, without a node/edge cap or GT-built mask.

    path_ids only encode path membership and iteration order. Each within-path
    node list is in global ordinal order, matching graph 195's complete IDs.
    The feature callback receives global ordinal int64[K,2] arrays, never GT.
    """

    def __init__(self, path_ids, roles, pair_features: Callable, graph_fingerprint=None):
        paths = (path_ids.detach().cpu().numpy().copy() if isinstance(path_ids, torch.Tensor)
                 else np.array(path_ids, copy=True))
        role_values = (roles.detach().cpu().numpy().copy() if isinstance(roles, torch.Tensor)
                       else np.array(roles, copy=True))
        if paths.ndim != 1 or paths.dtype.kind not in "iu":
            raise ValueError("one integer path membership per node required")
        if role_values.shape != paths.shape or role_values.dtype.kind not in "iu":
            raise ValueError("one integer semantic role per node required")
        if len(paths) and ((paths < 0).any() or (role_values < 0).any() or (role_values > 2).any()):
            raise ValueError("nonnegative path membership and S/E/C roles required")
        if not callable(pair_features):
            raise ValueError("runtime pair feature provider required")
        self.path_ids = paths.astype(np.int64, copy=True)
        self.roles = role_values.astype(np.int64, copy=True)
        self.path_ids.setflags(write=False)
        self.roles.setflags(write=False)
        self.node_count = len(paths)
        self.paths = {int(p): np.flatnonzero(paths == p).astype(np.int64)
                      for p in np.unique(paths)}
        self.role_groups = {
            p: tuple(ids[self.roles[ids] == r] for r in range(3))
            for p, ids in self.paths.items()
        }
        self.pair_features = pair_features
        self.graph_fingerprint = graph_fingerprint

    def pair_count(self, kind):
        if kind == "message":
            return sum(len(v) * (len(v) - 1) for v in self.paths.values())
        if kind == "duplicate":
            return sum(len(v) * (len(v) - 1) // 2 for v in self.paths.values())
        if kind == "child":
            return sum(sum(len(groups[a]) * len(groups[b]) -
                           (len(groups[a]) if a == b else 0)
                           for a, b in CHILD_ROLE_PAIRS)
                       for groups in self.role_groups.values())
        raise ValueError("relation kind child/message/duplicate required")

    def iter_pairs(self, kind, chunk_size=65536):
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("positive integer chunk_size required")
        expected = self.pair_count(kind)
        count = 0
        for p, ids in self.paths.items():
            groups = ([(self.role_groups[p][a], self.role_groups[p][b], a == b)
                       for a, b in CHILD_ROLE_PAIRS] if kind == "child"
                      else [(ids, ids, True)])
            for sources, targets, exclude_self in groups:
                if not len(sources) or not len(targets):
                    continue
                for begin in range(0, len(sources) * len(targets), chunk_size):
                    flat = np.arange(begin, min(begin + chunk_size,
                                               len(sources) * len(targets)), dtype=np.int64)
                    pairs = np.c_[sources[flat // len(targets)], targets[flat % len(targets)]]
                    if kind == "duplicate":
                        pairs = pairs[pairs[:, 0] < pairs[:, 1]]
                    elif exclude_self:
                        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
                    if len(pairs):
                        count += len(pairs)
                        yield pairs
        if count != expected:
            raise RuntimeError("NOT_IMPLEMENTED: incomplete runtime relation enumeration")


@dataclass(frozen=True)
class BoundaryScores:
    u: torch.Tensor
    a: torch.Tensor
    d: torch.Tensor
    child_pairs: torch.Tensor
    duplicate_pairs: torch.Tensor
    node_states: torch.Tensor = None
    coverage: Mapping = None
    graph_fingerprint: str = None

    def detached_solver_scores(self):
        """Full ordinal-aligned coefficients; no discretization or threshold."""
        _validate_scores(self)
        return {name: getattr(self, name).detach().to(torch.float64).cpu().numpy().copy()
                for name in ("u", "a", "d")}


def query_graph_195_fingerprint(graph):
    """Bind node payload, ALL relation ordinals, and the full frozen FG field."""
    paths = []
    for path_id, context in sorted(graph.path_contexts.items()):
        paths.append({"path_id": int(path_id), "query_count": int(context.query_count),
                      "step_mm": float(context.step_mm), "origin_mm": list(context.origin_mm),
                      "direction": list(context.direction), "raw_sha256": context.raw_sha256})
    receipt = {"schema": "graph195_complete_runtime_v1", "paths": paths,
               "complete_hashes": graph.complete_hashes()}
    return sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def query_graph_195_inputs(graph):
    """Static graph adapter; no model forward, data read or truth access."""
    from weld_pipeline.boundary_query_graph_195 import (
        NODE_GEOMETRY_NAMES as graph_geometry_names,
        PAIR_FEATURE_NAMES as graph_pair_names,
    )
    if tuple(graph_geometry_names) != NODE_GEOMETRY_NAMES or tuple(graph_pair_names) != PAIR_FEATURE_NAMES:
        raise ValueError("graph195 feature schema changed; explicit version review required")
    values = graph.node_tensors()
    spans = []
    paths = []
    for node in graph.nodes:
        context = graph.path_contexts[node.path_id]
        spans.append(max(context.query_count - 1, 0) * context.step_mm)
        paths.append(node.path_id)
    inputs = NodeInputs(
        torch.tensor(np.array(values["node_features"], copy=True)),
        torch.tensor(np.array(values["raw7"], copy=True)),
        torch.tensor(np.array(values["roles"], copy=True)),
        torch.tensor(np.array(values["node_geometry"], copy=True), dtype=torch.float64),
        torch.tensor(spans, dtype=torch.float64),
    )
    relations = CompletePathRelations(np.asarray(paths, np.int64), values["roles"], graph.pair_features,
                                      query_graph_195_fingerprint(graph))
    for kind in ("child", "message", "duplicate"):
        if relations.pair_count(kind) != graph.pair_count(kind):
            raise ValueError("graph195 full relation count differs from network topology")
    return inputs, relations


def _finite(tensor, name):
    if not torch.is_floating_point(tensor) or not bool(torch.isfinite(tensor).all()):
        raise ValueError(name + " must contain finite floating-point values")


class BoundaryIncidenceNet(nn.Module):
    """442 -> 128, two full-path message passes, shared u/a/d scores."""

    def __init__(self, feature_mean, feature_std):
        super().__init__()
        mean = torch.as_tensor(feature_mean, dtype=torch.float32).detach().clone()
        std = torch.as_tensor(feature_std, dtype=torch.float32).detach().clone()
        if mean.shape != (FEATURE_COUNT,) or std.shape != mean.shape:
            raise ValueError("explicit frozen175 mean/std buffers [425] required")
        _finite(mean, "feature mean")
        _finite(std, "feature std")
        if bool((std <= 0).any()):
            raise ValueError("strictly positive frozen175 standard deviations required")
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std)
        self.node_encoder = nn.Sequential(nn.Linear(INPUT_DIM, NODE_WIDTH), nn.GELU(),
                                          nn.Linear(NODE_WIDTH, NODE_WIDTH), nn.GELU())
        self.messages = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * NODE_WIDTH + PAIR_DIM, NODE_WIDTH), nn.GELU(),
                          nn.Linear(NODE_WIDTH, NODE_WIDTH)) for _ in range(2)])
        self.updates = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * NODE_WIDTH, NODE_WIDTH), nn.GELU(),
                          nn.Linear(NODE_WIDTH, NODE_WIDTH)) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(NODE_WIDTH) for _ in range(2)])
        self.unary = nn.Linear(NODE_WIDTH, 1)
        self.child = nn.Sequential(nn.Linear(2 * NODE_WIDTH + PAIR_DIM, NODE_WIDTH),
                                   nn.GELU(), nn.Linear(NODE_WIDTH, 1))
        self.duplicate = nn.Sequential(nn.Linear(2 * NODE_WIDTH + PAIR_DIM, NODE_WIDTH),
                                       nn.GELU(), nn.Linear(NODE_WIDTH, 1))

    def normalize_nodes(self, inputs):
        if not isinstance(inputs, NodeInputs):
            raise TypeError("NodeInputs required")
        n = inputs.node_features.shape[0]
        if inputs.node_features.shape != (n, FEATURE_COUNT) or inputs.raw7.shape != (n, 7):
            raise ValueError("complete frozen node features [N,425] and raw7[N,7] required")
        if inputs.roles.shape != (n,) or inputs.roles.dtype not in (torch.int32, torch.int64):
            raise ValueError("semantic integer roles [N] required")
        if n and (bool((inputs.roles < 0).any()) or bool((inputs.roles > 2).any())):
            raise ValueError("role must be start/end/cut")
        if inputs.node_geometry.shape != (n, 7) or inputs.path_span_mm.shape != (n,):
            raise ValueError("geometry [N,7] and path span [N] required")
        for name in ("node_features", "raw7", "node_geometry", "path_span_mm"):
            _finite(getattr(inputs, name), name)
        device = self.feature_mean.device
        dtype = self.feature_mean.dtype
        feature = inputs.node_features.detach().to(device=device, dtype=dtype)
        raw = inputs.raw7.detach().to(device=device, dtype=dtype)
        geometry = inputs.node_geometry.detach().to(device=device, dtype=torch.float64)
        span = inputs.path_span_mm.detach().to(device=device, dtype=torch.float64)
        if bool((span < 0).any()):
            raise ValueError("path span cannot be negative")
        # geometry[2] is preserved for auditing; recompute c-q in float64.
        delta = geometry[:, 1] - geometry[:, 0]
        if not torch.allclose(delta, geometry[:, 2], rtol=1e-10, atol=1e-9):
            raise ValueError("geometry centre-minus-query disagrees with frozen centre")
        length = span.clamp_min(16.)
        scaled = torch.cat((geometry[:, 0:2] / length[:, None], delta[:, None] / 16.,
                            geometry[:, 3:6] / length[:, None],
                            torch.log1p(span / 16.)[:, None]), dim=1).to(dtype)
        role = F.one_hot(inputs.roles.detach().to(device=device, dtype=torch.int64), 3).to(dtype)
        return torch.cat(((feature - self.feature_mean) / self.feature_std,
                          torch.asinh(raw / 4.), role, scaled), dim=1)

    def normalize_pairs(self, raw):
        values = torch.as_tensor(raw).detach().to(device=self.feature_mean.device, dtype=torch.float64)
        if values.ndim != 2 or values.shape[1] != PAIR_DIM:
            raise ValueError("complete pair features [K,16] required")
        _finite(values, "pair features")
        result = torch.cat((torch.asinh(values[:, :7] / 16.), values[:, 7:9],
                            torch.asinh(values[:, 9:10] / 16.), values[:, 10:]), dim=1)
        return result.to(self.feature_mean.dtype)

    def _pair_input(self, hidden, pairs, relations):
        indices = torch.tensor(pairs, device=hidden.device, dtype=torch.int64)
        features = self.normalize_pairs(relations.pair_features(pairs))
        return indices, torch.cat((hidden[indices[:, 0]], hidden[indices[:, 1]], features), dim=1)

    def forward(self, inputs, relations, chunk_size=65536):
        if not isinstance(relations, CompletePathRelations):
            raise TypeError("CompletePathRelations required; GT edge masks are not accepted")
        if len(inputs.roles) != relations.node_count or not np.array_equal(
                inputs.roles.detach().cpu().numpy(), relations.roles):
            raise ValueError("network nodes and complete runtime relation roles differ")
        hidden = self.node_encoder(self.normalize_nodes(inputs))
        coverage = {"nodes": len(hidden), "message": [], "child": 0, "duplicate": 0,
                    "complete_relations": True, "truncated": False}
        for message, update, norm in zip(self.messages, self.updates, self.norms):
            total = torch.zeros_like(hidden)
            degree = hidden.new_zeros((len(hidden),))
            count = 0
            for pairs in relations.iter_pairs("message", chunk_size):
                indices, pair_input = self._pair_input(hidden, pairs, relations)
                total = total.index_add(0, indices[:, 1], message(pair_input))
                degree = degree.index_add(0, indices[:, 1], torch.ones(len(pairs), device=hidden.device,
                                                                     dtype=hidden.dtype))
                count += len(pairs)
            if count != relations.pair_count("message"):
                raise RuntimeError("incomplete message relations")
            mean = total / degree.clamp_min(1.)[:, None]
            hidden = norm(hidden + update(torch.cat((hidden, mean), dim=1)))
            coverage["message"].append(count)
        child_scores, child_pairs = [], []
        for pairs in relations.iter_pairs("child", chunk_size):
            indices, pair_input = self._pair_input(hidden, pairs, relations)
            child_scores.append(self.child(pair_input).squeeze(1))
            child_pairs.append(indices)
            coverage["child"] += len(pairs)
        duplicate_scores, duplicate_pairs = [], []
        for pairs in relations.iter_pairs("duplicate", chunk_size):
            indices, forward_input = self._pair_input(hidden, pairs, relations)
            _, reverse_input = self._pair_input(hidden, pairs[:, ::-1].copy(), relations)
            duplicate_scores.append(F.softplus((self.duplicate(forward_input).squeeze(1) +
                                               self.duplicate(reverse_input).squeeze(1)) / 2.))
            duplicate_pairs.append(indices)
            coverage["duplicate"] += len(pairs)
        for kind in ("child", "duplicate"):
            if coverage[kind] != relations.pair_count(kind):
                raise RuntimeError("incomplete " + kind + " relations")
        empty_pair = torch.empty((0, 2), dtype=torch.int64, device=hidden.device)
        # Empty sums retain a differentiable zero without fabricating candidates.
        empty_score = hidden.sum(dim=1)[:0]
        return BoundaryScores(
            self.unary(hidden).squeeze(1),
            torch.cat(child_scores) if child_scores else empty_score,
            torch.cat(duplicate_scores) if duplicate_scores else empty_score,
            torch.cat(child_pairs) if child_pairs else empty_pair,
            torch.cat(duplicate_pairs) if duplicate_pairs else empty_pair,
            hidden, coverage, relations.graph_fingerprint,
        )


def _validate_scores(scores):
    if not isinstance(scores, BoundaryScores):
        raise TypeError("BoundaryScores required")
    for name in ("u", "a", "d"):
        value = getattr(scores, name)
        if value.ndim != 1:
            raise ValueError(name + " must be an ordinal-aligned vector")
        _finite(value, name)
    for name, count in (("child_pairs", len(scores.a)), ("duplicate_pairs", len(scores.d))):
        pairs = getattr(scores, name)
        if pairs.shape != (count, 2) or pairs.dtype != torch.int64:
            raise ValueError(name + " must be full int64[K,2] ordinals")
        if count and (bool((pairs < 0).any()) or bool((pairs >= len(scores.u)).any()) or
                      bool((pairs[:, 0] == pairs[:, 1]).any())):
            raise ValueError(name + " contains invalid node ordinals")
    if bool((scores.d < 0).any()):
        raise ValueError("duplicate competition must be nonnegative")


def _binary_selection(value, size, device, name):
    selection = torch.as_tensor(value, device=device).detach()
    if selection.shape != (size,) or not bool(((selection == 0) | (selection == 1)).all()):
        raise ValueError(name + " must be complete binary selection")
    return selection.to(dtype=torch.float64)


def differentiable_structure_score(scores, z, y):
    """Unified score, accumulated in float64 with gradients to u/a/d only."""
    _validate_scores(scores)
    device = scores.u.device
    if any(getattr(scores, name).device != device for name in ("a", "d", "child_pairs", "duplicate_pairs")):
        raise ValueError("all scores and ordinal arrays must share device")
    nodes = _binary_selection(z, len(scores.u), device, "z")
    edges = _binary_selection(y, len(scores.a), device, "y")
    pairs = scores.duplicate_pairs
    competition = nodes[pairs[:, 0]] * nodes[pairs[:, 1]]
    return ((scores.u.to(torch.float64) * nodes).sum() +
            (scores.a.to(torch.float64) * edges).sum() -
            (scores.d.to(torch.float64) * competition).sum())


@dataclass(frozen=True)
class CertifiedSelection:
    """Detached result from exact solver 199; never a neural feature."""
    z: object
    y: object
    optimal_proved: bool
    complete_graph_solution: bool
    objective_max: float
    mip_gap: float = 0.
    graph_fingerprint: str = None


def solver_199_scores(scores):
    """Detach complete coefficients while preserving the graph binding."""
    from weld_pipeline.structured_graph_solver_v1 import GraphScores
    values = scores.detached_solver_scores()
    return GraphScores(values["u"], values["a"], values["d"], scores.graph_fingerprint)


def certified_selection_199(result, scores):
    """Check a199 exact-solve receipt before using its detached selection."""
    from weld_pipeline.structured_graph_solver_v1 import SolveResult
    if not isinstance(result, SolveResult):
        raise TypeError("solver199 SolveResult required")
    if (result.status != "OPTIMAL" or result.optimal_proved is not True or
            result.complete_graph_solution is not True or result.selection is None):
        raise ValueError("NOT_IMPLEMENTED: solver199 did not prove a complete optimum")
    if result.graph_fingerprint != scores.graph_fingerprint or not scores.graph_fingerprint:
        raise ValueError("solver199 result belongs to a different complete graph")
    if (result.mip_gap is None or not math.isfinite(result.mip_gap) or result.mip_gap < 0 or
            result.mip_gap > 1e-9 or result.upper_bound_max is None or
            not math.isfinite(result.upper_bound_max) or not math.isfinite(result.objective_max) or
            not math.isclose(result.upper_bound_max, result.objective_max, rel_tol=1e-9, abs_tol=1e-9)):
        raise ValueError("solver199 maximization bound does not prove the optimum")
    for receipt in result.component_receipts:
        if receipt.get("raw_status") != 0 or receipt.get("exact_zero_gap_certificate") is not True:
            raise ValueError("solver199 component optimum is unproved")
    selected = result.selection
    base = differentiable_structure_score(scores, selected.z, selected.y)
    if not math.isclose(float(base.detach()), result.base_score, rel_tol=1e-10, abs_tol=1e-6):
        raise ValueError("solver199 base score differs from neural coefficients")
    nodes = _binary_selection(selected.z, len(scores.u), scores.u.device, "z")
    products = nodes[scores.duplicate_pairs[:, 0]] * nodes[scores.duplicate_pairs[:, 1]]
    actual = _binary_selection(selected.t, len(scores.d), scores.u.device, "duplicate products")
    if not torch.equal(products, actual):
        raise ValueError("solver199 duplicate products differ from z_left*z_right")
    chosen = _binary_selection(selected.y, len(scores.a), scores.u.device, "y").to(torch.bool)
    expected_pairs = scores.child_pairs[chosen].detach().cpu().numpy()
    if not np.array_equal(expected_pairs, np.asarray(selected.selected_child_pairs)):
        raise ValueError("solver199 selected child pair ordinals differ from full neural graph")
    return CertifiedSelection(selected.z, selected.y, True, True, float(result.objective_max),
                              float(result.mip_gap), result.graph_fingerprint)


def latent_structured_hinge_from_optima(scores, good, bad, delta):
    """S(bad)+Delta-S(good), only after BOTH complete optima are proved.

    Good maximizes the truth-consistent latent set; bad maximizes base score
    plus identity margin over the complete runtime feasible set. Solver 199
    owns those sets and Delta. Arbitrary or time-limited selections are refused.
    """
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta < 0:
        raise ValueError("finite nonnegative detached identity margin required")
    fingerprint = scores.graph_fingerprint
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64 or
            any(c not in "0123456789abcdef" for c in fingerprint)):
        raise ValueError("complete graph SHA256 fingerprint required for latent hinge")
    for name, selection in (("good", good), ("bad", bad)):
        if not isinstance(selection, CertifiedSelection):
            raise TypeError(name + " CertifiedSelection required")
        if selection.optimal_proved is not True or selection.complete_graph_solution is not True:
            raise ValueError("NOT_IMPLEMENTED: " + name + " complete optimum is unproved")
        if selection.graph_fingerprint != fingerprint:
            raise ValueError(name + " optimum belongs to a different complete candidate graph")
        if not math.isfinite(selection.mip_gap) or selection.mip_gap < 0 or selection.mip_gap > 1e-9:
            raise ValueError(name + " optimum gap is not zero")
        if not math.isfinite(selection.objective_max):
            raise ValueError(name + " certified objective must be finite")
    good_score = differentiable_structure_score(scores, good.z, good.y)
    bad_score = differentiable_structure_score(scores, bad.z, bad.y)
    for name, actual, expected in (("good", float(good_score.detach()), good.objective_max),
                                   ("bad", float(bad_score.detach()) + delta, bad.objective_max)):
        if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-6):
            raise ValueError(name + " objective differs from ordinal-aligned neural score")
    hinge = bad_score + delta - good_score
    if float(hinge.detach()) < -1e-6:
        raise ValueError("bad optimum is below truth-consistent optimum; solver contract failed")
    # Only roundoff below zero is clamped; an unproved optimum was rejected above.
    return hinge.clamp_min(0.)


@dataclass(frozen=True)
class SourcePathLoss:
    loss: torch.Tensor
    source_key: object
    path_key: object
    truth_unit_count: int


def source_path_balanced_loss(records: Sequence[SourcePathLoss]):
    """Mean source -> mean complete path -> child+event truth-unit scale.

    Empty / structurally impossible paths remain present with zero loss. Their
    traversal is counted by the future runner; zero-gradient optimizer steps
    cannot be used to consume its update budget. Keys only group offline loss.
    """
    if not records:
        raise ValueError("complete source/path loss records required")
    grouped = {}
    seen = set()
    for record in records:
        if not isinstance(record, SourcePathLoss) or not isinstance(record.loss, torch.Tensor) or record.loss.ndim != 0:
            raise ValueError("scalar SourcePathLoss required")
        _finite(record.loss, "structured path loss")
        if float(record.loss.detach()) < 0:
            raise ValueError("structured path loss cannot be negative")
        if isinstance(record.truth_unit_count, bool) or not isinstance(record.truth_unit_count, int) or record.truth_unit_count < 0:
            raise ValueError("known true child+event integer count required")
        key = (record.source_key, record.path_key)
        if key in seen:
            raise ValueError("duplicate source/path loss record")
        seen.add(key)
        grouped.setdefault(record.source_key, []).append(record.loss / max(1, record.truth_unit_count))
    return torch.stack([torch.stack(losses).mean() for losses in grouped.values()]).mean()


def solve_runtime_graph(relations, scores):
    """Label-free runtime decoder interface; use separately verified solver199."""
    raise NotImplementedError("NOT_IMPLEMENTED: complete graph decoder belongs to solver199")
