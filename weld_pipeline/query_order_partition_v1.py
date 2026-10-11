"""Opt-in, deterministic query-order event instantiation for trial 189.

This module has no model/data imports. An explicit query identity is required.
Original admission gates and mean confidence are retained; only instance
partitioning/fusion changes. No geometry distance creates or merges identity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping

import numpy as np

VERSION = "query_order_l1_with_weighted_median_partition_v1_trial189"
ROLES = ("start", "end", "cut")
RADIUS_MM = 16.0
RESOLUTION_MM = 8.0 * RADIUS_MM * np.finfo(np.float32).eps
LAMBDA_CANDIDATES = (0.5, 1.0, 2.0, 4.0, 8.0)


@dataclass(frozen=True)
class Vote:
    sequence_runtime_id: str
    role: str
    query_index: int
    query_position_mm: float
    predicted_center_mm: float
    support_probability: float
    support_admitted: bool
    radius_admitted: bool
    span_lower_admitted: bool
    span_upper_admitted: bool

    @property
    def identity(self) -> tuple[str, str, int]:
        return self.sequence_runtime_id, self.role, self.query_index

    @property
    def admitted(self) -> bool:
        return (self.support_admitted and self.radius_admitted
                and self.span_lower_admitted and self.span_upper_admitted)


def votes_from_fields(support, offsets, step: float,
                      sequence_runtime_id: str, threshold: float = 0.5) -> list[Vote]:
    """Trace the exact original four gates; offsets are physical-radius offsets.

    Bounded 175 raw must first use its existing cell-to-physical-offset conversion.
    This adapter neither loads raw/model weights nor changes that conversion.
    """
    support = np.asarray(support, float)
    offsets = np.asarray(offsets, float)
    if (support.shape != offsets.shape or support.ndim != 2
            or support.shape[1] != 3 or len(support) == 0
            or not np.isfinite(support).all() or not np.isfinite(offsets).all()
            or not math.isfinite(step) or step <= 0.0
            or not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0):
        raise ValueError("Three finite typed fields, positive step and probability threshold required")
    if not isinstance(sequence_runtime_id, str) or not sequence_runtime_id:
        raise ValueError("Explicit nonempty candidate runtime identity required")
    q = np.arange(len(support), dtype=float) * step
    span = float(q[-1])
    votes = []
    for ri, role in enumerate(ROLES):
        centers = q + offsets[:, ri] * RADIUS_MM
        for qi, center in enumerate(centers):
            votes.append(Vote(sequence_runtime_id, role, qi, float(q[qi]),
                              float(center), float(np.clip(support[qi, ri], 0.0, 1.0)),
                              bool(support[qi, ri] >= threshold),
                              bool(abs(offsets[qi, ri]) <= 1.0 + 1e-6),
                              bool(center >= -RESOLUTION_MM),
                              bool(center <= span + RESOLUTION_MM)))
    return votes


def _unique_votes(votes: Iterable[Vote]) -> list[Vote]:
    records: dict[tuple[str, str, int], Vote] = {}
    for vote in votes:
        if not isinstance(vote, Vote):
            raise TypeError("Vote records required")
        if (not vote.sequence_runtime_id or vote.role not in ROLES
                or isinstance(vote.query_index, bool)
                or not isinstance(vote.query_index, (int, np.integer)) or vote.query_index < 0
                or not all(math.isfinite(v) for v in
                           (vote.query_position_mm, vote.predicted_center_mm, vote.support_probability))
                or not 0.0 <= vote.support_probability <= 1.0
                or not all(type(v) is bool for v in
                           (vote.support_admitted, vote.radius_admitted,
                            vote.span_lower_admitted, vote.span_upper_admitted))):
            raise ValueError("Finite explicit query record and four independent boolean gates required")
        previous = records.get(vote.identity)
        if previous is not None and previous != vote:
            raise ValueError("Conflicting records for one explicit query identity")
        records[vote.identity] = vote
    result = sorted(records.values(), key=lambda v: (v.sequence_runtime_id, ROLES.index(v.role), v.query_index))
    for a, b in zip(result[:-1], result[1:]):
        if (a.sequence_runtime_id, a.role) == (b.sequence_runtime_id, b.role):
            if b.query_position_mm <= a.query_position_mm:
                raise ValueError("Query positions must increase with query index inside a candidate/role")
    return result


def weighted_median(centers, weights) -> float:
    centers = np.asarray(centers, float)
    weights = np.asarray(weights, float)
    if (centers.ndim != 1 or centers.shape != weights.shape or len(centers) == 0
            or not np.isfinite(centers).all() or not np.isfinite(weights).all()
            or np.any(weights < 0.0) or not float(weights.sum()) > 0.0):
        raise ValueError("Positive finite total vote weight required")
    order = np.argsort(centers, kind="stable")
    masses = _integer_masses(weights)
    total = sum(masses)
    cumulative = 0
    # Exact binary float mass avoids order-dependent half-weight ties in DP.
    for index in order:
        cumulative += masses[int(index)]
        if 2 * cumulative >= total:
            return float(centers[index])
    raise ArithmeticError("Positive median mass unexpectedly unavailable")


def _integer_masses(weights) -> list[int]:
    ratios = [float(value).as_integer_ratio() for value in weights]
    denominator = max(pair[1] for pair in ratios)
    return [numerator * (denominator // divisor) for numerator, divisor in ratios]


class _IntegerFenwick:
    def __init__(self, size: int):
        self.tree = [0] * (size + 1)

    def add(self, index: int, value: int) -> None:
        index += 1
        while index < len(self.tree):
            self.tree[index] += value
            index += index & -index

    def lower_bound(self, target: int) -> int:
        index = 0
        bit = 1 << (len(self.tree).bit_length() - 1)
        while bit:
            other = index + bit
            if other < len(self.tree) and self.tree[other] < target:
                target -= self.tree[other]
                index = other
            bit >>= 1
        return min(index, len(self.tree) - 2)


class _Fenwick:
    def __init__(self, size: int):
        self.tree = np.zeros(size + 1, float)

    def add(self, index: int, value: float) -> None:
        index += 1
        while index < len(self.tree):
            self.tree[index] += value
            index += index & -index

    def prefix(self, stop: int) -> float:
        value = 0.0
        while stop:
            value += float(self.tree[stop])
            stop -= stop & -stop
        return value

    def lower_bound(self, target: float) -> int:
        index = 0
        bit = 1 << (len(self.tree).bit_length() - 1)
        while bit:
            other = index + bit
            if other < len(self.tree) and self.tree[other] < target:
                target -= float(self.tree[other])
                index = other
            bit >>= 1
        return min(index, len(self.tree) - 2)


def partition_run(votes: Iterable[Vote], sigma_mm: float,
                  lambda_global: float) -> dict:
    """Exact DP of one ordered admitted run, L1 plus lambda/block.

    cost(B) = sum_i abs(c_i - weighted_median(B, p)) / sigma_role + lambda.
    p weights only the median and original mean confidence, not residual loss.
    Every nonempty block, including a single query, is allowed. O(n² log n)
    time and O(n) working memory; no maximum run silently drops data.
    """
    values = list(votes)
    if not math.isfinite(sigma_mm) or sigma_mm <= 0.0:
        raise ValueError("Frozen positive training-derived sigma required")
    if lambda_global not in LAMBDA_CANDIDATES:
        raise ValueError("Lambda must be one of five preregistered global candidates")
    if not values:
        return {"blocks": [], "objective": 0.0}
    if any(not v.admitted for v in values):
        raise ValueError("DP run accepts admitted votes only")
    if any(v.identity[:2] != values[0].identity[:2] for v in values):
        raise ValueError("DP run must stay within one candidate and role")
    if any(a.query_index >= b.query_index for a, b in zip(values[:-1], values[1:])):
        raise ValueError("DP run must follow strictly increasing unique query indices")
    centers = np.array([v.predicted_center_mm for v in values])
    weights = np.array([v.support_probability for v in values])
    if np.any(weights <= 0.0):
        raise ValueError("Admitted vote weights must be positive")
    # Shift moments near the run to reduce cancellation for large world offsets.
    anchor = float(np.min(centers))
    shifted = centers - anchor
    coordinates = np.unique(shifted)
    ranks = np.searchsorted(coordinates, shifted)
    masses = _integer_masses(weights)
    n = len(values)
    best = np.full(n + 1, np.inf)
    best[0] = 0.0
    block_count = np.full(n + 1, n + 1, dtype=int)
    block_count[0] = 0
    previous = np.zeros(n + 1, dtype=int)
    for stop in range(1, n + 1):
        mass_tree = _IntegerFenwick(len(coordinates))
        count_tree = _Fenwick(len(coordinates))
        moment_tree = _Fenwick(len(coordinates))
        total_mass = 0
        total_moment = 0.0
        for start in range(stop - 1, -1, -1):
            c = float(shifted[start])
            mass_tree.add(int(ranks[start]), masses[start])
            count_tree.add(int(ranks[start]), 1.0)
            moment_tree.add(int(ranks[start]), c)
            total_mass += masses[start]
            total_moment += c
            median_rank = mass_tree.lower_bound((total_mass + 1) // 2)
            median = float(coordinates[median_rank])
            left_count = count_tree.prefix(median_rank + 1)
            left_moment = moment_tree.prefix(median_rank + 1)
            absolute = (median * left_count - left_moment
                        + total_moment - left_moment - median * (stop - start - left_count))
            # Cancellation may produce tiny negative roundoff for identical centers.
            loss = max(0.0, float(absolute)) / sigma_mm
            candidate = float(best[start] + loss + lambda_global)
            count = int(block_count[start] + 1)
            if (candidate < best[stop]
                    or (candidate == best[stop] and (count < block_count[stop]
                        or (count == block_count[stop] and start < previous[stop])))):
                best[stop], block_count[stop], previous[stop] = candidate, count, start
    bounds = []
    stop = n
    while stop:
        start = int(previous[stop])
        block = values[start:stop]
        center = weighted_median([v.predicted_center_mm for v in block],
                                 [v.support_probability for v in block])
        loss = sum(abs(v.predicted_center_mm - center) for v in block) / sigma_mm
        bounds.append({"start": start, "stop": stop, "center_mm": center,
                       "residual_cost": float(loss), "cost": float(loss + lambda_global),
                       "query_indices": [int(v.query_index) for v in block]})
        stop = start
    bounds.reverse()
    return {"blocks": bounds, "objective": float(sum(b["cost"] for b in bounds))}


def partition_votes(votes: Iterable[Vote], span_mm: float,
                    sigma_by_role: Mapping[str, float], lambda_global: float = 0.5,
                    threshold: float = 0.5, quantize_output: bool = True) -> dict:
    """Return instances and legacy event pairs for exactly one runtime candidate.

    Support-positive consecutive query indices define runs. Radius/span failures
    stay in trace and cannot vote; they do not alter the support run definition.
    Distinct instance IDs survive an equal quantized output coordinate. The old
    coordinate-only decoder can still lose that distinction; this API reports it.
    """
    if (not math.isfinite(span_mm) or span_mm < 0.0 or not math.isfinite(threshold)
            or not 0.0 <= threshold <= 1.0 or type(quantize_output) is not bool):
        raise ValueError("Finite nonnegative span and probability threshold required")
    if set(sigma_by_role) != set(ROLES) or any(not math.isfinite(float(v)) or v <= 0.0 for v in sigma_by_role.values()):
        raise ValueError("Three fixed positive role sigma values required")
    if lambda_global not in LAMBDA_CANDIDATES:
        raise ValueError("Unregistered lambda")
    records = _unique_votes(votes)
    sequence_ids = {v.sequence_runtime_id for v in records}
    if len(sequence_ids) > 1:
        raise ValueError("Partition candidates independently; cross-candidate grouping forbidden")
    event_sets, role_instances, run_traces = [], [], []
    for role in ROLES:
        role_votes = [v for v in records if v.role == role]
        support_runs = []
        run = []
        previous_index = None
        for vote in role_votes:
            if not vote.support_admitted:
                if run:
                    support_runs.append(run)
                    run = []
                previous_index = vote.query_index
                continue
            if run and vote.query_index != previous_index + 1:
                support_runs.append(run)
                run = []
            run.append(vote)
            previous_index = vote.query_index
        if run:
            support_runs.append(run)
        instances, events = [], []
        for ri, support_run in enumerate(support_runs):
            admitted = [v for v in support_run if v.admitted]
            partition = partition_run(admitted, float(sigma_by_role[role]), lambda_global)
            run_traces.append({"role": role, "support_run_index": ri,
                               "support_query_indices": [v.query_index for v in support_run],
                               "admitted_query_indices": [v.query_index for v in admitted],
                               "objective": partition["objective"]})
            for block in partition["blocks"]:
                members = admitted[block["start"]:block["stop"]]
                center = block["center_mm"]
                output = (float(np.rint(center / RESOLUTION_MM) * RESOLUTION_MM)
                          if quantize_output else center)
                output = float(np.clip(output, 0.0, span_mm))
                confidence = float(np.mean([v.support_probability for v in members]))
                emitted = confidence >= threshold
                instance = {"instance_id": f"{members[0].sequence_runtime_id}:{role}:run{ri}:q{members[0].query_index}-{members[-1].query_index}",
                            "sequence_runtime_id": members[0].sequence_runtime_id,
                            "role": role, "support_run_index": ri,
                            "query_indices": block["query_indices"],
                            "center_unquantized_mm": center, "center_output_mm": output,
                            "confidence": min(confidence, 1.0 - 1e-4),
                            "raw_mean_confidence": confidence, "emitted": emitted,
                            "residual_cost": block["residual_cost"], "block_cost": block["cost"]}
                instances.append(instance)
                if emitted:
                    events.append([output, instance["confidence"]])
        event_sets.append(sorted(events, key=lambda e: e[0]))
        role_instances.append(instances)
    collisions = []
    for role, instances in zip(ROLES, role_instances):
        coordinates = {}
        for instance in instances:
            if instance["emitted"]:
                coordinates.setdefault(instance["center_output_mm"], []).append(instance["instance_id"])
        collisions.extend({"role": role, "center_output_mm": p, "instance_ids": ids,
                           "legacy_decoder_identity_interface": "COORDINATE_ONLY"}
                          for p, ids in coordinates.items() if len(ids) > 1)
    return {"version": VERSION, "events": event_sets, "instances": role_instances,
            "trace": [dict(asdict(v), finite=True, admitted=v.admitted) for v in records],
            "support_runs": run_traces, "output_coordinate_collisions": collisions,
            "lambda_global": lambda_global, "sigma_by_role_mm": dict(sigma_by_role),
            "quantize_output": quantize_output, "identity_ambiguity_not_inferred_from_distance": True}


def competing_hypotheses(case_id: str, scalar_records,
                         hypotheses: Iterable[Mapping]) -> dict:
    """Record declared artificial competing identities, never remove real G."""
    candidates = list(hypotheses)
    if len(candidates) < 2 or any("event_ids_by_query" not in h for h in candidates):
        raise ValueError("At least two explicit competing query-event identity hypotheses required")
    return {"case_id": case_id, "status": "EVENT_IDENTITY_AMBIGUOUS",
            "identical_scalar_records": scalar_records, "competing_hypotheses": candidates,
            "formal_G_exclusion_allowed": False,
            "scope": "Declared scalar-evidence ambiguity; no claim about full PCD features"}


def estimate_role_sigma_from_verified_events(event_records: Iterable[Mapping],
                                            ownership_verification: Mapping) -> dict:
    """Estimate the preregistered scale only after original identity verification.

    Each event record contains role, original_event_id, split='FIT', and votes.
    All original known supervised queries are used, independent of runtime gates.
    sigma_role = median_event(mean_query(abs(c - weighted_median(c,p)))).
    No GT-center error, pooled-query weighting, DEV/OBS tuning, or numeric floor.
    The caller must bind the original-generator verification receipt and samples.
    """
    if (ownership_verification.get("status") != "READY"
            or ownership_verification.get("identity_source") != "original_target_generator"
            or ownership_verification.get("query_identity_unique") is not True
            or not ownership_verification.get("receipt_sha256")
            or set(ownership_verification.get("expected_events_by_role", {})) != set(ROLES)
            or set(ownership_verification.get("expected_known_queries_by_role", {})) != set(ROLES)):
        raise ValueError("188 original-generator query ownership verification PENDING")
    dispersions = {r: [] for r in ROLES}
    known_counts = {r: 0 for r in ROLES}
    event_ids, query_ids = set(), set()
    for record in event_records:
        role, event_id = record.get("role"), record.get("original_event_id")
        if record.get("split") != "FIT" or role not in ROLES or not event_id:
            raise ValueError("Only identified original FIT events may estimate sigma")
        key = (role, str(event_id))
        if key in event_ids:
            raise ValueError("Duplicate original event identity in sigma estimation")
        event_ids.add(key)
        votes = _unique_votes(record.get("votes", []))
        if (not votes or any(v.role != role for v in votes)
                or len({v.sequence_runtime_id for v in votes}) != 1):
            raise ValueError("Each sigma event requires known supervised typed queries")
        if any(v.identity in query_ids for v in votes):
            raise ValueError("One query cannot be owned by two sigma events of its role")
        query_ids.update(v.identity for v in votes)
        known_counts[role] += len(votes)
        center = weighted_median([v.predicted_center_mm for v in votes],
                                 [v.support_probability for v in votes])
        dispersions[role].append(float(np.mean([abs(v.predicted_center_mm - center) for v in votes])))
    if any(not values for values in dispersions.values()):
        raise ValueError("All three roles require complete original FIT event coverage")
    if ({r: len(v) for r, v in dispersions.items()} != dict(ownership_verification["expected_events_by_role"])
            or known_counts != dict(ownership_verification["expected_known_queries_by_role"])):
        raise ValueError("Sigma original FIT event/query coverage differs from bound verification")
    sigma = {r: float(np.median(values)) for r, values in dispersions.items()}
    if any(value <= 0.0 for value in sigma.values()):
        raise ValueError("Zero role scale: stop trial; no hidden millimeter floor allowed")
    return {"sigma_by_role_mm": sigma,
            "events_by_role": {r: len(v) for r, v in dispersions.items()},
            "known_queries": len(query_ids), "known_queries_by_role": known_counts,
            "method": "median_event(mean_query(abs(c-weighted_median(c,p))))",
            "ownership_verification": dict(ownership_verification),
            "uses_GT_center_error": False, "DEV_OBS_used": False}


def center_modes_query_order(support, offsets, step: float,
                             sequence_runtime_id: str, sigma_by_role: Mapping[str, float],
                             lambda_global: float, threshold: float = 0.5,
                             quantize_output: bool = True) -> dict:
    """Dense runtime integration hook; caller keeps original restore/decoder."""
    records = votes_from_fields(support, offsets, step, sequence_runtime_id, threshold)
    return partition_votes(records, (len(support) - 1) * step, sigma_by_role,
                           lambda_global, threshold, quantize_output)
