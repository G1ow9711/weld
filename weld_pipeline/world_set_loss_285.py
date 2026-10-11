"""All original parents/children supervise B; never imported by runtime."""
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from .world_set_geometry_285 import parent_samples

MODE_NAMES = ("LINE_CHAIN", "CIRCLE_CHAIN", "MIXED_CHAIN")
CLASS_WEIGHTS = {("horizontal", "line"): 1605 / (3 * 1443),
                 ("vertical", "line"): 1605 / (3 * 138),
                 ("horizontal", "arc"): 1605 / (3 * 24)}


def semantics(target, queries, device):
    n = target["original_child_count"]
    kinds = [1 if child["original_kind"] == "line" else 2 for child in target["children"]]
    positive = [gap == "POSITIVE_GAP" for gap in target["gaps"]]
    direction = 1
    if target["mode"] == "CIRCLE_CHAIN":
        direction = 1 if target["children"][0]["world_representation"]["arc_parameters"]["sweep_rad"] > 0 else -1
    return dict(mode=MODE_NAMES.index(target["mode"]), child_count=n,
        positive_gaps=torch.tensor(positive, dtype=torch.bool, device=device).expand(queries, n-1),
        directions=torch.full((queries,), direction, dtype=torch.long, device=device),
        closed=torch.full((queries,), bool(target["is_closed"]), dtype=torch.bool, device=device),
        child_kinds=torch.tensor(kinds, dtype=torch.long, device=device).expand(queries, n))


def reference_samples(target, origin, device):
    values = np.asarray([child["loss_samples_mm"] for child in target["children"]], np.float64)
    return torch.tensor(values, dtype=torch.float64, device=device) - origin[None, None]


def _candidate_cost(output, target):
    q = len(output["presence_logits"])
    device = output["presence_logits"].device
    rows = torch.arange(q, dtype=torch.long, device=device)
    semantic = semantics(target, q, device)
    predicted, _ = parent_samples(output, rows, **semantic)
    reference = reference_samples(target, output["origin_mm"], device)
    error = torch.linalg.vector_norm(predicted - reference[None], dim=-1)
    endpoint = error[:, :, [0, -1]].mean(dim=(1, 2))
    curve = error.mean(dim=(1, 2))
    kinds = semantic["child_kinds"][0]
    kind_probability = output["sequence_kind_logits"][:, :len(kinds)].log_softmax(dim=-1)
    kind_cost = -kind_probability.gather(2, kinds[None, :, None].expand(q, -1, 1)).mean(dim=(1, 2))
    mode_cost = -output["mode_logits"].log_softmax(dim=-1)[:, semantic["mode"]]
    stop_cost = -output["sequence_kind_logits"][:, target["original_child_count"]].log_softmax(dim=-1)[:, 0]
    return .5 * endpoint + .5 * curve + 2 * (kind_cost + mode_cost + stop_cost) - F.logsigmoid(output["presence_logits"])


def full_world_set_loss(output, source_targets):
    """One-to-one complete source matching; all unmatched queries stay negative."""
    parents = source_targets["parents"]
    if len(parents) != source_targets["original_parents"]:
        raise ValueError("Original parent denominator differs")
    original_children = sum(parent["original_child_count"] for parent in parents)
    if original_children != source_targets["original_children"]:
        raise ValueError("Original child denominator differs")
    q = len(output["presence_logits"])
    device = output["presence_logits"].device
    if len(parents) > q:
        raise ValueError("Registered parent query capacity cannot cover original source")
    with torch.no_grad():
        costs = torch.stack([_candidate_cost(output, target) for target in parents], dim=1)
        if not torch.isfinite(costs).all():
            raise ValueError("Nonfinite world matching cost; no GT parent removed")
        matched_queries, matched_parents = linear_sum_assignment(costs.detach().numpy())
    if len(matched_parents) != len(parents):
        raise ValueError("Every original parent must receive a supervision match")
    exist = torch.zeros(q, dtype=torch.float32, device=device)
    exist[torch.tensor(matched_queries, dtype=torch.long, device=device)] = 1
    presence = output["presence_logits"]
    positive_mask = exist.bool()
    presence_loss = (F.binary_cross_entropy_with_logits(presence[positive_mask], exist[positive_mask])
                     if positive_mask.any() else presence.sum()*0)
    if (~positive_mask).any():
        presence_loss = presence_loss + F.binary_cross_entropy_with_logits(presence[~positive_mask], exist[~positive_mask])
    zero = presence.sum()*0
    sums = {name: zero for name in ("mode", "sequence", "stop", "tail", "weld_type", "gap", "closed", "direction", "turns", "endpoints", "curve")}
    weight_total = 0.
    represented = 0
    maximum_training_sample_error_mm = 0.
    for query, parent in zip(matched_queries.tolist(), matched_parents.tolist()):
        target = parents[parent]
        semantic = semantics(target, 1, device)
        row = torch.tensor([query], dtype=torch.long, device=device)
        points, _ = parent_samples(output, row, **semantic)
        points = points[0]
        reference = reference_samples(target, output["origin_mm"], device)
        n = target["original_child_count"]
        weights = torch.tensor([CLASS_WEIGHTS[(child["weld_type"], child["original_kind"])]
            for child in target["children"]], dtype=torch.float64, device=device)
        weight_total += float(weights.sum())
        represented += n
        sums["mode"] = sums["mode"] + F.cross_entropy(output["mode_logits"][query:query+1], torch.tensor([semantic["mode"]], device=device))
        sums["sequence"] = sums["sequence"] + (F.cross_entropy(output["sequence_kind_logits"][query, :n],
            semantic["child_kinds"][0], reduction="none") * weights).sum()
        sums["stop"] = sums["stop"] + F.cross_entropy(output["sequence_kind_logits"][query, n:n+1], torch.zeros(1, dtype=torch.long, device=device))
        if n + 1 < output["sequence_kind_logits"].shape[1]:
            tail = output["sequence_kind_logits"][query, n+1:]
            sums["tail"] = sums["tail"] + F.cross_entropy(tail, torch.zeros(len(tail), dtype=torch.long, device=device))
        weld = torch.tensor([0 if child["weld_type"] == "horizontal" else 1 for child in target["children"]], dtype=torch.long, device=device)
        sums["weld_type"] = sums["weld_type"] + (F.cross_entropy(output["weld_type_logits"][query, :n], weld, reduction="none") * weights).sum()
        if n > 1:
            sums["gap"] = sums["gap"] + F.cross_entropy(output["gap_logits"][query, :n-1], semantic["positive_gaps"][0].long())
        closure = torch.tensor([int(target["is_closed"])], dtype=torch.long, device=device)
        sums["closed"] = sums["closed"] + F.cross_entropy(output["closed_logits"][query:query+1], closure)
        sums["turns"] = sums["turns"] + F.cross_entropy(output["turns_logits"][query:query+1], closure)
        if semantic["mode"] == 1:
            direction = ((semantic["directions"] + 1) // 2).long()
            sums["direction"] = sums["direction"] + F.cross_entropy(output["direction_logits"][query:query+1], direction)
        endpoint_loss = F.smooth_l1_loss(points[:, [0, -1]], reference[:, [0, -1]], reduction="none", beta=1.).mean(dim=(1, 2))
        error = torch.linalg.vector_norm(points-reference, dim=-1)
        curve_loss = error.mean(dim=1) + error.max(dim=1).values
        sums["endpoints"] = sums["endpoints"] + (endpoint_loss * weights).sum()
        sums["curve"] = sums["curve"] + (curve_loss * weights).sum()
        maximum_training_sample_error_mm = max(maximum_training_sample_error_mm, float(error.detach().max()))
    if represented != original_children:
        raise ValueError("Loss omitted an original child")
    parent_denominator = max(1, len(parents))
    for name in sums:
        sums[name] = sums[name] / (weight_total if name in ("sequence", "weld_type", "endpoints", "curve") else parent_denominator)
    negative_tail = output["sequence_kind_logits"][~positive_mask, 0]
    if len(negative_tail):
        sums["tail"] = sums["tail"] + F.cross_entropy(negative_tail, torch.zeros(len(negative_tail), dtype=torch.long, device=device))
    loss = (10 * presence_loss + 2 * (sums["mode"] + sums["sequence"] + sums["stop"] + sums["weld_type"]
            + sums["gap"] + sums["closed"] + sums["direction"] + sums["turns"])
            + .05 * sums["tail"] + sums["endpoints"] + .5 * sums["curve"])
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite complete source loss")
    return loss, dict(original_parents=len(parents), original_children=original_children, supervised_children=represented,
        matched_queries=len(matched_queries), negative_query_count=q-len(parents), all_original_denominators_retained=True,
        presence_loss=float(presence_loss.detach()), losses={name:float(value.detach()) for name,value in sums.items()},
        total_loss=float(loss.detach()), maximum_training_sample_error_mm=maximum_training_sample_error_mm,
        training_sample_error_is_contract_precision=False, formal_E_reviewed_cells=0, formal_E_loss_not_trained=True)
