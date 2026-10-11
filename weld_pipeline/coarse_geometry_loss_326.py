"""GT-only geometric supervision with one whole-parent orientation choice."""
from copy import deepcopy
import math
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from .world_set_loss_285 import semantics, reference_samples
from .world_set_geometry_285 import parent_samples
from .contract_curve_geometry import CircularArc
from .current_process_runtime_324 import ENDPOINT_PAIRS

CLASS_COUNTS = {"horizontal/line": 170, "vertical/line": 126, "horizontal/arc": 24}
SCALE_MM, BETA = 100., .03


def reverse_supervision_parent(parent):
    """Reverse a label view, retaining identities and the original source file."""
    value = deepcopy(parent)
    value["children"].reverse()
    value["gaps"].reverse()
    if "endpoints_mm" in value:
        value["endpoints_mm"].reverse()
    for child in value["children"]:
        for key in ("loss_samples_mm", "original_points_mm"):
            if key in child:
                child[key].reverse()
        rep = child["world_representation"]
        if child["original_kind"] == "line":
            rep["endpoints_mm"].reverse()
        elif child["original_kind"] == "arc":
            arc = CircularArc.from_parameters(rep["arc_parameters"])
            end = arc.point_at(arc.length_mm)
            rep["arc_parameters"] = CircularArc(arc.center_mm, arc.radius_mm,
                arc.plane_normal, (end - arc.center_mm) / arc.radius_mm,
                -arc.sweep_rad, arc.is_closed).to_parameters()
        else:
            raise ValueError("UNSUPPORTED_REGISTERED_DIAGNOSTIC_CHILD_KIND")
        fields = child.get("original_attributes", {})
        for left, right in ENDPOINT_PAIRS:
            if left in fields or right in fields:
                fields[left], fields[right] = fields.get(right), fields.get(left)
    value["supervision_view"] = "WHOLE_PARENT_REVERSED_ORIGINAL_IDS_RETAINED"
    value["view_original_child_ids"] = [c["original_child_id"] for c in value["children"]]
    return value


def supervision_views(parent):
    if parent["mode"] not in ("LINE_CHAIN", "CIRCLE_CHAIN"):
        raise ValueError("DIAGNOSTIC_MIXED_GEOMETRY_HEAD_IS_FROZEN_NOT_A_TRAINED_PROVIDER")
    assert len(parent["children"]) == parent["original_child_count"]
    assert len(parent["gaps"]) == parent["original_child_count"] - 1
    for child in parent["children"]:
        if child["weld_type"] == "vertical":
            points = np.asarray(child["loss_samples_mm"], float)
            if child["original_kind"] != "line" or not points[0, 2] > points[-1, 2]:
                raise ValueError("REGISTERED_VERTICAL_TARGET_IS_NOT_CONFIRMED_DOWNWARD_LINE")
    if not parent["is_closed"] and all(c["weld_type"] == "horizontal" for c in parent["children"]):
        return (parent, reverse_supervision_parent(parent))
    return (parent,)


def variant_terms(output, view, rows):
    device = output["presence_logits"].device
    rows = torch.as_tensor(rows, dtype=torch.long, device=device)
    points, parameters = parent_samples(output, rows, **semantics(view, len(rows), device))
    reference = reference_samples(view, output["origin_mm"], device)
    errors = torch.linalg.vector_norm(points - reference[None], dim=-1)
    scaled = errors / SCALE_MM
    huber = F.smooth_l1_loss(scaled, torch.zeros_like(scaled), reduction="none", beta=BETA)
    terms = dict(endpoint=huber[:, :, [0, -1]].mean(dim=-1),
                 mean=huber.mean(dim=-1), tail=huber.max(dim=-1).values)
    weights = torch.tensor([1. / CLASS_COUNTS[c["weld_type"] + "/" + c["original_kind"]]
                            for c in view["children"]], dtype=torch.float64, device=device)
    losses = sum(terms.values())
    total = (losses * weights[None]).sum(dim=-1)
    if not torch.isfinite(total).all():
        raise ValueError("NONFINITE_REGISTERED_COARSE_GEOMETRY_LOSS")
    return total, terms, errors, points, parameters


@torch.no_grad()
def establish_fixed_correspondence(output, targets):
    parents = targets["parents"]
    assert len(parents) == targets["original_parents"]
    assert sum(p["original_child_count"] for p in parents) == targets["original_children"]
    query_rows = torch.arange(len(output["presence_logits"]))
    costs, variants = [], []
    for parent in parents:
        values = torch.stack([variant_terms(output, v, query_rows)[0] for v in supervision_views(parent)])
        best = values.argmin(dim=0)  # First/original wins ties.
        costs.append(values.gather(0, best[None])[0])
        variants.append(best)
    matrix = torch.stack(costs, dim=1)
    assert torch.isfinite(matrix).all()
    queries, rows = linear_sum_assignment(matrix.numpy())
    assert len(rows) == len(parents) and len(set(queries.tolist())) == len(parents)
    result = []
    for query, row in zip(queries.tolist(), rows.tolist()):
        parent = parents[row]
        result.append(dict(query=query, parent=row, original_parent_id=parent["original_parent_id"],
            original_child_ids=[c["original_child_id"] for c in parent["children"]],
            original_gap_identity=list(parent["gaps"]),
            original_parent_endpoints_mm=parent["endpoints_mm"],
            allowed_whole_parent_views=len(supervision_views(parent)),
            initial_selected_view=int(variants[row][query]),
            initial_same_geometric_loss_cost=float(matrix[query, row]),
            matching_has_no_presence_or_discrete_prediction_cost=True))
    return sorted(result, key=lambda x: x["parent"])


def fixed_geometry_loss(output, targets, matching):
    assert len(matching) == targets["original_parents"]
    assert {m["parent"] for m in matching} == set(range(len(targets["parents"])))
    total = None
    reported = {k: 0. for k in ("endpoint", "mean", "tail")}
    choices, represented = [], 0
    for match in matching:
        parent = targets["parents"][match["parent"]]
        assert parent["original_parent_id"] == match["original_parent_id"]
        assert [c["original_child_id"] for c in parent["children"]] == match["original_child_ids"]
        views = supervision_views(parent)
        calculations = [variant_terms(output, v, [match["query"]]) for v in views]
        selected = int(torch.stack([c[0][0].detach() for c in calculations]).argmin())
        value, components, error, _, _ = calculations[selected]
        total = value[0] if total is None else total + value[0]
        weights = torch.tensor([1. / CLASS_COUNTS[c["weld_type"] + "/" + c["original_kind"]]
            for c in views[selected]["children"]], dtype=torch.float64)
        for name in reported:
            reported[name] += float((components[name][0].detach() * weights).sum())
        represented += parent["original_child_count"]
        choices.append(dict(query=match["query"], original_parent_id=parent["original_parent_id"],
            selected_whole_parent_view=selected,
            selected_original_child_ids=[c["original_child_id"] for c in views[selected]["children"]],
            selected_gap_identity=list(views[selected]["gaps"]),
            sampled_maximum_errors_mm=error[0].detach().max(dim=-1).values.tolist()))
    assert represented == targets["original_children"] and total is not None and torch.isfinite(total)
    return total, dict(loss=float(total.detach()), components=reported,
        selected_whole_parent_orientations=choices, supervised_children=represented,
        fixed_query_parent_assignment=True, matching_recomputed_during_training=False,
        per_child_independent_orientation=False, physical_scale_mm=SCALE_MM, beta=BETA,
        fixed_class_counts=CLASS_COUNTS, original_labels_modified=False,
        sampled_training_loss_is_contract_precision=False)
