"""Fixed-query complete analytic geometry diagnostic; never detection accuracy."""
import math
import numpy as np
import torch

from .coarse_geometry_loss_326 import supervision_views, variant_terms
from .contract_curve_geometry import CircularArc, sample_arc
from .world_set_evaluation_285 import distance_to_primitive
from .process_geometry_constraints_326 import validate_component_geometry, ProcessGeometryValidationError
from .customer_process_rules_302 import is_long_enough


def finite_line_hausdorff(predicted, reference):
    def distances(points, a, b):
        delta = b - a
        norm = float(delta @ delta)
        if norm <= 0:
            return np.full(len(points), np.inf)
        position = np.clip((points - a) @ delta / norm, 0., 1.)
        return np.linalg.norm(points - a - position[:, None] * delta, axis=1)
    return max(float(distances(predicted[[0, -1]], reference[0], reference[-1]).max()),
               float(distances(reference[[0, -1]], predicted[0], predicted[-1]).max()))


def arc_certificate(predicted_arc, reference_arc):
    history = []
    for level in range(9):
        step = 1. / (2 ** level)
        a = sample_arc(predicted_arc, max_step_mm=step, max_chord_error_mm=.1)
        b = sample_arc(reference_arc, max_step_mm=step, max_chord_error_mm=.1)
        lower = max(float(distance_to_primitive(np.asarray(a["points"]), b).max()),
                    float(distance_to_primitive(np.asarray(b["points"]), a).max()))
        # Any point lies within half an included arc-length interval of a
        # sample. Distance to a fixed primitive is 1-Lipschitz in arc length.
        bound = max(float(distance_to_primitive(np.asarray(a["points"]), b).max())
                    + predicted_arc.length_mm / (2 * (len(a["points"]) - 1)),
                    float(distance_to_primitive(np.asarray(b["points"]), a).max())
                    + reference_arc.length_mm / (2 * (len(b["points"]) - 1)))
        history.append(dict(maximum_step_mm=step, sampled_lower_bound_mm=lower,
                            continuous_upper_bound_mm=bound))
        if bound <= 3.:
            return dict(status="CERTIFIED_PASS", pass_3mm=True, refinements=history)
        if lower > 3.:
            return dict(status="CERTIFIED_FAIL", pass_3mm=False, refinements=history)
    return dict(status="UNPROVEN_AFTER_REGISTERED_REFINEMENT_LIMIT", pass_3mm=False, refinements=history)


@torch.no_grad()
def evaluate_fixed_source(output, targets, matching):
    rows = []
    for match in matching:
        parent = targets["parents"][match["parent"]]
        views = supervision_views(parent)
        variants = [variant_terms(output, view, [match["query"]]) for view in views]
        selected = int(torch.stack([v[0][0] for v in variants]).argmin())
        view = views[selected]
        _, _, error, points, parameters = variants[selected]
        predicted = points[0].numpy() + output["origin_mm"].numpy()
        reference = np.asarray([c["loss_samples_mm"] for c in view["children"]])
        segments, child_results = [], []
        for index, child in enumerate(view["children"]):
            ends = float(error[0, index, [0, -1]].max())
            segment = dict(kind=child["original_kind"], points=predicted[index].tolist())
            length = float(np.linalg.norm(predicted[index, -1] - predicted[index, 0]))
            if child["original_kind"] == "line":
                whole = finite_line_hausdorff(predicted[index], reference[index])
                certificate = dict(status="EXACT_FINITE_LINE", continuous_maximum_mm=whole,
                                   pass_3mm=whole <= 3.)
            else:
                center = parameters["center"][0].numpy() + output["origin_mm"].numpy()
                radius = float(parameters["radius"][0])
                sweep = float(parameters["total_sweep"][0]) * float(
                    parameters["parameter_right"][0, index] - parameters["parameter_left"][0, index])
                arc = CircularArc(center, radius, parameters["normal"][0].numpy(),
                    (predicted[index, 0] - center) / radius, sweep,
                    bool(view["is_closed"] and len(view["children"]) == 1))
                segment["arc_parameters"] = arc.to_parameters()
                length = arc.length_mm
                if ends <= 3.:
                    try:
                        certificate = arc_certificate(arc, CircularArc.from_parameters(child["world_representation"]["arc_parameters"]))
                    except (ValueError, RuntimeError, OverflowError) as error_value:
                        certificate = dict(status="UNPROVEN_ANALYTIC_CERTIFICATE_ERROR", pass_3mm=False, error=str(error_value))
                else:
                    certificate = dict(status="ENDPOINT_FAILED_NO_WHOLE_CURVE_PASS_CLAIM", pass_3mm=False)
            segments.append(segment)
            child_results.append(dict(original_child_id=child["original_child_id"],
                class_name=child["weld_type"] + "/" + child["original_kind"],
                endpoint_maximum_mm=ends, sampled_corresponding_maximum_mm=float(error[0, index].max()),
                full_geometry_certificate=certificate, predicted_length_mm=length,
                physical_minimum_pass=is_long_enough(child["original_kind"], length),
                pass_3mm=bool(ends <= 3. and certificate["pass_3mm"] and is_long_enough(child["original_kind"], length))))
        try:
            topology = validate_component_geometry(parent["mode"], segments, view["gaps"])
            topology_pass = True
        except ProcessGeometryValidationError as error_value:
            topology, topology_pass = dict(error=str(error_value)), False
        if not topology_pass:
            for child in child_results:
                child["pass_3mm"] = False
        rows.append(dict(original_parent_id=parent["original_parent_id"], fixed_query=match["query"],
            selected_whole_parent_view=selected, whole_parent_topology=topology,
            children=child_results, matching_reselected=False))
    counts = {name: dict(G=0, pass_count=0) for name in ("overall", "horizontal/line", "vertical/line", "horizontal/arc")}
    for row in rows:
        for child in row["children"]:
            for name in ("overall", child["class_name"]):
                counts[name]["G"] += 1
                counts[name]["pass_count"] += int(child["pass_3mm"])
    assert counts["overall"]["G"] == targets["original_declared_child_count"]
    return dict(counts=counts, parent_diagnoses=rows, fixed_query_only=True,
                conditional_geometry_not_detection_accuracy=True, full_contract_pass=False)
