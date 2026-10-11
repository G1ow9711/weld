"""Strict ordered world geometry FIT scorer; supervised targets stay here.

Whole-curve distances use <=1 mm samples against true analytic primitives.
Never flips endpoints, swaps children, ignores invalid outputs or shrinks G.
Formal attributes/side truth and independent delivery remain unaccepted.
"""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from .contract_curve_geometry import CircularArc, sample_arc
from .world_set_runtime_285 import line_segment

VERSION = "STRICT_ORDERED_ANALYTIC_WORLD_FULL_G_FIT_285_v1"
CATEGORIES = ("overall", "horizontal", "vertical", "line", "curve", "Hline", "Vline", "HArc")


def category_keys(weld_type, kind):
    keys = ["overall", weld_type, "line" if kind == "line" else "curve"]
    joint = {("horizontal", "line"):"Hline", ("vertical", "line"):"Vline", ("horizontal", "arc"):"HArc"}.get((weld_type, kind))
    if joint:
        keys.append(joint)
    return keys


def distance_to_primitive(points, segment):
    points = np.asarray(points, np.float64)
    if segment["kind"] == "line":
        a, b = np.asarray(segment["points"], np.float64)[[0, -1]]
        delta = b-a
        length2 = float(delta @ delta)
        if length2 <= 0:
            raise ValueError("Degenerate analytic scoring line")
        parameter = np.clip((points-a) @ delta / length2, 0., 1.)
        return np.linalg.norm(points-(a+parameter[:, None]*delta), axis=1)
    arc = CircularArc.from_parameters(segment["arc_parameters"])
    radial = points - arc.center_mm
    height = radial @ arc.plane_normal
    plane = radial-height[:, None]*arc.plane_normal
    angular = np.arctan2(plane @ np.cross(arc.plane_normal, arc.start_direction), plane @ arc.start_direction)
    ordered_angle = (math.copysign(1., arc.sweep_rad)*angular) % (2*math.pi)
    endpoints = arc.point_at(np.array([0., arc.length_mm]))
    distance = np.minimum(np.linalg.norm(points-endpoints[0], axis=1), np.linalg.norm(points-endpoints[1], axis=1))
    inside = np.ones(len(points), bool) if arc.is_closed else ordered_angle <= abs(arc.sweep_rad)
    circle = np.hypot(height, np.linalg.norm(plane, axis=1)-arc.radius_mm)
    return np.where(inside, np.minimum(distance, circle), distance)


def reference_segment(child):
    value = child["world_representation"]
    return line_segment(*value["endpoints_mm"]) if value["kind"] == "line" else sample_arc(CircularArc.from_parameters(value["arc_parameters"]))


def child_error(prediction, truth, reference):
    if prediction["weld_type"] != truth["weld_type"] or prediction["kind"] != truth["original_kind"]:
        return None
    a, b = np.asarray(prediction["points"], np.float64), np.asarray(reference["points"], np.float64)
    errors = np.linalg.norm(a[[0, -1]]-b[[0, -1]], axis=1)
    if float(errors.max()) > 3.:
        return None
    if prediction["kind"] == "arc":
        pa, ta = [CircularArc.from_parameters(s["arc_parameters"]) for s in (prediction, reference)]
        tangent_p = math.copysign(1., pa.sweep_rad)*np.cross(pa.plane_normal, pa.start_direction)
        tangent_t = math.copysign(1., ta.sweep_rad)*np.cross(ta.plane_normal, ta.start_direction)
        if float(tangent_p @ tangent_t) <= 0:
            return None
    whole = float(max(distance_to_primitive(a, reference).max(), distance_to_primitive(b, prediction).max()))
    if whole > 3.:
        return None
    return dict(ordered_endpoint_errors_mm=errors.tolist(), bidirectional_sampled_maximum_mm=whole,
        sampling_maximum_step_mm=1., no_endpoint_or_sequence_reversal=True)


def score_source(decoded, targets):
    parents, truths = decoded["parents"], targets["parents"]
    if sum(t["original_child_count"] for t in truths) != targets["original_children"]:
        raise ValueError("Full original scoring denominator differs")
    references = [[reference_segment(child) for child in t["children"]] for t in truths]
    cost = np.full((len(parents), len(truths)), 1e12, np.float64)
    accepted = {}
    for pi, prediction in enumerate(parents):
        n = prediction["child_count"]
        for ti, truth in enumerate(truths):
            if (n != truth["original_child_count"] or prediction["is_closed"] != truth["is_closed"]
                    or prediction["gaps"] != truth["gaps"]):
                continue
            rows = {}
            for child in range(n):
                error = child_error(prediction["segments"][child], truth["children"][child], references[ti][child])
                if error is not None:
                    rows[child] = error
            if rows:
                # Cardinality reward dominates all accepted (<3 mm) errors.
                cost[pi, ti] = -1e6*len(rows)+sum(sum(e["ordered_endpoint_errors_mm"])+e["bidirectional_sampled_maximum_mm"] for e in rows.values())
                accepted[(pi, ti)] = rows
    matches = []
    if cost.size:
        pp, tt = linear_sum_assignment(cost)
        for pi, ti in zip(pp.tolist(), tt.tolist()):
            for child, error in accepted.get((pi, ti), {}).items():
                matches.append(dict(predicted_parent_index=pi, original_parent_index=ti, child_index=child,
                    original_child_id=truths[ti]["children"][child]["original_child_id"], **error))
    categories = {key:dict(G=0, predicted=0, TP=0, FP=0) for key in CATEGORIES}
    for truth in truths:
        for child in truth["children"]:
            for key in category_keys(child["weld_type"], child["original_kind"]):
                categories[key]["G"] += 1
    for parent in parents:
        for child in parent["segments"]:
            for key in category_keys(child["weld_type"], child["kind"]):
                categories[key]["predicted"] += 1
    for invalid in decoded["invalid_predictions"]:
        for child in range(invalid["counted_false_children"]):
            weld = invalid["predicted_weld_types"][min(child, len(invalid["predicted_weld_types"])-1)]
            kind = invalid["predicted_kinds"][min(child, len(invalid["predicted_kinds"])-1)]
            # Unknown invalid kinds still count in the global false denominator.
            keys = category_keys("horizontal" if weld == 0 else "vertical", "line" if kind == 1 else "arc") if kind in (1, 2) else ("overall", "horizontal" if weld == 0 else "vertical")
            for key in keys:
                categories[key]["predicted"] += 1
    for match in matches:
        child = truths[match["original_parent_index"]]["children"][match["child_index"]]
        for key in category_keys(child["weld_type"], child["original_kind"]):
            categories[key]["TP"] += 1
    for value in categories.values():
        value["FP"] = value["predicted"]-value["TP"]
    if categories["overall"]["predicted"] != decoded["total_prediction_children"]:
        raise ValueError("Invalid/overflow output ledger lost a false prediction")
    return dict(version=VERSION, categories=categories, matches=matches, original_children=targets["original_children"],
        original_parents=len(truths), invalid_parent_hypotheses=len(decoded["invalid_predictions"]),
        all_original_denominators_retained=True, full_contract_pass=False)


def aggregate_sources(scores):
    categories = {key:{field:sum(s["categories"][key][field] for s in scores) for field in ("G", "predicted", "TP", "FP")} for key in CATEGORIES}
    for value in categories.values():
        g = value["G"]
        value.update(valid_rate=value["TP"]/g if g else None, false_rate_per_G=value["FP"]/g if g else None,
            status="PASS" if g and value["TP"]/g > .95 and value["FP"]/g <= .05 else ("FAIL" if g else "NOT_EVALUATED_NO_GT"))
    return dict(version=VERSION, role="ALL25_FIT_NOT_INDEPENDENT_ACCEPTANCE", categories=categories,
        geometry_gates_pass=all(v["status"]=="PASS" for v in categories.values() if v["G"]),
        formal_E_reviewed_cells=0, each_of_14_E_status="NOT_EVALUATED_UNREVIEWED_LABELS",
        runtime_signed_board_normals_evaluated=False, runtime_board_order_or_side_evaluated=False,
        independent_800_100_200_available=False, full_contract_pass=False)
