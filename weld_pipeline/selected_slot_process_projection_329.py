"""Project learned selected parameters onto process constraints, without GT.

Selected slots, carrier, common learned pose and learned q are the only inputs.
Slots never change. Shared boundaries use one variable. A positive gap remains
positive. Equal 5/2 mm lengths are legal; no universal minimum gap is invented.
"""
import math
import numpy as np
from scipy.optimize import linprog
from .bounded_analytic_residuals_327 import BoundedAnalyticGeometry

VERSION = "SOURCE_ONLY_SELECTED_SLOT_PROCESS_L1_PROJECTION_329_v1"


def project_selected_parameters(carrier, boundary_nodes, children, gaps, raw_residual):
    geometry = BoundedAnalyticGeometry(carrier, raw_residual)
    identities = [n["node_id"] for n in boundary_nodes]
    if len(set(identities)) != len(identities):
        raise ValueError("Repeated authoritative boundary")
    locations = {value:index for index, value in enumerate(identities)}
    predicted = np.array([n["q_mm"] for n in boundary_nodes], np.float64)
    specs = [carrier.slot_spec(int(n["slot_index"])) for n in boundary_nodes]
    bounds = [(s["lower_mm"], s["upper_mm"]) for s in specs]
    if not np.isfinite(predicted).all() or any(not lo <= q <= hi for q,(lo,hi) in zip(predicted,bounds)):
        raise ValueError("Learned parameter outside selected slot")
    if carrier.kind == "line":
        scale = float(np.linalg.norm(carrier.direction +
            (geometry.end_corrections[1]-geometry.end_corrections[0])/carrier.length_mm))
        minimum = 5.
    else:
        scale = geometry.arc.radius_mm / carrier.arc.radius_mm
        minimum = 2.
    required = minimum / scale
    if required * scale < minimum:
        required = float(np.nextafter(required, math.inf))
    directions = [np.sign(predicted[locations[c["end_node"]]] - predicted[locations[c["start_node"]]]) for c in children]
    if not children or len(gaps) != len(children)-1 or not directions[0] or len(set(directions)) != 1:
        raise ValueError("Nonempty consistent learned native traversal required")
    direction = float(directions[0])
    if carrier.weld_type == "vertical" and direction < 0:
        raise ValueError("Vertical selected intervals must travel down")
    n = len(identities)
    rows, right = [], []
    # Numeric gap floor is only the same curve-scale roundoff used by runtime.
    world = geometry.point_at(predicted)
    roundoff = 64*np.finfo(np.float64).eps*max(1.,float(np.abs(world).max()),carrier.length_mm)
    gap_floor = float(np.nextafter(roundoff / scale, math.inf))
    for child in children:
        row = np.zeros(n)
        row[locations[child["start_node"]]] = direction
        row[locations[child["end_node"]]] = -direction
        rows.append(row); right.append(-required)
    for i,gap in enumerate(gaps):
        before,after = children[i]["end_node"], children[i+1]["start_node"]
        if gap == "SHARED":
            if before != after:
                raise ValueError("SHARED must use one authority variable")
        elif gap == "POSITIVE_GAP":
            if before == after:
                raise ValueError("A positive gap cannot share authority")
            row = np.zeros(n)
            row[locations[before]] = direction; row[locations[after]] = -direction
            rows.append(row); right.append(-gap_floor)
        else:
            raise ValueError("Unknown predicted adjacency")
    matrix, limit = np.asarray(rows), np.asarray(right)
    if np.all(matrix @ predicted <= limit):
        return boundary_nodes, dict(version=VERSION, projected=False, changed_slot_indices=False,
            maximum_parameter_change_mm=0., physical_interval_scale=scale, GT_or_STEP_queries=0)
    # L1 distance from the learned q, not from a truth or a reference table.
    a = np.zeros((len(rows)+2*n, 2*n)); b = np.r_[limit,predicted,-predicted]
    a[:len(rows),:n] = matrix
    for i in range(n):
        a[len(rows)+i,i] = 1.; a[len(rows)+i,n+i] = -1.
        a[len(rows)+n+i,i] = -1.; a[len(rows)+n+i,n+i] = -1.
    solved = linprog(np.r_[np.zeros(n),np.ones(n)], A_ub=a, b_ub=b,
        bounds=bounds+[(0.,None)]*n, method="highs",
        options=dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
    if not solved.success:
        raise ValueError("Selected model intervals have no certified process projection")
    values = np.clip(solved.x[:n],[v[0] for v in bounds],[v[1] for v in bounds])
    # Do not publish a solver's tolerant near-feasible short or zero-gap result.
    if np.any(matrix @ values > limit):
        raise ValueError("Selected interval projection did not meet exact process inequalities")
    corrected = [dict(node,q_mm=float(q)) for node,q in zip(boundary_nodes,values)]
    return corrected, dict(version=VERSION, projected=True, changed_slot_indices=False,
        maximum_parameter_change_mm=float(np.abs(values-predicted).max()),
        physical_interval_scale=scale, GT_or_STEP_queries=0, objective="L1_FROM_LEARNED_Q_WITHIN_SELECTED_SLOTS")
