"""Validate common analytic authority and full ordered zero/one-turn travel."""
import math
import numpy as np
from .process_geometry_constraints_326 import ProcessGeometryValidationError


def validate_component_geometry(mode, segments, gaps):
    if len(gaps) != len(segments) - 1 or not segments:
        raise ProcessGeometryValidationError("INVALID_COMPONENT_ADJACENCY_COUNT")
    values = [np.asarray(c["points"], np.float64) for c in segments]
    if any(p.ndim != 2 or p.shape[1] != 3 or len(p) < 2 or not np.isfinite(p).all() for p in values):
        raise ProcessGeometryValidationError("NONFINITE_OR_MISSING_COMPONENT_GEOMETRY")
    scale = max(1., *(float(np.abs(p).max()) for p in values),
                *(float(np.linalg.norm(p[-1] - p[0])) for p in values))
    roundoff_mm = 128 * np.finfo(np.float64).eps * scale
    bounds = None
    common_authority = False
    if mode == "LINE_CHAIN":
        origin = values[0][0]
        delta = values[0][-1] - origin
        length = float(np.linalg.norm(delta))
        if length <= 0:
            raise ProcessGeometryValidationError("DEGENERATE_LINE_PARENT")
        axis = delta / length
        bounds = []
        for p in values:
            positions = (p - origin) @ axis
            off_line = p - origin - positions[:, None] * axis
            if float(np.linalg.norm(off_line, axis=1).max()) > roundoff_mm:
                raise ProcessGeometryValidationError("LINE_CHILD_NOT_ON_COMMON_ANALYTIC_AUTHORITY")
            if positions[-1] <= positions[0] or np.any(np.diff(positions) < -roundoff_mm):
                raise ProcessGeometryValidationError("NONPOSITIVE_OR_REVERSED_CHILD_TRAVEL_INTERVAL")
            bounds.append((float(positions[0]), float(positions[-1])))
        common_authority = True
    elif mode == "CIRCLE_CHAIN":
        from .contract_curve_geometry import CircularArc
        arcs = [CircularArc.from_parameters(c["arc_parameters"]) for c in segments]
        first = arcs[0]
        u, v = first.start_direction, np.cross(first.plane_normal, first.start_direction)
        travel_sign = float(np.sign(first.sweep_rad))
        angular_roundoff = max(128 * np.finfo(np.float64).eps, roundoff_mm / first.radius_mm)
        bounds = []
        for i, (arc, points) in enumerate(zip(arcs, values)):
            if float(np.linalg.norm(arc.center_mm - first.center_mm)) > roundoff_mm or abs(arc.radius_mm - first.radius_mm) > roundoff_mm:
                raise ProcessGeometryValidationError("CIRCLE_CHILD_NOT_ON_COMMON_CENTER_RADIUS")
            normal_cosine = float(arc.plane_normal @ first.plane_normal)
            if not (abs(normal_cosine - 1.) <= 128 * np.finfo(np.float64).eps or abs(normal_cosine + 1.) <= 128 * np.finfo(np.float64).eps):
                raise ProcessGeometryValidationError("CIRCLE_CHILD_NOT_ON_COMMON_PLANE")
            # This transport concerns an analytic circle coordinate axis only.
            # Material normals and their signs never enter this validator.
            transported_sign = float(np.sign(arc.sweep_rad)) * (1. if normal_cosine >= 0 else -1.)
            if transported_sign != travel_sign:
                raise ProcessGeometryValidationError("CIRCLE_CHILD_TRAVERSAL_SIGNS_DISAGREE")
            radial = points - first.center_mm
            if float(np.abs(radial @ first.plane_normal).max()) > roundoff_mm or float(np.abs(np.linalg.norm(radial, axis=1) - first.radius_mm).max()) > roundoff_mm:
                raise ProcessGeometryValidationError("CIRCLE_SAMPLES_NOT_ON_COMMON_ANALYTIC_AUTHORITY")
            if abs(arc.sweep_rad) > 2 * math.pi + angular_roundoff:
                raise ProcessGeometryValidationError("MULTITURN_PROVIDER_NOT_REGISTERED")
            direction = radial[0] / first.radius_mm
            angle = travel_sign * math.atan2(float(direction @ v), float(direction @ u))
            start = 0. if i == 0 else angle % (2 * math.pi)
            if i and gaps[i - 1] == "SHARED" and np.array_equal(points[0], values[i - 1][-1]):
                start = bounds[-1][1]
            end = start + abs(arc.sweep_rad)
            if end <= start:
                raise ProcessGeometryValidationError("NONPOSITIVE_CHILD_TRAVEL_INTERVAL")
            if end > 2 * math.pi + angular_roundoff:
                raise ProcessGeometryValidationError("WHOLE_CIRCLE_PARENT_EXCEEDS_ONE_TURN")
            bounds.append((start, end))
        common_authority = True
    elif mode == "MIXED_CHAIN":
        if "POSITIVE_GAP" in gaps:
            raise ProcessGeometryValidationError("MIXED_POSITIVE_GAP_TRAVEL_PROVIDER_NOT_REGISTERED")
    else:
        raise ProcessGeometryValidationError("UNKNOWN_ANALYTIC_PARENT_MODE")
    for i, gap in enumerate(gaps, 1):
        previous, current = values[i - 1][-1], values[i][0]
        if gap == "SHARED":
            if not np.array_equal(previous, current):
                raise ProcessGeometryValidationError("SHARED_PREDICTED_BOUNDARY_COORDINATES_DISAGREE")
            if bounds is not None and abs(bounds[i][0] - bounds[i - 1][1]) > roundoff_mm:
                raise ProcessGeometryValidationError("SHARED_ANALYTIC_TRAVEL_PARAMETERS_DISAGREE")
        elif gap == "POSITIVE_GAP":
            if float(np.linalg.norm(current - previous)) <= 0:
                raise ProcessGeometryValidationError("POSITIVE_GAP_HAS_ZERO_PHYSICAL_LENGTH")
            if bounds is None or bounds[i][0] <= bounds[i - 1][1]:
                raise ProcessGeometryValidationError("POSITIVE_GAP_IS_NOT_POSITIVE_FORWARD_TRAVEL")
        else:
            raise ProcessGeometryValidationError("UNKNOWN_PREDICTED_GAP")
    return dict(positive_physical_gaps_verified=True, no_3mm_gap_merging=True,
        ordered_travel_bounds=bounds, common_analytic_authority_verified=common_authority,
        whole_circle_travel_checked=mode == "CIRCLE_CHAIN", zero_or_one_turn_only=True,
        numerical_roundoff_tolerance_mm=roundoff_mm, material_normal_signs_not_used=True)
