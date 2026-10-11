"""Validate real ordered analytic gaps; never merge gaps using 3 mm tolerance."""
import math
import numpy as np


class ProcessGeometryValidationError(ValueError):
    pass


def validate_component_geometry(mode, segments, gaps):
    if len(gaps) != len(segments) - 1 or not segments:
        raise ProcessGeometryValidationError("INVALID_COMPONENT_ADJACENCY_COUNT")
    values = [np.asarray(c["points"], np.float64) for c in segments]
    if any(p.ndim != 2 or p.shape[1] != 3 or len(p) < 2 or not np.isfinite(p).all() for p in values):
        raise ProcessGeometryValidationError("NONFINITE_OR_MISSING_COMPONENT_GEOMETRY")
    if mode == "LINE_CHAIN":
        axis = values[-1][-1] - values[0][0]
        length = float(np.linalg.norm(axis))
        if length <= 0:
            raise ProcessGeometryValidationError("DEGENERATE_LINE_PARENT")
        axis /= length
        bounds = [(float((p[0] - values[0][0]) @ axis), float((p[-1] - values[0][0]) @ axis)) for p in values]
        if any(end <= start for start, end in bounds):
            raise ProcessGeometryValidationError("NONPOSITIVE_CHILD_TRAVEL_INTERVAL")
    elif mode == "CIRCLE_CHAIN":
        from .contract_curve_geometry import CircularArc
        arcs = [CircularArc.from_parameters(c["arc_parameters"]) for c in segments]
        first = arcs[0]
        sign = 1. if first.sweep_rad > 0 else -1.
        u, v = first.start_direction, np.cross(first.plane_normal, first.start_direction)
        bounds = []
        for i, arc in enumerate(arcs):
            if np.sign(arc.sweep_rad) != sign:
                raise ProcessGeometryValidationError("CIRCLE_CHILD_TRAVERSAL_SIGNS_DISAGREE")
            if abs(arc.sweep_rad) > 2 * math.pi:
                raise ProcessGeometryValidationError("MULTITURN_PROVIDER_NOT_REGISTERED")
            angle = sign * math.atan2(float(arc.start_direction @ v), float(arc.start_direction @ u))
            start = 0. if i == 0 else angle % (2 * math.pi)
            if i and gaps[i - 1] == "SHARED" and np.array_equal(values[i][0], values[i - 1][-1]):
                start = bounds[-1][1]
            bounds.append((start, start + abs(arc.sweep_rad)))
    else:
        # Mixed parents lack this provider's common travel parameter. Retain an
        # explicit invalid ledger rather than certify an unverified gap.
        if "POSITIVE_GAP" in gaps:
            raise ProcessGeometryValidationError("MIXED_POSITIVE_GAP_TRAVEL_PROVIDER_NOT_REGISTERED")
        bounds = None
    for i, gap in enumerate(gaps, 1):
        previous, current = values[i - 1][-1], values[i][0]
        if gap == "SHARED":
            if not np.array_equal(previous, current):
                raise ProcessGeometryValidationError("SHARED_PREDICTED_BOUNDARY_COORDINATES_DISAGREE")
        elif gap == "POSITIVE_GAP":
            if float(np.linalg.norm(current - previous)) <= 0:
                raise ProcessGeometryValidationError("POSITIVE_GAP_HAS_ZERO_PHYSICAL_LENGTH")
            if bounds is None or bounds[i][0] <= bounds[i - 1][1]:
                raise ProcessGeometryValidationError("POSITIVE_GAP_IS_NOT_POSITIVE_FORWARD_TRAVEL")
        else:
            raise ProcessGeometryValidationError("UNKNOWN_PREDICTED_GAP")
    return dict(positive_physical_gaps_verified=True, no_3mm_gap_merging=True,
                ordered_travel_bounds=bounds)
