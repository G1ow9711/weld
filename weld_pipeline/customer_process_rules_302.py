"""Customer-confirmed process semantics; keep unavailable truth explicit."""
from __future__ import annotations

import math
import numpy as np

VERSION = "CUSTOMER_PROCESS_RULES_20261004_v1"
ENDPOINT_PAIRS = (
    ("FrontWallDistance", "BackWallDistance"),
    ("startHeight", "endHeight"),
    ("StartViaHoleHeight", "EndViaHoleHeight"),
    ("StartBasePlateEdgeDistance", "EndBasePlateEdgeDistance"),
    ("WorkRangeFrontSplit", "WorkRangeBackSplit"),
)


def minimum_length_mm(kind):
    if str(kind).lower() in ("line", "straight"):
        return 5.0
    if str(kind).lower() in ("arc", "circle", "curve", "polyline", "spline"):
        return 2.0
    raise ValueError("Unknown geometry type cannot inherit a length threshold")


def is_long_enough(kind, length_mm):
    value = float(length_mm)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Finite nonnegative millimetre length required")
    return value >= minimum_length_mm(kind)


def orient_path(points_mm, weld_type, up=(0.0, 0.0, 1.0)):
    points = np.asarray(points_mm, dtype=np.float64)
    axis = np.asarray(up, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("An ordered finite XYZ path is required")
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("A physical up vector in the same frame is required")
    axis = axis / norm
    if weld_type == "vertical":
        delta = float((points[-1] - points[0]) @ axis)
        if abs(delta) <= 1e-8:
            return points.copy(), False, "VERTICAL_WITHOUT_HEIGHT_DIFFERENCE_UNRESOLVED"
        reverse = delta > 0
        ordered = points[::-1].copy() if reverse else points.copy()
        rising = np.diff(ordered, axis=0) @ axis
        state = "VERTICAL_TOP_TO_BOTTOM" if np.max(rising) <= 1e-6 else "VERTICAL_PATH_HAS_RISING_INTERVAL_UNRESOLVED"
        return ordered, reverse, state
    if weld_type != "horizontal":
        raise ValueError("Weld type must come from the bound customer annotation")
    # Horizontal progression is unrestricted. A deterministic direction makes
    # the two plate labels reproducible; reverse-equivalent scoring remains.
    reverse = bool(tuple(points[-1]) < tuple(points[0]))
    return points[::-1].copy() if reverse else points.copy(), reverse, "HORIZONTAL_EITHER_DIRECTION"


def reverse_endpoint_attributes(attributes):
    result = dict(attributes)
    for a, b in ENDPOINT_PAIRS:
        result[a], result[b] = attributes.get(b), attributes.get(a)
    return result


def legacy_field(field, value):
    source = "CUSTOMER_EXCEL"
    if value is None or value == "":
        return dict(value=None, state="unknown", source=source, reason="EMPTY_NOT_ZERO_OR_NOT_APPLICABLE")
    if field in ("FrontWallDistance", "BackWallDistance"):
        number = float(value)
        if number == -1:
            return dict(value=None, state="absent_or_out_of_range_unresolved", source=source,
                        legacy_value=-1, reason="PDF_USES_MINUS1_FOR_NO_WALL_AND_DISTANCE_GT50")
        if number > 50:
            return dict(value=None, state="out_of_range", lower_bound_mm=50.0, source=source, legacy_value=number)
        if number < 0:
            return dict(value=None, state="unknown", source=source, legacy_value=number, reason="INVALID_NEGATIVE_DISTANCE")
        return dict(value=number, state="valid", source=source)
    if field in ("StartBasePlateEdgeDistance", "EndBasePlateEdgeDistance"):
        number = float(value)
        if number == 51 or number > 50:
            return dict(value=None, state="out_of_range", lower_bound_mm=50.0, source=source, legacy_value=number)
        if number < 0:
            return dict(value=None, state="unknown", source=source, legacy_value=number)
        return dict(value=number, state="valid", source=source)
    if field in ("StartViaHoleHeight", "EndViaHoleHeight"):
        number = float(value)
        if number == 0:
            return dict(value=0.0, state="absent", source=source, reason="CUSTOMER_ZERO_MEANS_NO_HOLE")
        if number < 0:
            return dict(value=None, state="unknown", source=source, legacy_value=number)
        return dict(value=number, state="valid", source=source)
    if isinstance(value, bool):
        return dict(value=value, state="valid", source=source)
    number = float(value)
    return dict(value=number if math.isfinite(number) else None,
                state="valid" if math.isfinite(number) else "unknown", source=source)


def internal_endpoint_unknown(field):
    return dict(value=None, state="unknown", source="MISSING_CHILD_ENDPOINT_ATTRIBUTE",
                reason="PARENT_ENDPOINT_ATTRIBUTE_CANNOT_BE_COPIED_TO_INTERNAL_CUT", field=field)


def order_contact_plates(pair, directed_tangent):
    """Keep each plate's thickness and outward normal together, using handedness."""
    if len(pair) != 2 or pair[0]["volume"] == pair[1]["volume"]:
        raise ValueError("Two distinct associated plates required")
    normals = [np.asarray(p["outward_normal_world"], dtype=float) for p in pair]
    tangent = np.asarray(directed_tangent, dtype=float)
    if any(not np.isfinite(n).all() or abs(np.linalg.norm(n) - 1) > 1e-5 for n in normals):
        raise ValueError("Unit, signed, material-outward surface normals required")
    product = float(np.cross(normals[0], normals[1]) @ tangent)
    if not math.isfinite(product) or abs(product) <= 1e-8:
        raise ValueError("Plate order is geometrically degenerate")
    return list(pair) if product > 0 else [pair[1], pair[0]]


def process_split_decision(clearance_mm=None, *, true_geometric_gap=False,
                           side_plate_intersection=False, original_cut=False,
                           contact_plate_pair_changes=False):
    reasons = []
    if true_geometric_gap:
        reasons.append("GEOMETRIC_DISCONTINUITY")
    if side_plate_intersection:
        reasons.append("SIDE_PLATE_INTERSECTION")
    if original_cut:
        reasons.append("ORIGINAL_CUSTOMER_CUT")
    if contact_plate_pair_changes:
        reasons.append("CONTACT_PLATE_PAIR_CHANGE")
    if clearance_mm is not None:
        clearance = float(clearance_mm)
        if not math.isfinite(clearance) or clearance < 0:
            raise ValueError("A measured nonnegative clearance is required")
        # Equality was confirmed by the customer earlier; the PDF shows only
        # strict less-than and greater-than cases.
        if clearance <= 35:
            reasons.append("TORCH_CLEARANCE_AT_MOST35")
    return dict(split=bool(reasons), reasons=reasons,
                unmeasured_clearance_is_unknown=clearance_mm is None,
                workpiece_name_used=False)
