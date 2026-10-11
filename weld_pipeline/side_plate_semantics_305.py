"""Side-local process decisions; callers supply verified finite geometry.

This module does not locate plates, measure clearances, or manufacture cuts.
The live 303/304 training run does not import it.
"""
from __future__ import annotations

import math

VERSION = "SIDE_LOCAL_PROCESS_SEMANTICS_305"
DEVELOPMENT_NEAR_THRESHOLD_MM = 10.0


def _measurement(value, name, *, positive=False):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def side_plate_decision(*, side_relation, finite_side_distance_mm=None,
                        near_threshold_mm=None, torch_clearance_mm=None,
                        torch_clearance_applicable=None,
                        same_measurement_for_both_checks=False):
    """Decide whether this weld side needs a side-plate process cut.

    ``side_relation`` refers to this particular weld side, established from
    the actual plate geometry; screen-left or world-minus-X is insufficient.
    The near threshold is strict (<T).  The 35 mm clearance rule includes
    equality under the earlier customer confirmation. If both observations
    are the same physical gap, use the clearance check once rather than
    silently treating them as two independent measurements.

    None means unresolved, not a negative label. Existing customer cuts
    remain authoritative in the caller; this result never removes them.
    """
    if side_relation not in ("affected", "unaffected", "unknown"):
        raise ValueError("A verified weld-side relation or explicit unknown is required")
    if torch_clearance_applicable not in (True, False, None):
        raise ValueError("Clearance applicability must be true, false, or unknown")
    distance = _measurement(finite_side_distance_mm, "finite side distance")
    threshold = _measurement(near_threshold_mm, "near threshold", positive=True)
    clearance = _measurement(torch_clearance_mm, "torch clearance")
    if clearance is not None and torch_clearance_applicable is not True:
        raise ValueError("A clearance observation needs an applicable local obstruction")
    if same_measurement_for_both_checks and (distance is None or clearance is None):
        raise ValueError("Shared-gap selection requires both explicitly bound observations")
    if same_measurement_for_both_checks and not math.isclose(distance, clearance, abs_tol=1e-6, rel_tol=0):
        raise ValueError("One physical gap cannot carry contradictory measurements")

    reasons = []
    missing = []
    if side_relation == "unknown":
        missing.append("WELD_SIDE_RELATION")
        required = None
    elif side_relation == "unaffected":
        required = False
    else:
        if distance == 0:
            reasons.append("SIDE_PLATE_CONNECTION")
        elif same_measurement_for_both_checks:
            # The PDF already gives a passability threshold for this gap.
            # The customer's example 10 mm is not a second fixed cutoff.
            pass
        elif distance is None or threshold is None:
            missing.append("SIDE_DISTANCE_OR_CONFIGURED_NEAR_THRESHOLD")
        elif distance < threshold:
            reasons.append("FINITE_SIDE_DISTANCE_BELOW_CONFIGURED_THRESHOLD")
        if torch_clearance_applicable is True:
            if clearance is None:
                missing.append("APPLICABLE_TORCH_CLEARANCE")
            elif clearance <= 35.0:
                reasons.append("APPLICABLE_TORCH_CLEARANCE_AT_MOST_35MM")
        elif torch_clearance_applicable is None:
            missing.append("TORCH_CLEARANCE_APPLICABILITY")
        required = True if reasons else (None if missing else False)

    return dict(version=VERSION, side_plate_split_required=required,
                reasons=reasons, unresolved_observations=missing,
                configured_near_threshold_mm=threshold,
                customer_fixed_10mm_threshold=False,
                threshold_role="CONFIGURED_DEVELOPMENT_PARAMETER" if threshold is not None else "UNSET",
                near_rule_suppressed_for_same_physical_gap=bool(same_measurement_for_both_checks),
                cut_positions_mm=None, split_geometry_ready=False,
                geometry_status="LOCAL_SUPPORTED_CUT_POSITIONS_REQUIRED" if required else "NO_NEW_CUT_GEOMETRY_CREATED",
                preserve_existing_customer_cuts=True)
