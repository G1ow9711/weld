"""Offline E3 label preparation. No inference, training or truth approval.

Keep each original field, plate side, signed normal and along-path position.
Legacy sentinels become explicit states only where the bound process rules
already define them. Unknown applicability and review remain unresolved.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .contract_field_acceptance_297 import FIELDS

VERSION = "OFFLINE_CANONICAL_E16_PAYLOAD_330_v1"
DISTANCES = set(FIELDS[:12]) - {"normal_1", "normal_2"}
HOLES = {"start_hole_height", "end_hole_height"}
CENSORED = {"front_wall", "back_wall", "start_base_edge", "end_base_edge"}
BOOLEANS = {"has_eave", "side_plate_split"}


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def unit_vector(value):
    return (isinstance(value, (list, tuple)) and len(value) == 3
            and all(finite_number(v) for v in value)
            and abs(math.hypot(*value) - 1.) <= .001)


def canonical_field(field, raw, reference_key):
    """Prepare a typed label, never infer an unknown value or applicability."""
    if field not in FIELDS or not isinstance(raw, dict) or not reference_key:
        raise ValueError("Original field record and fixed reference required")
    original = deepcopy(raw)
    value, state = raw.get("value"), raw.get("state", "unknown")
    output = dict(
        key=reference_key, reference_key=reference_key, status="unknown", value=None,
        source=raw.get("source"), original_record=original,
        known_payload_available=False,
        original_official_acceptance_truth=raw.get("official_acceptance_truth") is True,
        review_and_applicability_not_approved_by_adapter=True,
    )
    if raw.get("known") is not True or not isinstance(raw.get("source"), str) or not raw["source"]:
        output["reason"] = "ORIGINAL_UNKNOWN_OR_UNBOUND_SOURCE"
        return output
    if state == "no_object":
        if field not in HOLES or not (value is None or (finite_number(value) and value == 0)):
            output["reason"] = "NO_OBJECT_ENCODING_HAS_NO_BOUND_FIELD_SEMANTICS"
            return output
        output.update(status="no_object", value=None, known_payload_available=True,
                      encoding_source="CUSTOMER_PROCESS_RULES_302_ZERO_MEANS_NO_HOLE")
        return output
    if state == "out_of_range":
        if field not in CENSORED or not (value is None or (finite_number(value) and value > 50)):
            output["reason"] = "CENSOR_ENCODING_CONFLICT"
            return output
        output.update(status="out_of_range", value=None, known_payload_available=True,
                      censor=dict(threshold_mm=50., relation="gt", units="mm"),
                      encoding_source="CUSTOMER_PROCESS_RULES_302_DISTANCE_GREATER_THAN50")
        return output
    if state != "value":
        output["reason"] = "UNRESOLVED_OR_UNSUPPORTED_ORIGINAL_STATE"
        return output
    if field in DISTANCES:
        valid = (finite_number(value) and value >= 0
                 and (not field.startswith("thickness_") or value > 0)
                 and (field not in CENSORED or value <= 50))
    elif field.startswith("normal_"):
        valid = unit_vector(value)
    elif field in BOOLEANS:
        valid = type(value) is bool
    else:
        valid = False
    if not valid:
        output["reason"] = "PAYLOAD_INVALID_OR_STRUCTURAL_CATEGORY_UNFROZEN"
        return output
    output.update(status="value", value=deepcopy(value), known_payload_available=True)
    return output


def canonical_material_samples(samples, reference_key):
    """Validate STEP point payloads without upgrading them to E1/E4 truth."""
    if not isinstance(samples, list):
        raise ValueError("Ordered original sample list required")
    result, previous = [], -math.inf
    for index, raw in enumerate(samples):
        if not isinstance(raw, dict):
            raise ValueError("Sample must preserve the original material record")
        fraction = raw.get("fraction")
        xyz, thickness, normals = (raw.get(k) for k in ("xyz_world_mm", "thickness_mm", "normals"))
        valid = (finite_number(fraction) and 0 <= fraction <= 1 and fraction > previous
                 and isinstance(xyz, list) and len(xyz) == 3 and all(finite_number(v) for v in xyz)
                 and isinstance(thickness, list) and len(thickness) == 2
                 and all(finite_number(v) and v > 0 for v in thickness)
                 and isinstance(normals, list) and len(normals) == 2 and all(unit_vector(v) for v in normals)
                 and raw.get("fixed_plate_order") is True)
        if finite_number(fraction):
            previous = fraction
        result.append(dict(
            key=f"{reference_key}/material_sample:{index}",
            original_sample=deepcopy(raw), pointwise_payload_valid=bool(valid),
            plate_1_thickness_mm=thickness[0] if valid else None,
            plate_2_thickness_mm=thickness[1] if valid else None,
            plate_1_signed_normal=deepcopy(normals[0]) if valid else None,
            plate_2_signed_normal=deepcopy(normals[1]) if valid else None,
            fraction=deepcopy(fraction), xyz_world_mm=deepcopy(xyz),
            original_customer_reviewed=raw.get("customer_reviewed") is True,
            formal_E_truth_certified=False,
            full_surface_contact_and_plate_order_review_required=True,
        ))
    return result
