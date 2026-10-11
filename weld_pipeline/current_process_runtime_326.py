"""Add physical gap, whole-sample reversal and complete invalid-output ledgers.

Previous303/324 stay immutable. This version delegates their verified filtering
and own-arc reversal, then validates each complete original query before export.
"""
from copy import deepcopy
import numpy as np
from . import current_process_runtime_324 as previous
from .world_set_runtime_295 import decode_local_predictions
from .customer_process_rules_302 import is_long_enough
from .process_geometry_constraints_326 import validate_component_geometry, ProcessGeometryValidationError

VERSION = "PREDICTED_COMPONENTS_PHYSICAL_GAPS_FULL_SAMPLE_LEDGER_326_v1"
FIXED_SAMPLE_METADATA = {"material_names", "plate_ids", "board_ids", "fixed_plate_ids", "plate_identity_order",
                        "board_identity_order", "units", "normal_axis_names", "along_path_fields", "sample_axis_fields"}
TANGENT_FIELDS = {"tangent", "tangent_world", "tangent_world_mm", "tangents", "tangents_world", "tangents_world_mm"}


def reverse_along_path_record(record):
    """Reverse complete sample rows/parallel arrays; retain fixed board axes."""
    value = deepcopy(record)
    if not isinstance(value, dict):
        raise ProcessGeometryValidationError("ALONG_PATH_RECORD_MUST_HAVE_EXPLICIT_FIELDS")
    choices = ("samples", "fractions", "xyz_world_mm", "positions", "thickness_mm", "CAD_outward_unit_normals_world")
    first = next((k for k in choices if isinstance(value.get(k), (list, tuple))), None)
    if first is None:
        return value
    count = len(value[first])
    for key, rows in list(value.items()):
        if key in FIXED_SAMPLE_METADATA or not isinstance(rows, (list, tuple)):
            continue
        if len(rows) == count:
            result = list(reversed(deepcopy(rows)))
            if key == "fractions":
                result = [1. - float(f) for f in result]
            elif key in TANGENT_FIELDS:
                result = (-np.asarray(result, np.float64)).tolist()
            elif key == "samples":
                for sample in result:
                    if not isinstance(sample, dict):
                        raise ProcessGeometryValidationError("COMPLETE_ALONG_PATH_SAMPLE_ROWS_REQUIRED")
                    if "fraction" in sample:
                        sample["fraction"] = 1. - float(sample["fraction"])
                    for field in TANGENT_FIELDS:
                        if field in sample:
                            sample[field] = (-np.asarray(sample[field], np.float64)).tolist()
            value[key] = result
        elif key in choices:
            raise ProcessGeometryValidationError("ALONG_PATH_PARALLEL_ARRAY_COUNTS_DISAGREE")
    value["reversal_schema"] = "COMPLETE_FIRST_SAMPLE_AXIS_FIXED_PLATE_AXES_AND_SIGNED_NORMALS_PRESERVED_326"
    return value


def parent_mode(parent, segments):
    mode = parent.get("predicted_mode", parent.get("mode"))
    if mode in (0, 1, 2):
        return ("LINE_CHAIN", "CIRCLE_CHAIN", "MIXED_CHAIN")[mode]
    if mode in ("LINE_CHAIN", "CIRCLE_CHAIN", "MIXED_CHAIN"):
        return mode
    if all(c["kind"] == "arc" for c in segments):
        return "CIRCLE_CHAIN"
    if all(c["kind"] == "line" for c in segments):
        a = np.asarray(segments[0]["points"][0], float)
        delta = np.asarray(segments[0]["points"][-1], float) - a
        length = np.linalg.norm(delta)
        if length > 0:
            values = np.asarray([p for c in segments for p in (c["points"][0], c["points"][-1])], float)
            roundoff = 64 * np.finfo(np.float64).eps * max(1., float(np.abs(values).max()), float(length))
            if np.linalg.norm(np.cross(values - a, delta / length), axis=1).max() <= roundoff:
                return "LINE_CHAIN"
    return "MIXED_CHAIN"


def invalid_query(parent, selected, reason):
    return dict(query_index=parent["query_index"], original_parent_query_index=parent["query_index"],
        predicted_child_count=len(selected), counted_false_children=len(selected),
        predicted_weld_types=[int(c["weld_type"] == "vertical") for c in selected],
        predicted_kinds=[1 if c["kind"] == "line" else 2 for c in selected],
        capacity_overflow=False, reason=reason, retained_unresolved_candidates=deepcopy(selected),
        invalid_output_ledger_version=VERSION, full_contract_pass=False)


def normalize_process_paths(decoded):
    result = deepcopy(decoded)
    published, invalid, short = [], list(result["invalid_predictions"]), []
    for parent in decoded["parents"]:
        original_segments = deepcopy(parent["segments"])
        selected = [c for c in original_segments if is_long_enough(c["kind"], c["length_mm"])]
        try:
            turns = parent.get("turns", int(parent["is_closed"]))
            if turns not in (0, 1) or int(turns) != int(parent["is_closed"]):
                raise ProcessGeometryValidationError("TURN_AND_CLOSURE_PROVIDER_NOT_REGISTERED")
            if parent["is_closed"]:
                if parent.get("closure_gap", "SHARED") != "SHARED":
                    raise ProcessGeometryValidationError("POSITIVE_CLOSURE_GAP_PROVIDER_NOT_REGISTERED")
                if not np.array_equal(np.asarray(original_segments[0]["points"][0]),
                                      np.asarray(original_segments[-1]["points"][-1])):
                    raise ProcessGeometryValidationError("PREDICTED_CLOSURE_COORDINATES_DISAGREE")
            one = previous.normalize_process_paths(dict(parents=[deepcopy(parent)], invalid_predictions=[]))
            short.extend(one["internal_unpublished_short_candidates"])
            if one["invalid_predictions"]:
                invalid.extend(one["invalid_predictions"])
                continue
            components = []
            for component in one["parents"]:
                mode = parent_mode(parent, component["segments"])
                proof = validate_component_geometry(mode, component["segments"], component["gaps"])
                for child in component["segments"]:
                    sequence = child["model_sequence_index"]
                    original_material = original_segments[sequence].get("unordered_material_estimate")
                    if original_material is not None:
                        child["unordered_material_estimate"] = reverse_along_path_record(original_material) if child.get("path_reversed_for_process_rule") else deepcopy(original_material)
                component.update(relation_rebuild_version=VERSION, physical_gap_validation=proof,
                    published_parent_id="query_%d_component_%d" % (parent["query_index"], component["prediction_component_index"]),
                    current_provider_scope="ZERO_OR_ONE_TURN_EXPLICIT_SHARED_CLOSURE",
                    mixed_positive_gap_provider_registered=False)
                components.append(component)
            # Publish only after every component of the original query passes.
            published.extend(components)
        except (ProcessGeometryValidationError, ValueError) as error:
            invalid.append(invalid_query(parent, selected, str(error)))
            # Preserve filtered candidate records even when prior normalization
            # failed before returning; all original query candidates stay traced.
            for index, child in enumerate(original_segments):
                if not is_long_enough(child["kind"], child["length_mm"]):
                    if not any(s["query_index"] == parent["query_index"] and s["model_sequence_index"] == index for s in short):
                        short.append(dict(query_index=parent["query_index"], model_sequence_index=index,
                            kind=child["kind"], length_mm=child["length_mm"], published=False,
                            original_candidate=deepcopy(child), reason="BELOW_GLOBAL_PROCESS_MINIMUM"))
    result.update(version=VERSION, parents=published, invalid_predictions=invalid,
        internal_unpublished_short_candidates=short,
        total_prediction_children=sum(p["child_count"] for p in published) + sum(p["counted_false_children"] for p in invalid),
        present_parent_count=len(published) + len(invalid), invalid_output_ledger_complete=True,
        runtime_GT_or_STEP_queries=0, board_side_exchange_used=False, signed_material_normal_flip_used=False,
        short_filter_relations_rebuilt=True, process_rules_applied_without_GT_or_STEP=True,
        full_contract_pass=False)
    return result


def decode_current_predictions(output):
    return normalize_process_paths(decode_local_predictions(output))
