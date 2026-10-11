"""Reverse only declared/registered sample axes; fixed metadata stays fixed."""
from copy import deepcopy
import numpy as np
from .process_geometry_constraints_326 import ProcessGeometryValidationError

TANGENT_FIELDS = {"tangent", "tangent_world", "tangent_world_mm", "tangents", "tangents_world", "tangents_world_mm"}
REGISTERED_SAMPLE_FIELDS = {"samples", "fractions", "xyz_world_mm", "positions", "thickness_mm",
    "CAD_outward_unit_normals_world", "masks", "object_ids", "confidence", "confidence_scores",
    "source_receipts", "provenance_rows", "sample_ids", *TANGENT_FIELDS}
FIXED_FIELDS = {"origin", "origin_mm", "center_mm", "basis_uv", "basis_world", "local_basis", "plane_normal",
    "material_names", "plate_ids", "board_ids", "fixed_plate_ids", "plate_identity_order", "board_identity_order",
    "units", "normal_axis_names", "along_path_fields", "sample_axis_fields"}


def reverse_along_path_record(record):
    if not isinstance(record, dict):
        raise ProcessGeometryValidationError("ALONG_PATH_RECORD_MUST_HAVE_EXPLICIT_FIELDS")
    value = deepcopy(record)
    declared = value.get("sample_axis_fields", [])
    if not isinstance(declared, (list, tuple)) or any(type(k) is not str for k in declared) or len(set(declared)) != len(declared):
        raise ProcessGeometryValidationError("INVALID_EXPLICIT_SAMPLE_AXIS_SCHEMA")
    if set(declared).intersection(FIXED_FIELDS):
        raise ProcessGeometryValidationError("FIXED_METADATA_DECLARED_AS_SAMPLE_AXIS")
    for key in declared:
        if key not in value:
            raise ProcessGeometryValidationError("DECLARED_SAMPLE_FIELD_MISSING:" + key)
    sample_fields = (REGISTERED_SAMPLE_FIELDS.intersection(value) | set(declared)) - FIXED_FIELDS
    if not sample_fields:
        return value
    priority = ("samples", "fractions", "xyz_world_mm", "positions", "thickness_mm", "CAD_outward_unit_normals_world")
    first = next((k for k in priority if k in sample_fields), sorted(sample_fields)[0])
    if not isinstance(value[first], (list, tuple)):
        raise ProcessGeometryValidationError("SAMPLE_AXIS_MUST_BE_AN_EXPLICIT_ARRAY:" + first)
    count = len(value[first])
    for key in sorted(sample_fields):
        rows = value[key]
        if not isinstance(rows, (list, tuple)) or len(rows) != count:
            raise ProcessGeometryValidationError("ALONG_PATH_PARALLEL_ARRAY_COUNTS_DISAGREE:" + key)
        if key == "thickness_mm" and np.asarray(rows, dtype=object).shape != (count, 2):
            raise ProcessGeometryValidationError("THICKNESS_REQUIRES_SAMPLE_AXIS_AND_FIXED_TWO_BOARD_AXIS")
        if key == "CAD_outward_unit_normals_world" and np.asarray(rows, dtype=object).shape != (count, 2, 3):
            raise ProcessGeometryValidationError("NORMALS_REQUIRE_SAMPLE_AXIS_FIXED_TWO_BOARDS_AND_XYZ")
        result = list(reversed(rows))
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
    value["reversal_schema"] = "EXPLICIT_SAMPLE_AXES_FIXED_METADATA_AND_SIGNED_NORMALS_PRESERVED_327"
    return value
