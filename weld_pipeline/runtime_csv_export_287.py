"""C1 CSV and analytic-curve references; unknown material truth is not zero."""
import csv
import io
import json
import os
from pathlib import Path
import time

ATTRIBUTES = ("thickness_1", "thickness_2", "normal_1", "normal_2", "front_wall", "back_wall",
    "start_height", "end_height", "start_hole_height", "end_hole_height", "start_base_edge", "end_base_edge",
    "has_eave", "hole_edge", "side_plate_split", "segment_reason")
GEOMETRY_COLUMNS = ("workpiece_id", "weld_id", "parent_id", "segment_index", "segment_count", "geometry_type",
    "weld_type", "weld_side", "weld_side_state", "weld_side_reason", "is_closed", "parent_is_closed",
    "length_mm", "start_x_mm", "start_y_mm", "start_z_mm", "end_x_mm", "end_y_mm", "end_z_mm",
    "direction", "direction_definition", "gap_before", "boundary_start_ref", "boundary_end_ref", "points_file", "arc_parameters_json",
    "confidence", "confidence_definition", "status", "error_code", "reason")
COLUMNS = GEOMETRY_COLUMNS + tuple(column for field in ATTRIBUTES for column in
    (field, field+"_state", field+"_reason", field+"_source", field+"_confidence"))


def atomic_bytes(path, contents):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".%d.tmp" % os.getpid())
    with temporary.open("xb") as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())
    deadline = time.monotonic()+15.
    while True:
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.025)


def atomic_json(path, values):
    atomic_bytes(path, json.dumps(values, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8"))


def empty_row(workpiece_id, weld_id):
    row = {column: "" for column in COLUMNS}
    row.update(workpiece_id=workpiece_id, weld_id=weld_id,
        weld_side_state="unknown", weld_side_reason="WELD_SIDE_LABEL_SCHEMA_NOT_FROZEN")
    for field in ATTRIBUTES:
        row.update({field+"_state": "unknown", field+"_reason": "ATTRIBUTE_NOT_TRAINED_OR_SCHEMA_NOT_FROZEN",
            field+"_source": "WORLD_MODEL_285_NO_ATTRIBUTE_SUPERVISION"})
    return row


def write_csv(path, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, stream.getvalue().encode("utf-8"))


def export_decoded(directory, workpiece_id, decoded):
    directory = Path(directory)
    if decoded["runtime_attributes_trained"] or decoded["GT_or_STEP_reads"]:
        raise ValueError("This exporter is for the unchanged untrained-attribute pure B runtime")
    rows, sequence = [], 0
    for parent in decoded["parents"]:
        parent_id = "p%04d" % parent["query_index"]
        boundary_path = Path("boundaries")/(parent_id+".json")
        atomic_json(directory/boundary_path, dict(units="mm", parent_id=parent_id,
            coordinates_mm=parent["boundaries_mm"], gaps=parent["gaps"], is_closed=parent["is_closed"]))
        for segment in parent["segments"]:
            sequence += 1
            weld_id = "w%06d" % sequence
            curve_path = Path("curves")/(weld_id+".json")
            atomic_json(directory/curve_path, segment)
            row = empty_row(workpiece_id, weld_id)
            first, last = segment["points"][0], segment["points"][-1]
            index = segment["segment_index"]
            row.update(parent_id=parent_id, segment_index=index, segment_count=parent["child_count"],
                geometry_type=segment["kind"], weld_type=segment["weld_type"],
                is_closed=segment["is_closed"], parent_is_closed=parent["is_closed"],
                length_mm=segment["length_mm"], start_x_mm=first[0], start_y_mm=first[1], start_z_mm=first[2],
                end_x_mm=last[0], end_y_mm=last[1], end_z_mm=last[2],
                direction=(1 if segment["kind"] == "line" or segment["arc_parameters"]["sweep_rad"] > 0 else -1),
                direction_definition="ORDERED_ENDPOINTS" if segment["kind"] == "line" else "SIGNED_SWEEP_AROUND_EXPORTED_CIRCLE_PLANE_NORMAL",
                gap_before="START" if index == 0 else parent["gaps"][index-1],
                boundary_start_ref=boundary_path.as_posix()+"#"+str(segment["boundary_refs"][0]),
                boundary_end_ref=boundary_path.as_posix()+"#"+str(segment["boundary_refs"][1]),
                points_file=curve_path.as_posix(), arc_parameters_json=json.dumps(segment.get("arc_parameters"),
                    ensure_ascii=False, allow_nan=False) if segment["kind"] == "arc" else "",
                confidence=parent["presence_probability"], confidence_definition="UNCALIBRATED_PARENT_PRESENCE_PROBABILITY",
                status="GEOMETRY_PREDICTED_ATTRIBUTES_UNKNOWN")
            rows.append(row)
    invalid_rows = 0
    for invalid in decoded["invalid_predictions"]:
        for child in range(invalid["counted_false_children"]):
            sequence += 1
            invalid_rows += 1
            row = empty_row(workpiece_id, "w%06d" % sequence)
            kind = invalid["predicted_kinds"][min(child, len(invalid["predicted_kinds"])-1)]
            weld = invalid["predicted_weld_types"][min(child, len(invalid["predicted_weld_types"])-1)]
            row.update(parent_id="p%04d" % invalid["query_index"], segment_index=child,
                segment_count=invalid["counted_false_children"], geometry_type={1:"line", 2:"arc"}.get(kind, "unknown"),
                weld_type="horizontal" if weld == 0 else "vertical", status="INVALID_PREDICTED_GEOMETRY",
                error_code="PREDICTION_INVALID_OR_CAPACITY_OVERFLOW", reason=invalid["reason"],
                confidence=invalid["presence_probability"], confidence_definition="UNCALIBRATED_PARENT_PRESENCE_PROBABILITY")
            rows.append(row)
    if len(rows) != decoded["total_prediction_children"]:
        raise ValueError("Published prediction denominator lost an invalid or valid row")
    write_csv(directory/"welds.csv", rows)
    atomic_json(directory/"invalid_predictions.json", decoded["invalid_predictions"])
    return dict(published_rows=len(rows), valid_geometry_rows=len(rows)-invalid_rows,
        invalid_prediction_rows=invalid_rows, invalid_predictions_retained=True,
        attributes_trained=False, attributes_unknown_not_zero=True, CSV_encoding="UTF-8",
        CSV_field_count=len(COLUMNS), full_contract_pass=False)
