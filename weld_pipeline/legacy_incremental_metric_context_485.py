"""Metric context computed from current XYZ and the current candidate chart.

The raw cloud range and counts are features, never board-height measurements.
No names, case identifiers, references, labels, stored predictions or I/O.
"""
import numpy as np

CONTEXT_COUNT = 32
CONTEXT_VERSION = "CURRENT_XYZ_CHART_METRIC_CONTEXT32_v1"


def raw_xyz_context(xyz_mm):
    points = np.asarray(xyz_mm)
    assert points.ndim == 2 and points.shape[1] == 3 and len(points) > 0
    assert np.isfinite(points).all()
    return dict(minimum_mm=points.min(axis=0).astype(float),
                maximum_mm=points.max(axis=0).astype(float), point_count=len(points), valid=True)


def chart_world_points(sequence, parameters_mm):
    at = np.asarray(parameters_mm, np.float64)
    if sequence.get("curve_kind") == "arc":
        curve = sequence["curve"]
        radius = float(curve["radius_mm"])
        angle = float(curve["sequence_start_angle_rad"]) + at / radius
        center = np.asarray(curve["center_xy_mm"], np.float64)
        return np.column_stack((center[0] + radius * np.cos(angle),
                                center[1] + radius * np.sin(angle),
                                np.full(at.shape, float(curve["height_mm"]))))
    return np.asarray(sequence["xyz"][0], np.float64)[None] + at[:, None] * np.asarray(sequence["direction"], np.float64)[None]


def metric_context(sequence, left_mm, right_mm, raw_context):
    left, right = float(left_mm), float(right_mm)
    length = (len(sequence["xyz"]) - 1) * float(sequence["sample_step_mm"])
    assert length > 0 and np.isfinite([left, right, length]).all()
    origin = np.asarray(sequence["xyz"][0], np.float64)
    tangent = np.asarray(sequence["direction"], np.float64)
    endpoints = chart_world_points(sequence, np.asarray([left, right]))
    valid = bool(raw_context.get("valid", False))
    cloud_min = np.asarray(raw_context["minimum_mm"], np.float64) if valid else np.zeros(3)
    cloud_max = np.asarray(raw_context["maximum_mm"], np.float64) if valid else np.zeros(3)
    cloud_count = int(raw_context.get("point_count", 0)) if valid else 0
    values = np.r_[
        length / 1000., left / 1000., right / 1000., abs(right - left) / 1000.,
        origin / 1000., tangent,
        cloud_min / 1000., cloud_max / 1000., (cloud_max - cloud_min) / 1000.,
        endpoints[0] / 1000., endpoints[1] / 1000.,
        float(sequence.get("curve_kind") == "arc"), float(valid),
        float(0. <= min(left, right) <= max(left, right) <= length),
        cloud_count / 1e6, len(sequence["xyz"]) / 1e4,
        float(np.isfinite(sequence["features"][[0, -1]]).all()), float(valid and cloud_count > 0),
    ].astype(np.float32)
    assert values.shape == (CONTEXT_COUNT,) and np.isfinite(values).all()
    return values


CONTEXT_MEANINGS = (
    "candidate_length_m", "left_parameter_m", "right_parameter_m", "child_length_m",
    "chart_origin_x_m", "chart_origin_y_m", "chart_origin_z_m",
    "chart_tangent_x", "chart_tangent_y", "chart_tangent_z",
    "raw_cloud_min_x_m", "raw_cloud_min_y_m", "raw_cloud_min_z_m",
    "raw_cloud_max_x_m", "raw_cloud_max_y_m", "raw_cloud_max_z_m",
    "raw_cloud_range_x_m", "raw_cloud_range_y_m", "raw_cloud_range_z_m",
    "child_start_x_m", "child_start_y_m", "child_start_z_m",
    "child_end_x_m", "child_end_y_m", "child_end_z_m",
    "is_arc", "raw_extent_valid", "ordered_parameters_inside_complete_candidate",
    "raw_cloud_point_count_millions", "candidate_sample_count_ten_thousands",
    "current84_endpoint_features_finite", "raw_count_valid",
)
assert len(CONTEXT_MEANINGS) == CONTEXT_COUNT
