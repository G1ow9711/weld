"""Unselected finite column hypotheses from all original XYZ rows.

The quantization grid groups observed support; it does not move raw points.
Complete observed ranges and fixed-gap components are both retained. No
reference weld, material identity, source name or expected count is an input.
Both process hypotheses are emitted; contact and weld selection remain open.
"""
import numpy as np

VERSION = "PURE_ALL_XYZ_COMPLETE_OBSERVED_VERTICAL_COLUMNS_336_v1"
XY_BIN_MM = .01
MINIMUM_DISTINCT_HEIGHTS = 3
MINIMUM_LINE_MM = 5.
COMPONENT_GAP_MM = 25.


def observed_vertical_column_carriers(xyz_mm, progress=lambda *args, **kwargs: None):
    xyz = np.asarray(xyz_mm, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("All finite original XYZ rows are required")
    scaled = xyz[:, :2] / XY_BIN_MM
    if np.abs(scaled).max() >= np.iinfo(np.int64).max - 1:
        raise ValueError("XY grouping exceeds integer range")
    keys = np.rint(scaled).astype(np.int64)
    del scaled
    order = np.lexsort((xyz[:, 2], keys[:, 1], keys[:, 0]))
    ordered_keys = keys[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.any(ordered_keys[1:] != ordered_keys[:-1], axis=1))]
    ends = np.r_[starts[1:], len(order)]
    assert len(order) == len(xyz) and int(np.sum(ends - starts)) == len(xyz)
    statistics = dict(version=VERSION, original_rows=len(xyz), original_rows_grouped=len(order),
        observed_xy_groups=len(starts), xy_bin_mm=XY_BIN_MM,
        quantization="ROUND_TO_NEAREST_INTEGER_TIES_TO_EVEN", minimum_distinct_heights=MINIMUM_DISTINCT_HEIGHTS,
        minimum_line_mm=MINIMUM_LINE_MM, component_gap_mm=COMPONENT_GAP_MM,
        actual_candidates=0, complete_observed_column_domains=0, additional_gap_component_domains=0,
        horizontal_class_hypotheses=0, descending_vertical_class_hypotheses=0,
        maximum_support_transverse_distance_mm=0., all_original_rows_retained=True,
        all_eligible_observed_columns_retained=True, finite_ranges_never_extended=True,
        GT_reads=0, STEP_reads=0, source_or_filename_branches=False,
        material_contact_and_weld_side_unverified=True, full_contract_pass=False)

    def emit():
        domain = 0

        def records(rows, key, source_kind):
            nonlocal domain
            z = xyz[rows, 2]
            heights = np.unique(z)
            if len(heights) < MINIMUM_DISTINCT_HEIGHTS or heights[-1] - heights[0] < MINIMUM_LINE_MM:
                return
            xy = xyz[rows, :2].mean(axis=0)
            distance = np.linalg.norm(xyz[rows, :2] - xy, axis=1)
            assert distance.max() <= np.sqrt(2.) * XY_BIN_MM + 1e-9
            top = np.r_[xy, heights[-1]]
            bottom = np.r_[xy, heights[0]]
            span = float(heights[-1] - heights[0])
            base = dict(segments=[dict(kind="line", points=[top.tolist(), bottom.tolist()])],
                raw_support_rows=rows.tolist(), source_kind=source_kind,
                column_quantized_xy_key=key.tolist(), xy_bin_mm=XY_BIN_MM,
                distinct_observed_height_count=len(heights),
                line_origin_mm=top.tolist(), line_direction=[0., 0., -1.],
                observed_parameter_range_mm=[0., span],
                observed_height_range_mm=[float(heights[0]), float(heights[-1])],
                maximum_support_transverse_distance_mm=float(distance.max()),
                material_identity="MATERIAL_UNVERIFIED", weld_identity="UNSELECTED_GEOMETRY_HYPOTHESIS",
                oriented_E16_normal_truth=False, actual_contact_and_weld_side_unverified=True,
                complete_observed_range_may_bridge_unobserved_gaps=source_kind == "RAW_COMPLETE_OBSERVED_VERTICAL_COLUMN")
            statistics["maximum_support_transverse_distance_mm"] = max(
                statistics["maximum_support_transverse_distance_mm"], float(distance.max()))
            statistics["actual_candidates"] += 1
            statistics["horizontal_class_hypotheses"] += 1
            yield dict(base, id=f"observed_column_h_{domain:07d}", type="horizontal")
            statistics["actual_candidates"] += 1
            statistics["descending_vertical_class_hypotheses"] += 1
            yield dict(base, id=f"observed_column_v_{domain:07d}", type="vertical")
            domain += 1

        for group, (begin, end) in enumerate(zip(starts, ends)):
            rows = order[begin:end]
            if len(rows) >= MINIMUM_DISTINCT_HEIGHTS:
                heights = xyz[rows, 2]
                eligible = len(np.unique(heights)) >= MINIMUM_DISTINCT_HEIGHTS and heights[-1] - heights[0] >= MINIMUM_LINE_MM
                if eligible:
                    key = ordered_keys[begin]
                    statistics["complete_observed_column_domains"] += 1
                    yield from records(rows, key, "RAW_COMPLETE_OBSERVED_VERTICAL_COLUMN")
                    breaks = 1 + np.flatnonzero(np.diff(heights) > COMPONENT_GAP_MM)
                    if len(breaks):
                        component_starts, component_ends = np.r_[0, breaks], np.r_[breaks, len(rows)]
                        for first, last in zip(component_starts, component_ends):
                            component = rows[first:last]
                            values = np.unique(xyz[component, 2])
                            if len(values) >= MINIMUM_DISTINCT_HEIGHTS and values[-1] - values[0] >= MINIMUM_LINE_MM:
                                statistics["additional_gap_component_domains"] += 1
                                yield from records(component, key, "RAW_OBSERVED_VERTICAL_COLUMN_GAP_COMPONENT")
            if group % 5000 == 0 or group + 1 == len(starts):
                progress("PURE_ALL_ORIGINAL_XYZ_OBSERVED_VERTICAL_COLUMNS", complete_xy_groups=group + 1,
                    total_xy_groups=len(starts), original_rows_retained=len(xyz),
                    actual_candidates=statistics["actual_candidates"],
                    complete_observed_column_domains=statistics["complete_observed_column_domains"])
        assert statistics["actual_candidates"] == 2 * (statistics["complete_observed_column_domains"] + statistics["additional_gap_component_domains"])
        assert statistics["actual_candidates"] == statistics["horizontal_class_hypotheses"] + statistics["descending_vertical_class_hypotheses"]

    return emit(), statistics
