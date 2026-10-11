"""Complete observed raw support along every nondegenerate finite face edge.

The raw hull edge seeds a direction. All original rows define its measured
finite range, including collinear points omitted from a hull's short facet.
This proposes geometry only: continuity, contact and weld identity are open.
"""
import numpy as np

VERSION = "PURE_XYZ_COMPLETE_RAW_SUPPORT_FOR_ALL_FACE_BOUNDARIES_337_v1"
SUPPORT_MM = .12
MINIMUM_LINE_MM = 5.
ORIGINAL_ROW_BLOCK = 65536


def measured_line_support(xyz, origin, direction):
    parts, lower, upper = [], np.inf, -np.inf
    for begin in range(0, len(xyz), ORIGINAL_ROW_BLOCK):
        block = xyz[begin:begin + ORIGINAL_ROW_BLOCK]
        delta = block - origin
        q = delta @ direction
        residual = delta - q[:, None] * direction
        keep = np.einsum("ni,ni->n", residual, residual) <= SUPPORT_MM ** 2
        local = np.flatnonzero(keep)
        if len(local):
            parts.append(local + begin)
            lower = min(lower, float(q[local].min()))
            upper = max(upper, float(q[local].max()))
    rows = np.concatenate(parts) if parts else np.empty(0, np.int64)
    return rows, lower, upper


def full_support_face_boundary_carriers(xyz_mm, faces, progress=lambda *args, **kwargs: None):
    xyz = np.asarray(xyz_mm, np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("Complete finite original XYZ required")
    statistics = dict(version=VERSION, original_rows=len(xyz), seed_faces=len(faces),
        seed_boundary_edges=0, nondegenerate_seed_edges=0, eligible_full_support_domains=0,
        actual_candidates=0, horizontal_class_hypotheses=0, descending_vertical_class_hypotheses=0,
        support_distance_mm=SUPPORT_MM, minimum_line_mm=MINIMUM_LINE_MM,
        minimum_distinct_support_positions=3, original_row_block=ORIGINAL_ROW_BLOCK,
        every_original_row_examined_for_each_seed=True, finite_range_never_padded=True,
        GT_reads=0, STEP_reads=0, source_or_filename_branches=False,
        material_contact_and_weld_side_unverified=True, full_contract_pass=False)

    def emit():
        domain = 0
        for face in faces:
            points = np.asarray(face["finite_hull_vertices_mm"], np.float64)
            seeds = np.asarray(face["hull_original_row_indices"], np.int64)
            assert points.shape == (len(seeds), 3) and np.array_equal(points, xyz[seeds])
            for edge in range(len(points)):
                statistics["seed_boundary_edges"] += 1
                origin = points[edge]
                direction = points[(edge + 1) % len(points)] - origin
                length = float(np.linalg.norm(direction))
                if length <= 1e-8:
                    continue
                direction /= length
                statistics["nondegenerate_seed_edges"] += 1
                rows, lower, upper = measured_line_support(xyz, origin, direction)
                if len(rows) < 3 or upper - lower < MINIMUM_LINE_MM:
                    continue
                q = (xyz[rows] - origin) @ direction
                if len(np.unique(q)) < 3:
                    continue
                endpoints = origin + np.array([lower, upper])[:, None] * direction
                base = dict(segments=[dict(kind="line", points=endpoints.tolist())],
                    raw_support_rows=rows.tolist(), source_kind="RAW_COMPLETE_FACE_BOUNDARY_LINE_SUPPORT",
                    seed_face_id=int(face["face_id"]), seed_hull_edge_index=edge,
                    seed_original_row_indices=seeds[[edge, (edge + 1) % len(seeds)]].tolist(),
                    seed_boundary_length_mm=length, line_origin_mm=origin.tolist(), line_direction=direction.tolist(),
                    observed_parameter_range_mm=[float(lower), float(upper)], support_distance_mm=SUPPORT_MM,
                    distinct_observed_parameter_positions=len(np.unique(q)),
                    material_identity="MATERIAL_UNVERIFIED", weld_identity="UNSELECTED_GEOMETRY_HYPOTHESIS",
                    oriented_E16_normal_truth=False, actual_contact_and_weld_side_unverified=True,
                    complete_observed_range_may_bridge_unobserved_gaps=True)
                statistics["eligible_full_support_domains"] += 1
                statistics["actual_candidates"] += 1
                statistics["horizontal_class_hypotheses"] += 1
                yield dict(base, id=f"full_boundary_h_{domain:07d}", type="horizontal")
                if abs(float(endpoints[1, 2] - endpoints[0, 2])) > 1e-10:
                    statistics["actual_candidates"] += 1
                    statistics["descending_vertical_class_hypotheses"] += 1
                    yield dict(base, id=f"full_boundary_v_{domain:07d}", type="vertical")
                domain += 1
            progress("PURE_ALL_XYZ_FULL_FACE_BOUNDARY_LINE_SUPPORT", complete_seed_faces=int(face["face_id"]) + 1,
                total_seed_faces=len(faces), actual_candidates=statistics["actual_candidates"],
                seed_boundary_edges=statistics["seed_boundary_edges"], original_rows_retained=len(xyz))
        assert statistics["actual_candidates"] == statistics["horizontal_class_hypotheses"] + statistics["descending_vertical_class_hypotheses"]
        assert statistics["horizontal_class_hypotheses"] == statistics["eligible_full_support_domains"]

    return emit(), statistics
