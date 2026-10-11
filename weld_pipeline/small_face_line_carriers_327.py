"""Additional finite LINE carriers from observed small vertical faces.

Accepts XYZ and a geometry dictionary previously computed from those XYZ.
It neither reads labels nor claims reviewed material/contact/weld identities.
Support holes remain split. Every returned interval has original point rows.
"""
import numpy as np

from weld_pipeline.small_finite_surface_events_v3 import small_surface_planes


VERSION = "pure_XYZ_small_finite_face_line_carriers_327_v1"


def small_face_line_carriers(xyz_mm, geometry):
    xyz = np.asarray(xyz_mm, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("Finite original millimeter XYZ required")
    planes, statistics = small_surface_planes(xyz, geometry)
    levels = np.asarray(geometry["horizontal_levels"], dtype=np.float64)
    candidates = []
    for zi, height in enumerate(levels):
        raw_rows = np.flatnonzero(np.abs(xyz[:, 2] - height) < .015)
        near = xyz[raw_rows]
        for pi, plane in enumerate(planes):
            if not plane["z_range_mm"][0] - .05 <= height <= plane["z_range_mm"][1] + .05:
                continue
            normal = np.asarray(plane["normal_xy"], dtype=np.float64)
            direction = np.asarray(plane["direction_xy"], dtype=np.float64)
            mask = np.abs(near[:, :2] @ normal - plane["offset_mm"]) < .12
            rows = raw_rows[mask]
            if len(rows) < 2:
                continue
            values = xyz[rows, :2] @ direction
            order = np.argsort(values, kind="stable")
            rows, values = rows[order], values[order]
            unique = np.unique(np.round(values, 3))
            differences = np.diff(unique)
            step = float(np.median(differences[differences > 1])) if np.any(differences > 1) else 8.
            breaks = np.r_[0, np.flatnonzero(differences > max(15., 1.8 * step)) + 1, len(unique)]
            for begin, end in zip(breaks[:-1], breaks[1:]):
                if end - begin < 2 or unique[end - 1] - unique[begin] < 5.:
                    continue
                lower, upper = float(unique[begin]), float(unique[end - 1])
                support = rows[(values >= lower - .0005) & (values <= upper + .0005)]
                xy = normal[None] * float(plane["offset_mm"]) + np.array([lower, upper])[:, None] * direction[None]
                points = np.column_stack((xy, np.full(2, height)))
                candidates.append(dict(id=f"small_face_h_{len(candidates)+1:05d}", type="horizontal",
                    segments=[dict(kind="line", points=points.tolist())],
                    source_kind="RAW_SMALL_FINITE_FACE_AT_OBSERVED_HORIZONTAL_LEVEL",
                    small_plane_id=int(pi), observed_height_index=int(zi),
                    raw_support_rows=support.tolist(),
                    material_identity="MATERIAL_UNVERIFIED", weld_identity="UNSELECTED_GEOMETRY_HYPOTHESIS"))
    return candidates, dict(version=VERSION, small_faces=planes, statistics=statistics,
        horizontal_levels=levels.tolist(), candidate_count=len(candidates),
        target_reads_for_generation=0, STEP_reads=0, full_contract_pass=False)
