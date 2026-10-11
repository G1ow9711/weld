"""Additional finite hypotheses from complete XYZ face support, without GT.

Keep original component intervals. At every actually observed horizontal
intersection, also propose its vertical plane's full measured tangent range.
Cross-height support gives a finite domain, never an infinite or GT-padded ray.
These hypotheses do not certify continuous material contact or weld necessity.
"""
import math
import numpy as np

VERSION = "PURE_XYZ_FULL_FACE_SUPPORT_OBSERVED_HORIZONTAL_DOMAINS_334_v1"


def full_support_horizontal_carriers(ordinary_candidates,geometry):
    planes = geometry["vertical_planes"]
    observed = {}
    for record in ordinary_candidates:
        if record["type"] != "horizontal":
            continue
        if record["segments"][0]["kind"] != "line":
            raise ValueError("Original observed horizontal face intersections required")
        plane_id,level = int(record["plane_id"]),float(record["surface_z_mm"])
        if not 0 <= plane_id < len(planes) or not math.isfinite(level):
            raise ValueError("Bound original XYZ plane/height required")
        observed.setdefault((plane_id,level),[]).append(record["id"])
    candidates = []
    for (plane_id,level),components in observed.items():
        plane = planes[plane_id]
        normal,direction = np.asarray(plane["normal_xy"],float),np.asarray(plane["direction_xy"],float)
        lower,upper = map(float,plane["range_mm"])
        offset = float(plane["offset_mm"])
        if (normal.shape != (2,) or direction.shape != (2,) or not np.isfinite(normal).all()
                or not np.isfinite(direction).all() or not all(math.isfinite(v) for v in (lower,upper,offset))
                or abs(float(np.linalg.norm(normal))-1.) > 1e-8
                or abs(float(np.linalg.norm(direction))-1.) > 1e-8
                or abs(float(normal @ direction)) > 1e-8 or upper-lower < 5.):
            raise ValueError("Finite original measured orthonormal face support required")
        if not plane["z_range_mm"][0]-.05 <= level <= plane["z_range_mm"][1]+.05:
            raise ValueError("Observed horizontal height must meet measured vertical extent")
        a,b = normal*offset+direction*lower,normal*offset+direction*upper
        candidates.append(dict(id=f"full_face_h_{len(candidates):06d}",type="horizontal",
            segments=[dict(kind="line",points=[[*a,float(level)],[*b,float(level)]])],
            plane_id=plane_id,surface_z_mm=level,
            observed_original_horizontal_component_ids=components,
            measured_complete_vertical_plane_range_mm=[lower,upper],
            measured_vertical_plane_normal_xy=normal.tolist(),measured_vertical_plane_direction_xy=direction.tolist(),
            measured_vertical_plane_offset_mm=offset,
            finite_range_source="ALL_ORIGINAL_XYZ_FACE_INLIERS_AT_ALL_HEIGHTS_WITH_OBSERVED_HORIZONTAL_INTERSECTION",
            no_endpoint_padding_or_reference_derived_extension=True,
            physical_continuous_contact_and_weld_side_unverified=True))
    return candidates,dict(version=VERSION,observed_plane_height_pairs=len(observed),
        added_candidates=len(candidates),all_original_horizontal_components_retained=True,
        proposal_range_is_finite_measured_face_support_not_GT=True,
        input_uses_original_XYZ_derived_face_statistics_only=True,source_identifiers_or_filename_branches=False,
        STEP_reads=0,target_reads_for_generation=0,known_weld_or_material_labels_used=False,full_contract_pass=False)
