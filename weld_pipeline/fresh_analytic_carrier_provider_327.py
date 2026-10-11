"""One source-independent pure XYZ LINE/observed horizontal CIRCLE provider.

All proposals remain unselected. Production inference reconstructs this bank
from its original XYZ; offline training banks are not a runtime input. Finite
surface interpretations stay MATERIAL_UNVERIFIED. Unsupported types stay open.
"""
import numpy as np
from .robust_surface_planes import surface_candidates
from .small_face_line_carriers_327 import small_face_line_carriers
from .arc_surface_candidates import arc_surface_candidates
from .contract_curve_geometry import CircularArc
from .analytic_carrier_slots_327 import build_carriers

VERSION = "FRESH_SOURCE_INDEPENDENT_PURE_XYZ_ANALYTIC_CARRIERS_327_v1"


def fresh_analytic_carriers(xyz_mm, progress=None):
    xyz = np.asarray(xyz_mm, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("All finite original millimeter XYZ required")
    if len(xyz) < 8:
        raise ValueError("Insufficient observed support; never invent a carrier")
    report = progress or (lambda stage, **values: None)
    report("FULL_ORIGINAL_XYZ_ROBUST_SURFACES", original_XYZ=len(xyz))
    ordinary, geometry = surface_candidates(xyz)
    report("FULL_ORIGINAL_XYZ_MULTISCALE_SMALL_FACES", ordinary_candidates=len(ordinary))
    small, small_statistics = small_face_line_carriers(xyz, geometry)
    report("FULL_ORIGINAL_XYZ_ORDERED_CIRCLE_SURFACE_RELATIONS", small_candidates=len(small))
    arcs, arc_statistics = arc_surface_candidates(xyz)
    for candidate in arcs:
        parameters = candidate["arc"]
        phase = parameters["start_angle_rad"]
        arc = CircularArc(np.array([*parameters["center_xy_mm"], candidate["height_mm"]]),
            parameters["radius_mm"], np.array([0., 0., 1.]), np.array([np.cos(phase), np.sin(phase), 0.]),
            parameters["sweep_rad"], False)
        candidate["segments"][0]["arc_parameters"] = arc.to_parameters()
    candidates = ordinary + small + arcs
    # Associate exact original row ordinals for proposals that do not already
    # retain them. These are observation references, never material identities.
    for index, candidate in enumerate(candidates):
        if "raw_support_rows" not in candidate:
            carrier = build_carriers([candidate])[0]
            if carrier.kind == "line":
                q = (xyz - carrier.origin_mm) @ carrier.direction
                residual = xyz - carrier.origin_mm - q[:, None] * carrier.direction
                within = (q >= 0.) & (q <= carrier.length_mm)
                mask = within & (np.linalg.norm(residual, axis=1) <= .2)
            else:
                radial = xyz - carrier.arc.center_mm
                height = radial @ carrier.arc.plane_normal
                planar = radial - height[:, None] * carrier.arc.plane_normal
                angle = (np.sign(carrier.arc.sweep_rad) * np.arctan2(planar @ np.cross(carrier.arc.plane_normal, carrier.arc.start_direction),
                    planar @ carrier.arc.start_direction)) % (2 * np.pi)
                mask = (np.abs(height) <= .2) & (np.abs(np.linalg.norm(planar, axis=1) - carrier.arc.radius_mm) <= .2) & (angle <= abs(carrier.arc.sweep_rad))
            candidate["raw_support_rows"] = np.flatnonzero(mask).tolist()
        candidate["material_identity"] = "MATERIAL_UNVERIFIED"
        candidate["weld_identity"] = "UNSELECTED_GEOMETRY_HYPOTHESIS"
    return candidates, dict(version=VERSION, ordinary_candidates=len(ordinary), small_candidates=len(small), arc_candidates=len(arcs),
        all_original_XYZ_processed=len(xyz), ordinary_statistics=geometry, small_statistics=small_statistics,
        arc_statistics=arc_statistics, source_identifiers_or_filename_branches=False,
        target_reads_for_generation=0, STEP_reads=0, learned_label_reads=0,
        raw_support_references_retained=True, runtime_requires_fresh_XYZ_reconstruction=True,
        supported_geometry_scope="FINITE_LINE_AND_VERIFIED_HORIZONTAL_OPEN_CIRCLE_ONLY_NOT_ALL_CONTRACT_TYPES",
        full_contract_pass=False)
