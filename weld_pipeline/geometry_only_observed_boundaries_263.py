"""Independent complete observed geometry; material enumeration remains unknown."""
from dataclasses import asdict

import numpy as np

from .finite_plate_binding_v1_204b import CandidateConfig, FinitePlateCandidates
from .fresh_observed_boundaries_234 import DETECTOR_SCOPE, FreshObservedBoundaries
from .primitive_incidence_209 import array_sha
from .runtime_primitive_provider_adapter_212 import ObservationBinding, ProviderError


VERSION = "FRESH_XYZ_ALL_FINITE_GEOMETRY_INDEPENDENT_MATERIAL_STAGE_NOT_RUN_263_v1"


class GeometryOnlyCandidates(FinitePlateCandidates):
    def _material_pairs(self):
        # Geometry is already complete when the original constructor calls this.
        # This empty storage is omitted from export; it does not assert no plates.
        self.plates = []
        self.material_enumeration_status = "NOT_RUN_IN_INDEPENDENT_GEOMETRY_BRANCH"

    def bind_child(self, points_mm):
        raise ProviderError("Material contact binding is unavailable in this geometry-only branch")

    def to_dict(self):
        result = super().to_dict()
        result.pop("plate_candidates", None)
        result["version"] = VERSION
        result["material_candidates"] = {
            "status": self.material_enumeration_status,
            "complete_candidate_count": None,
            "identity_verified": False,
            "numerical_material_features_known": False,
        }
        return result


class GeometryOnlyObservedBoundaries(FreshObservedBoundaries):
    def __init__(self, xyz_mm, binding, config=CandidateConfig()):
        if not isinstance(binding, ObservationBinding) or binding.kind != "runtime_pcd":
            raise ProviderError("Current original PCD binding required")
        xyz = np.asarray(xyz_mm)
        if xyz.dtype != np.float64 or xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ProviderError("All original finite float64 millimetre XYZ rows required")
        self.xyz, self.binding = xyz, binding
        self.before_xyz_sha256 = array_sha(xyz)
        self.candidates = GeometryOnlyCandidates.from_raw_xyz(xyz, config)
        if array_sha(xyz) != self.before_xyz_sha256:
            raise ProviderError("Geometry discovery modified raw coordinates or row order")
        self.source_receipt = {
            "version": VERSION, "current_observation_sha256": binding.observation_sha256,
            "original_xyz_array_sha256": self.before_xyz_sha256, "original_points": len(xyz),
            "source_names_or_coordinates_used_as_identity": False, "cached_geometry_or_features_read": False,
            "actual_fresh_plane_and_face_discovery": True, "detector_scope": DETECTOR_SCOPE,
            "discovery_config": asdict(config), "finite_faces": len(self.candidates.faces),
            "material_pair_candidates": None, "material_enumeration_status": "NOT_RUN_IN_INDEPENDENT_GEOMETRY_BRANCH",
            "material_identity_verified": False, "GT_STEP_Excel_used": False,
            "original_raw_row_order_and_values_preserved": True, "no_short_boundary_edge_deleted": True,
            "weld_identity_or_type_verified": False, "full_contract_pass": False,
        }
