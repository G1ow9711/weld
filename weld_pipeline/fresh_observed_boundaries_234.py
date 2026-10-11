"""Current-XYZ discovery and every observed finite-face boundary, without truth.

This is an observation provider, not a weld/material classifier. The frozen
plane discovery scope does not certify arbitrary curved surfaces or solids.
"""
from dataclasses import asdict

import numpy as np

from .finite_plate_binding_v1_204b import FinitePlateCandidates, CandidateConfig
from .primitive_incidence_209 import array_sha
from .runtime_primitive_provider_adapter_212 import ObservationBinding, RuntimeQueryPath, ProviderError
from .typed_point_features_217 import trace_observed_boundary_paths

VERSION = 'FRESH_XYZ_ALL_FINITE_FACE_OBSERVED_BOUNDARY_STREAM_234_v1'
DETECTOR_SCOPE = 'FROZEN_XYZ_VERTICAL_AND_HORIZONTAL_FINITE_SAMPLED_PATCHES'


class FreshObservedBoundaries:
    def __init__(self, xyz_mm, binding, config=CandidateConfig()):
        if not isinstance(binding, ObservationBinding) or binding.kind != 'runtime_pcd':
            raise ProviderError('Current runtime PCD observation binding required')
        xyz = np.asarray(xyz_mm)
        if xyz.dtype != np.float64 or xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ProviderError('Complete finite float64 original millimeter XYZ rows required')
        self.xyz, self.binding = xyz, binding
        self.before_xyz_sha256 = array_sha(xyz)
        self.candidates = FinitePlateCandidates.from_raw_xyz(xyz, config)
        if array_sha(xyz) != self.before_xyz_sha256:
            raise ProviderError('Discovery changed original raw XYZ/order')
        self.source_receipt = dict(version=VERSION, current_observation_sha256=binding.observation_sha256,
            original_xyz_array_sha256=self.before_xyz_sha256, original_points=len(xyz),
            source_names_or_coordinates_used_as_identity=False, cached_geometry_or_features_read=False,
            actual_fresh_plane_and_face_discovery=True, detector_scope=DETECTOR_SCOPE,
            discovery_config=asdict(config), finite_faces=len(self.candidates.faces),
            material_pair_candidates=len(self.candidates.plates),
            original_raw_row_order_and_values_preserved=True, GT_STEP_Excel_used=False,
            material_identity_verified=False, weld_identity_or_type_verified=False,
            no_short_boundary_edge_deleted=True, full_contract_pass=False)

    def iter_face_boundaries(self):
        arrays = self.candidates.support_arrays()
        for face in self.candidates.faces:
            raw_ids = arrays[face['unique_point_array_key']]
            edges = arrays[face['boundary_array_key']]
            traced = trace_observed_boundary_paths(self.xyz[raw_ids], edges, raw_ids)
            if (len(edges) != face['boundary_edge_count']
                    or not traced['every_original_edge_retained_exactly_once']):
                raise ProviderError('Every original finite-face boundary edge must be retained')
            yield dict(face_id=face['face_id'], plane_id=face['plane_id'], component_id=face['component_id'],
                       raw_representative_ids_sha256=array_sha(raw_ids), original_edges_sha256=array_sha(edges),
                       face=face, trace=traced)
        if array_sha(self.xyz) != self.before_xyz_sha256:
            raise ProviderError('Boundary processing changed raw XYZ')

    def unclassified_query_contexts(self, boundary, step_mm=2.):
        if boundary['geometry'] is None:
            return ()
        return tuple(RuntimeQueryPath(boundary['geometry'], context, self.binding, step_mm=step_mm)
                     for context in ('horizontal', 'vertical'))
