"""204b JSON normalization and explicit approximate-pair measurement limits.

The frozen 204 raw run remains bound to its original 204 module. This additive
revision does not re-extract points or revise that receipt/geometry.
"""
import numpy as np

from weld_pipeline.finite_plate_binding_v1_204 import (CandidateConfig,
    FinitePlateCandidates as _FrozenCandidates)

VERSION = 'finite_plate_binding_candidates_v1_204b'


class FinitePlateCandidates(_FrozenCandidates):
    @classmethod
    def from_raw_xyz(cls, xyz_mm, config=CandidateConfig()):
        """Rebuild every proposal from current XYZ without any cache/labels/CAD.

        Existing raw generators may use two cKDTree workers. This fresh entry
        is prepared but NOT_RUN on the real 204 fixture; 204 used frozen pure
        XYZ engineering proposals and then reassociated actual original points.
        """
        from weld_pipeline.robust_surface_planes import surface_candidates
        from weld_pipeline.small_finite_surface_events_v3 import small_surface_planes
        xyz = np.asarray(xyz_mm, np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError('Finite N by 3 millimeter XYZ required')
        if not len(xyz):
            return cls(xyz, [], config)
        _, geometry = surface_candidates(xyz)
        vertical, _ = small_surface_planes(xyz, geometry)
        hypotheses = [dict(normal=[*p['normal_xy'], 0.], offset_mm=p['offset_mm']) for p in vertical]
        hypotheses.extend(dict(normal=[0., 0., 1.], offset_mm=z) for z in geometry['horizontal_levels'])
        return cls(xyz, hypotheses, config)

    def to_dict(self):
        result = super().to_dict()
        result['version'] = VERSION
        result['config'] = dict(plane_tolerance_mm=float(self.config.plane_tolerance_mm),
            joint_tolerance_mm=float(self.config.joint_tolerance_mm),
            parallel_cosine=float(self.config.parallel_cosine),
            triangle_scaled_edge_limit=float(self.config.triangle_scaled_edge_limit),
            neighbor_count=int(self.config.neighbor_count))
        return result

    def _material_pairs(self):
        super()._material_pairs()
        for pair in self.plates:
            left, right = pair['face_ids']
            a, b = self._geometry[left], self._geometry[right]
            signed = self.faces[right]['offset_mm'] - a['xyz'] @ b['normal']
            pair['signed_separation_at_left_support_range_mm'] = [float(signed.min()), float(signed.max())]
            pair['thickness_measurement_scope'] = 'CANDIDATE_NORMAL_SEPARATION_NOT_CONTRACT_E2_THICKNESS'
            pair['possible_crossing_within_left_support'] = bool(signed.min() <= 0. <= signed.max())
            direction = np.sign(signed[0])
            pair['conditional_material_outward_normals'] = [
                (-direction * np.sign(a['normal'] @ b['normal']) * a['normal']).tolist(),
                (direction * b['normal']).tolist()]
            pair['signed_customer_normal_is_not_confirmed'] = True
