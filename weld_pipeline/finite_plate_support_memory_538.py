"""Bounded resident finite-plane payloads, preserving the original proposals.

Every warmed key retains its original first normal and point. Reconstruction
calls the unchanged 522 formulas; eviction removes cache storage only.
"""
from collections import OrderedDict
import numpy as np
from weld_pipeline.finite_plate_support_522 import FiniteSurfaceIndex, frame

class BoundedFiniteSurfaceIndex(FiniteSurfaceIndex):
    def __init__(self, current_original_xyz_mm, *, maximum_resident_planes=2, **kwargs):
        super().__init__(current_original_xyz_mm, **kwargs)
        assert type(maximum_resident_planes) is int and maximum_resident_planes >= 2
        self.maximum_resident_planes = maximum_resident_planes
        self.planes = OrderedDict()
        self.first_proposals = {}
        self.maximum_actual_resident_planes = 0
        self.reconstructed_planes = 0

    @staticmethod
    def proposal_key(normal, point):
        n, _ = frame(normal)
        offset = float(np.asarray(point) @ n)
        return (*n.round(7), round(offset, 3))

    def register_lazy_plane(self, normal, point):
        key = self.proposal_key(normal, point)
        if key not in self.first_proposals:
            self.first_proposals[key] = (np.asarray(normal).copy(), np.asarray(point).copy())
        return key

    def propose(self, normal, point):
        key = self.register_lazy_plane(normal, point)
        if key in self.planes:
            self.planes.move_to_end(key)
            return self.planes[key]
        # Evict before reconstruction, without filtering any component/hypothesis.
        while len(self.planes) >= self.maximum_resident_planes:
            self.planes.popitem(last=False)
        first_normal, first_point = self.first_proposals[key]
        result = super().propose(first_normal, first_point)
        assert key in self.planes
        self.reconstructed_planes += 1
        self.maximum_actual_resident_planes = max(self.maximum_actual_resident_planes, len(self.planes))
        assert len(self.planes) <= self.maximum_resident_planes
        return result

def build_finite_surface_index(current_original_xyz_mm, global_config=None):
    return BoundedFiniteSurfaceIndex(current_original_xyz_mm, **(global_config or {}))
