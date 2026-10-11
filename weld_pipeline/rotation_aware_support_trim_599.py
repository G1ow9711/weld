"""Resident-page release only; original finite-plane arithmetic stays frozen587."""
from weld_pipeline.rotation_aware_support_memory_588 import BoundedFiniteSurfaceIndex
from weld_pipeline.reports.goal_contract_training_20261005.trim_owned_current_working_set_437 import trim_current_owned_working_set

class TrimmedFiniteSurfaceIndex(BoundedFiniteSurfaceIndex):
    def propose(self, normal, point):
        key=self.proposal_key(normal,point)
        if key not in self.planes:
            # Only this calling process's pseudohandle; arrays/weights remain.
            self.last_owned_idle_trim=trim_current_owned_working_set()
        return super().propose(normal,point)

def build_finite_surface_index(current_original_xyz_mm, global_config=None):
    return TrimmedFiniteSurfaceIndex(current_original_xyz_mm, **(global_config or {}))
