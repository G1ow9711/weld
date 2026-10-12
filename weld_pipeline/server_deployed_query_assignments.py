"""Retain complete targets at the same dense input grid deployment already scores."""
import numpy as np

from .server_finite_carrier_grid import finite_carrier_grid
from .server_full_training_samples import FullTrainingAssignments
from .server_runtime_aligned_training_samples import fixed_query_positions


def eligible_query_positions(inp, pid):
    grid = finite_carrier_grid(inp.curve(int(pid)).length, np.empty(0))['positions_mm']
    return np.unique(np.r_[grid, fixed_query_positions(inp, int(pid))])


class DeployedGridTrainingAssignments(FullTrainingAssignments):
    def build(self, supervision, witnesses, seed):
        sample = super().build(supervision, witnesses, seed)
        for pid in np.unique(sample['profile_ids']):
            ids = np.flatnonzero(sample['profile_ids'] == pid)
            allowed = eligible_query_positions(self.inp, int(pid))
            positions = sample['input_query_positions_mm'][ids]
            if not np.isin(positions, allowed).all():
                raise ValueError('An assigned query is absent from the actual input grid')
            bounds = sample['targets']['interval_ratios'][ids]
            ratio = positions / self.inp.curve(int(pid)).length
            sample['targets']['interval_known'][ids] &= (bounds[:, 0] <= ratio) & (ratio <= bounds[:, 1])
        sample['report'].update(input_queries_are_deployed_dense_grid_positions=True,
            reference_creates_no_query_coordinates=True,
            all_free_geometry_and_process_targets_retained=True,
            proposal_retention_after_scoring_not_certified=True)
        return sample
