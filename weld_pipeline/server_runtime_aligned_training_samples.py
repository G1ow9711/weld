"""Complete TRAIN labels assigned only to declared deployed fixed queries.

References choose a supervised assignment; they never create a query, carrier,
event or feature column. Out-of-bracket targets keep their free geometry loss
and mask the incompatible carrier-only interval loss.
"""
import math
import numpy as np

from .contract_curve_geometry import CircularArc
from .server_full_training_samples import FullTrainingAssignments, query_frame, geometry_descriptor

FIXED_QUERY_FRACTIONS = (.125, .375, .5, .625, .875)


def fixed_query_positions(inp, pid):
    begin, end = map(int, inp.offsets[pid:pid + 2])
    spans = inp.spans[begin:end, :2]
    return np.unique((spans[:, :1] + np.asarray(FIXED_QUERY_FRACTIONS)[None]
                      * (spans[:, 1:2] - spans[:, :1])).ravel())


class RuntimeAlignedTrainingAssignments(FullTrainingAssignments):
    def build(self, supervision, witnesses, seed):
        sample = super().build(supervision, witnesses, seed)
        children = [c for p in supervision.parents for c in p['children']]
        fixed = {int(pid): fixed_query_positions(self.inp, int(pid))
                 for pid in np.unique(sample['profile_ids'])}
        if any(not len(values) for values in fixed.values()):
            raise ValueError('Complete child assignment has no deployed query')
        for owner, child in enumerate(children):
            pid = int(sample['profile_ids'][owner])
            previous = sample['input_query_positions_mm'][owner]
            candidates = fixed[pid]
            query = float(candidates[np.argmin(abs(candidates - previous))])
            point, basis, known = query_frame(self.inp, pid, query)
            curve = self.curves[pid]
            if child['original_kind'] == 'line':
                endpoints = np.asarray(child['representation']['endpoints_mm'], float)
            else:
                arc = CircularArc.from_parameters(child['representation']['arc_parameters'])
                endpoints = arc.point_at(np.array([0., arc.length_mm]))
                sample['targets']['arc_shape'][owner] = np.r_[
                    (arc.center_mm - point) @ basis / 1000, math.log(arc.radius_mm / 1000),
                    arc.plane_normal @ basis, arc.start_direction @ basis, arc.sweep_rad / math.pi]
            sample['targets']['line_points'][owner] = (endpoints - point) @ basis / 1000
            sample['targets']['frame_axes'][owner] = supervision.raw_export_axes[owner] @ basis
            sample['targets']['frame_offsets'][owner] = (
                supervision.frame_translations[owner] - endpoints) @ basis / 1000
            lo, hi = sample['targets']['interval_ratios'][owner].astype(float)
            ratio = query / curve.length
            sample['targets']['interval_known'][owner] &= lo <= ratio <= hi
            sample['geometry'][owner] = geometry_descriptor(curve, point, basis, known)
            sample['input_query_positions_mm'][owner] = query
            sample['input_query_xyz'][owner] = point
            sample['input_query_bases'][owner] = basis
        for pid in fixed:
            ids = np.flatnonzero(sample['profile_ids'] == pid)
            sample['features'][ids] = self.inp.features_at(pid, sample['input_query_positions_mm'][ids])
        for row, (left, right) in enumerate(zip(sample['link_left'], sample['link_right'])):
            basis = sample['input_query_bases'][left]
            delta = (sample['input_query_xyz'][right] - sample['input_query_xyz'][left]) @ basis / 1000
            tangent = sample['input_query_bases'][right][:, 0] @ basis
            sample['link_relative'][row] = np.r_[delta, tangent, np.linalg.norm(delta),
                                     float(sample['profile_ids'][left] == sample['profile_ids'][right])]
        sample['report'].update(input_queries_are_frozen_grid_points=False,
                               input_queries_are_deployed_fixed_fragment_positions=True,
                               fixed_query_fractions=list(FIXED_QUERY_FRACTIONS),
                               incompatible_carrier_bracket_losses_masked=int(
                                   (sample['targets']['interval_known'] == False).sum()),
                               free_geometry_targets_retained_for_all_children=True)
        return sample
