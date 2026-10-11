"""Evaluate fixed XYZ plane-pair requests together, restore original output order.

Initial plane proposals are bound in original order before scheduling changes.
645 uses only the fixed world vertical section direction. For a fixed first
plane normal this makes the section projection direction identical in every
request, including rounded-key collisions. No candidate or point is dropped.
"""
from collections import OrderedDict
import hashlib
import time
import numpy as np
from weld_pipeline.endpoint_local_support_trim_661 import TrimmedFiniteSurfaceIndex

VERSION = 'GROUPED_FIRST_PROPOSAL_ENDPOINT_SUPPORT_V1'


def compact_hypothesis(result):
    """Retain scalar and identity witnesses without keeping evicted plane arrays."""
    pairs=[]
    for pair in result['pairs']:
        row={k:v for k,v in pair.items() if k not in ('base','upright')}
        for name in ('base','upright'):
            surface=pair[name]
            row[name]={k:surface[k] for k in ('normal','basis','plane_offset_mm','uv_origin','resolution_mm','component_index','original_rows','boundary_kind','maximum_plane_residual_mm','field_uncertainty_mm')}
            for field,label in (('original_row_indices','original_row_ids_sha256'),('outer_boundary_cells','outer_cells_sha256'),('inner_boundary_cells','inner_cells_sha256')):
                row[name][label]=hashlib.sha256(surface[field].astype(np.int64).tobytes()).hexdigest()
        pairs.append(row)
    return dict(result,pairs=pairs)


class GroupedFiniteSurfaceIndex(TrimmedFiniteSurfaceIndex):
    def ordered_hypotheses_many(self, requests):
        groups = OrderedDict()
        for i,(sequence,a,z) in enumerate(requests):
            q = np.asarray(sequence['xyz'],float)
            chart = np.arange(len(q))*float(sequence['sample_step_mm'])
            midpoint = np.array([np.interp((a+z)/2.,chart,q[:,j]) for j in range(3)])
            normals = np.asarray(sequence['features'])[len(q)//2,60:66].reshape(2,3)
            keys = tuple(self.register_lazy_plane(n,midpoint) if np.linalg.norm(n)>1e-8 else None for n in normals)
            groups.setdefault(keys,[]).append(i)
        first = {key:(n.copy(),p.copy()) for key,(n,p) in self.first_proposals.items()}
        outputs = [None]*len(requests)
        count = self.reconstructed_planes
        started = time.perf_counter()
        for indices in groups.values():
            for i in indices:
                sequence,a,z = requests[i]
                outputs[i] = compact_hypothesis(self.ordered_hypotheses(sequence,a,z))
        for key,(n,p) in first.items():
            assert np.array_equal(n,self.first_proposals[key][0]) and np.array_equal(p,self.first_proposals[key][1])
        self.last_grouped_report = dict(requests=len(requests),plane_pair_groups=len(groups),
            reconstructed_planes=self.reconstructed_planes-count,seconds=time.perf_counter()-started,
            maximum_resident_planes=self.maximum_actual_resident_planes,
            first_plane_parameters_exactly_preserved=True,
            returned_results_do_not_retain_evicted_plane_payloads=True,
            section_direction='FIXED_WORLD_VERTICAL_AS_IN_645',original_output_order_restored=True)
        return outputs


def build_finite_surface_index(current_original_xyz_mm, global_config=None):
    return GroupedFiniteSurfaceIndex(current_original_xyz_mm, **(global_config or {}))
