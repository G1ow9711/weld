"""Same-cell role order follows complete-instance identity, not projection jitter."""
from .curveonly_01a1ac7c_paired_boundary_v193 import *
from . import curveonly_01a1ac7c_paired_boundary_v193 as previous


def targets(row, chosen):
    result = previous.targets(row, chosen)
    both = result['order_known']
    result['projected_offset_order'] = result['fields'][5].copy()
    result['fields'][5, both] = result['start_instance'][both] != result['end_instance'][both]
    result['shared_cell_order_supervision'] = 'COMPLETE_INSTANCE_IDENTITY'
    support = []
    position = np.arange(len(row['grid']))*row['grid_step_mm']
    for choice, code in zip(chosen, result['codes']):
        if code['full_cycle']:
            continue
        take = np.zeros(len(position), bool)
        for turn in ((-1, 0, 1) if row['closed'] else (0,)):
            s = position+turn*row['length_mm']
            take |= (s >= choice['start_mm']) & (s <= choice['end_mm'])
        if not take.any():
            ids = [code['start_index'], code['end_index']]
            result['fields'][0, ids] = 1.
            support.append(dict(truth=choice['truth'], endpoint_support_cells=ids))
    result['positive_subgrid_support'] = support
    return result
