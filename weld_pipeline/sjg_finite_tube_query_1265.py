"""Conservative broad phase for the existing finite-curve projection.

Every finite curve location is at most half a grid step in arc length from a
sample. A point within tube_mm of that location is therefore within
half_step+tube_mm of a sample (triangle inequality). Query this superset, then
apply the unchanged projection and exact distance threshold. No voxel sampling,
GT, STEP, or candidate pruning is used. Standalone, not wired into frozen421.
"""
import numpy as np


def tube_projection(curve,points,tree,tube_mm=.1,grid_step_mm=50.):
    if not np.isfinite(tube_mm) or tube_mm<0 or not np.isfinite(grid_step_mm) or grid_step_mm<=0:
        raise ValueError('Finite nonnegative tube and positive spacing required')
    if tree.n!=len(points):raise ValueError('Tree and original point array must be aligned')
    grid=curve.grid(grid_step_mm);centers=curve.at(grid)
    radius=float(np.max(np.diff(grid))/2+tube_mm)
    radius+=max(1e-6,float(np.max(np.abs(centers)))*np.finfo(float).eps*128)
    blocks=[]
    for first in range(0,len(centers),32):
        neighbours=tree.query_ball_point(centers[first:first+32],radius,p=2.,eps=0.,workers=2)
        blocks.extend(np.asarray(v,np.int64) for v in neighbours if len(v))
    ids=np.unique(np.concatenate(blocks)) if blocks else np.empty(0,np.int64)
    q,dist=curve.project(np.asarray(points[ids]))
    keep=dist<=tube_mm
    return dict(point_ids=ids[keep],q_mm=q[keep],distance_mm=dist[keep],
        broad_phase_points=len(ids),original_points=len(points),grid_centers=len(grid))
