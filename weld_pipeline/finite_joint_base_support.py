"""Finite local support hypotheses in the contact base's own plane coordinates.

Sample-supported triangles are hypotheses, not proof of a closed solid or
customer truth. Missing interior sampling and unresolved holes stay unknown.
"""
import numpy as np
from scipy.spatial import cKDTree, Delaunay, QhullError

from weld_pipeline.local_base_sampling import sampling_fill_radius

VERSION = 'finite_joint_base_plane_uv_support_v1'
PLANE_TOLERANCE_MM = .15
JOINT_TOLERANCE_MM = 3.


def unit(vector):
    value = np.asarray(vector, float).copy()
    length = np.linalg.norm(value)
    if value.shape != (3,) or not np.isfinite(value).all() or length < 1e-8:
        raise ValueError('Finite nondegenerate three-dimensional direction required')
    return value / length


class FiniteJointBases:
    def __init__(self, xyz):
        self.xyz = np.asarray(xyz, float)
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3 or not np.isfinite(self.xyz).all():
            raise ValueError('Finite XYZ in mm required')
        self.axes, self.planes = {}, {}

    def candidates(self, point, base_normal):
        q, n = np.asarray(point, float), unit(base_normal)
        if n[np.argmax(np.abs(n))] < 0:
            n *= -1
        axis = tuple(n.round(7))
        if axis not in self.axes:
            self.axes[axis] = self.xyz @ n
        projected = self.axes[axis]
        nearby = projected[np.abs(projected - q @ n) <= JOINT_TOLERANCE_MM]
        if len(nearby) < 20:
            return []
        bins, counts = np.unique(np.round(nearby / .05).astype(np.int64), return_counts=True)
        offsets = []
        for index in np.argsort(counts)[::-1]:
            coarse = bins[index] * .05
            values = nearby[np.abs(nearby - coarse) <= PLANE_TOLERANCE_MM]
            if len(values) < 20:
                continue
            offset = float(np.median(values))
            if any(abs(offset - other) <= 2 * PLANE_TOLERANCE_MM for other in offsets):
                continue
            if np.quantile(np.abs(values - offset), .95) > PLANE_TOLERANCE_MM:
                continue
            offsets.append(offset)
        result = []
        for offset in sorted(offsets, key=lambda value: abs(value - q @ n)):
            key = (*axis, round(offset, 3))
            if key not in self.planes:
                # A normal-derived basis makes the same tilted finite material
                # observable at every Z without collecting other parallel faces.
                reference = np.eye(3)[np.argmin(np.abs(n))]
                u = unit(np.cross(n, reference))
                v = unit(np.cross(n, u))
                basis = np.stack((u, v))
                points = self.xyz[np.abs(projected - offset) <= PLANE_TOLERANCE_MM]
                uv = np.unique(np.round(points @ basis.T, 5), axis=0)
                self.planes[key] = dict(normal=n.copy(), offset=offset, basis=basis,
                                        uv=uv, tree=cKDTree(uv), cache={})
            result.append(self.planes[key])
        return result

    def support(self, point, base_normal, toward_interior):
        q = np.asarray(point, float)
        attempts = []
        for plane in self.candidates(q, base_normal):
            n, basis, offset, tree = (plane[k] for k in ('normal', 'basis', 'offset', 'tree'))
            projected_q = q + n * (offset - q @ n)
            inward3 = np.asarray(toward_interior, float)
            inward3 = inward3 - n * (inward3 @ n)
            if np.linalg.norm(inward3) < 1e-8:
                return None, dict(reason='DEGENERATE_CHILD_IN_BASE_PLANE')
            inward = unit(inward3) @ basis.T
            lateral = np.array([-inward[1], inward[0]])
            uvq = projected_q @ basis.T
            ids = tree.query_ball_point(uvq, 48.)
            nearest = float(tree.query(uvq, k=1)[0])
            failure = dict(plane_normal=n.tolist(), plane_offset_mm=offset,
                           endpoint_to_plane_mm=float(abs(q @ n-offset)),
                           closest_joint_sample_mm=nearest, local_points=len(ids))
            if len(ids) < 9 or nearest > JOINT_TOLERANCE_MM:
                attempts.append({**failure, 'reason':'NO_SAMPLED_JOINT_ON_FINITE_BASE'})
                continue
            local = tree.data[ids]
            coordinates = (local - uvq) @ np.stack((inward, lateral)).T
            # Two outline rows cannot prove an intervening sheet of material.
            transverse = np.unique(np.round(coordinates[:, 1] / .15).astype(np.int64))
            if len(transverse) < 3:
                attempts.append({**failure, 'reason':'BASE_INTERIOR_ONLY_OUTLINE_ROWS'})
                continue
            radius, sampling = sampling_fill_radius(tree, uvq, inward, lateral)
            if sampling['state'] != 'LOCAL_2D_SAMPLING_GAPS_MEASURED':
                attempts.append({**failure, 'reason':sampling['state']})
                continue
            try:
                triangulation = Delaunay(local)
            except QhullError:
                attempts.append({**failure, 'reason':'NO_TWO_DIMENSIONAL_BASE_TRIANGULATION'})
                continue
            # Empty holes and separated islands must not be filled merely by a
            # convex hull. Reject triangles that exceed the measured local grid.
            triangle_points = local[triangulation.simplices]
            edges = np.stack((triangle_points[:, 1]-triangle_points[:, 0],
                              triangle_points[:, 2]-triangle_points[:, 1],
                              triangle_points[:, 0]-triangle_points[:, 2]), axis=1)
            du = sampling['inward_spacing_mm']; dv = sampling['lateral_spacing_mm']
            scaled_edges = np.stack((edges @ inward / du, edges @ lateral / dv), axis=-1)
            supported_triangles = np.linalg.norm(scaled_edges, axis=-1).max(1) <= 1.8
            # Validate both an ordinary patch and a narrow, still two-dimensional
            # finite patch. Their sizes are global; no per-workpiece cutoffs.
            accepted = None
            for side in (-1., 1.):
                for distances in ((6., 18.), (2., 6.)):
                    probes = np.asarray([uvq + u * inward + side * v * lateral
                                         for u in (6., 18.) for v in distances])
                    simplices = triangulation.find_simplex(probes)
                    inside = simplices >= 0
                    good = inside.copy()
                    good[inside] &= supported_triangles[simplices[inside]]
                    if good.all():
                        accepted = dict(material_side=side, lateral_probe_mm=list(distances),
                                        probes_world_mm=(probes @ basis + offset*n).tolist())
                        break
                if accepted is not None:
                    break
            if accepted is None:
                attempts.append({**failure, 'reason':'LOCAL_BASE_MATERIAL_PROBES_NOT_SUPPORTED',
                                 'sampling':sampling, 'supported_triangles':int(supported_triangles.sum())})
                continue
            value = dict(version=VERSION, plane_normal=n.tolist(), plane_offset_mm=offset,
                         projected_joint_mm=projected_q.tolist(), local_base_z_mm=float(projected_q[2]),
                         sampling=sampling, support=accepted, local_points=len(ids),
                         reason='SAMPLED_FINITE_BASE_HYPOTHESIS', material_identity_verified=False,
                         hidden_holes_ruled_out=False, reviewed_customer_truth=False)
            return value, dict(attempts=attempts)
        return None, dict(reason='FINITE_JOINT_BASE_NOT_RESOLVED', attempts=attempts)
