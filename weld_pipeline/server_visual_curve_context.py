"""Genuine raw/image/geometry context for input queries on native or new curves.

References and source IDs are not arguments. Image values come from the nearest
original XYZ point, with actual query distances retained by the raw neighborhood
features. The 147-column route is separate from legacy 457-column features.
"""
import math
import numpy as np

from .server_query_point_neighborhood import RawQueryNeighborhood
from .server_tiled_multiview_visual_features import visual_features


class VisualCurveContext:
    def __init__(self, xyz):
        self.raw = RawQueryNeighborhood(xyz)
        self.image, self.audit = visual_features(self.raw.xyz)

    def image_at(self, points):
        distance, ids = self.raw.tree.query(np.asarray(points, np.float64), workers=2)
        return self.image[ids], dict(raw_point_ids=ids, raw_distance_mm=distance)

    def features(self, inp, pid, positions_mm):
        curve = inp.curve(int(pid)); q = np.asarray(positions_mm, np.float64)
        if q.ndim != 1 or not np.isfinite(q).all() or np.any(q < 0) or np.any(q > curve.length):
            raise ValueError('Context queries must be positions on an input curve')
        if not len(q):
            return np.empty((0, 147), np.float32), dict(raw_point_ids=np.empty(0, np.int64))
        points, tangent = curve.at(q), curve.tangent(q)
        supplied = (curve.arc.plane_normal if curve.kind == 'arc'
                    else np.asarray(inp.profiles[int(pid)].get('support_normal', [0., 0., 0.]), float))
        normal = supplied[None] - tangent * (tangent @ supplied)[:, None]
        norms = np.linalg.norm(normal, axis=1)
        known = np.isfinite(normal).all(1) & (norms > 1e-10)
        if not known.all():
            axis = np.eye(3)[np.argmin(abs(tangent[~known]), axis=1)]
            normal[~known] = axis - tangent[~known] * np.sum(axis * tangent[~known], axis=1)[:, None]
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        bases = np.stack((tangent, np.cross(normal, tangent), normal), -1)
        geometry = np.zeros((len(q), 15), np.float32)
        geometry[:, 2] = math.log1p(curve.length); geometry[:, 14] = known
        if curve.kind == 'arc':
            arc = curve.arc; geometry[:, 0] = 1; geometry[:, 1] = arc.is_closed
            geometry[:, 3:6] = np.einsum('ni,nij->nj', arc.center_mm - points, bases) / 1000.
            geometry[:, 6] = math.log(arc.radius_mm / 1000.)
            geometry[:, 7:10] = np.einsum('i,nij->nj', arc.plane_normal, bases)
            geometry[:, 10:13] = np.einsum('i,nij->nj', arc.start_direction, bases)
            geometry[:, 13] = arc.sweep_rad / math.pi
        image, mapping = self.image_at(points)
        values = np.column_stack((self.raw.features(inp, int(pid), q), image, geometry)).astype(np.float32)
        if values.shape != (len(q), 147) or not np.isfinite(values).all():
            raise ValueError('Complete genuine raw/image/geometry context required')
        return values, dict(mapping, points_mm=points, bases=bases, geometry=geometry,
                            reference_reads=0, legacy_feature_columns_fabricated=False)
