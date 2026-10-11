"""Closest points on an immutable finite triangle union; no material inference.

AABB nodes bound all original triangles. No convex-hull filling, positive-gap
joining, score pruning, or label-derived geometry enters this distance API.
Degenerate triangles retain their closed segment/point support.
"""
from __future__ import annotations

import heapq
import numpy as np

VERSION = 'closed_finite_triangle_union_distance_243_v1'


def closest_on_triangles(point, triangles):
    """Return each closed triangle's closest XYZ, squared distance and barycentric.

    The plane interior and all three closed edges are separate candidates. An
    exact-zero cross product retains the edge union rather than filling a face.
    """
    p = np.asarray(point, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.float64)
    if p.shape != (3,) or t.ndim != 3 or t.shape[1:] != (3, 3):
        raise ValueError('One XYZ and N by 3 by 3 triangles required')
    if not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError('Finite coordinates required')
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    e0, e1 = b-a, c-a
    n = np.cross(e0, e1)
    n2 = np.einsum('ij,ij->i', n, n)
    height = np.divide(np.einsum('ij,ij->i', p-a, n), n2,
                       out=np.zeros(len(t)), where=n2 > 0)
    projected = p-height[:, None]*n
    u = np.divide(np.einsum('ij,ij->i', np.cross(projected-a, e1), n), n2,
                  out=np.zeros(len(t)), where=n2 > 0)
    v = np.divide(np.einsum('ij,ij->i', np.cross(e0, projected-a), n), n2,
                  out=np.zeros(len(t)), where=n2 > 0)
    interior = (n2 > 0) & (u >= 0) & (v >= 0) & (u+v <= 1)
    witness = projected.copy()
    barycentric = np.stack((1-u-v, u, v), axis=1)
    squared = np.where(interior, np.einsum('ij,ij->i', p-projected, p-projected), np.inf)
    for start, end in ((0, 1), (1, 2), (2, 0)):
        edge = t[:, end]-t[:, start]
        edge2 = np.einsum('ij,ij->i', edge, edge)
        q = np.clip(np.divide(np.einsum('ij,ij->i', p-t[:, start], edge), edge2,
                             out=np.zeros(len(t)), where=edge2 > 0), 0, 1)
        on_edge = t[:, start]+q[:, None]*edge
        distance = np.einsum('ij,ij->i', p-on_edge, p-on_edge)
        improve = distance < squared
        squared[improve] = distance[improve]
        witness[improve] = on_edge[improve]
        barycentric[improve] = 0
        barycentric[improve, start] = 1-q[improve]
        barycentric[improve, end] = q[improve]
    return witness, squared, barycentric


class TriangleUnionDistance:
    """Complete branch-and-bound over original triangle ordinals.

    ``triangle_ordinal`` refers to the supplied order, including aliases. Bounds
    are expanded outwards by floating-point enclosure only; returned distances
    always use original finite triangles. AABB ties are retained for inspection.
    """
    def __init__(self, triangles, *, leaf_size=32):
        triangles = np.asarray(triangles, dtype=np.float64)
        if (triangles.ndim != 3 or triangles.shape[1:] != (3, 3)
                or len(triangles) == 0 or not np.isfinite(triangles).all()):
            raise ValueError('Nonempty finite N by 3 by 3 triangles required')
        if isinstance(leaf_size, bool) or not isinstance(leaf_size, int) or leaf_size < 1:
            raise ValueError('Positive integer leaf_size required')
        self.triangles = triangles
        self.lower = np.nextafter(triangles.min(axis=1), -np.inf)
        self.upper = np.nextafter(triangles.max(axis=1), np.inf)
        self.centres = self.lower/2+self.upper/2
        self.order = np.arange(len(triangles), dtype=np.int64)
        self.nodes = []

        def build(begin, end):
            ids = self.order[begin:end]
            lo, hi = self.lower[ids].min(axis=0), self.upper[ids].max(axis=0)
            node_id = len(self.nodes)
            self.nodes.append(None)
            if end-begin <= leaf_size:
                self.nodes[node_id] = (lo, hi, -1, -1, begin, end)
                return node_id
            extent = np.ptp(self.centres[ids], axis=0)
            axis = int(np.argmax(extent))
            half = (end-begin)//2
            permutation = np.argpartition(self.centres[ids, axis], half)
            self.order[begin:end] = ids[permutation]
            middle = begin+half
            left, right = build(begin, middle), build(middle, end)
            self.nodes[node_id] = (lo, hi, left, right, begin, end)
            return node_id

        build(0, len(triangles))
        # Construction-only arrays do not remain attached to this observer.
        del self.centres, self.lower, self.upper

    @staticmethod
    def _lower_bound_squared(point, node):
        lo, hi = node[:2]
        delta = np.maximum(np.maximum(lo-point, point-hi), 0)
        squared = float(delta @ delta)
        # Enclosure permits extra work; it never expands a triangle or fills a gap.
        return max(0., squared-128*np.finfo(float).eps*max(1., squared))

    def query(self, point):
        p = np.asarray(point, dtype=np.float64)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError('Finite XYZ required')
        queue = [(self._lower_bound_squared(p, self.nodes[0]), 0)]
        best2, best_id, best_point, best_barycentric = np.inf, -1, None, None
        visited, examined = 0, 0
        while queue:
            bound, node_id = heapq.heappop(queue)
            if bound > best2:
                break
            node = self.nodes[node_id]
            visited += 1
            if node[2] >= 0:
                for child in node[2:4]:
                    child_bound = self._lower_bound_squared(p, self.nodes[child])
                    if child_bound <= best2:
                        heapq.heappush(queue, (child_bound, child))
                continue
            ids = self.order[node[4]:node[5]]
            points, distances, barycentrics = closest_on_triangles(p, self.triangles[ids])
            examined += len(ids)
            best_local = int(np.argmin(distances))
            candidate2, candidate_id = float(distances[best_local]), int(ids[best_local])
            if candidate2 < best2 or (candidate2 == best2 and candidate_id < best_id):
                best2, best_id = candidate2, candidate_id
                best_point, best_barycentric = points[best_local], barycentrics[best_local]
        if best_id < 0:
            raise ArithmeticError('No finite triangle distance obtained')
        return dict(distance_mm=float(np.sqrt(best2)), triangle_ordinal=best_id,
                    closest_point_mm=best_point, barycentric=best_barycentric,
                    nodes_visited=visited, triangles_examined=examined)
