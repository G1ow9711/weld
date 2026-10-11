"""Exact distance to all observed finite segments; no geometry is discarded.

Vertex distance gives an upper bound. A segment that can improve it must have
its midpoint inside that bound plus the segment's half length. Length buckets
index every segment once and avoid losing long segments between sparse vertices.
This is observed geometry, not weld classification or material identity.
"""
import numpy as np
from scipy.spatial import cKDTree

VERSION = 'ALL_FINITE_SEGMENT_LENGTH_BUCKET_EXACT_DISTANCE_235_v1'


class ObservedSegmentDistance:
    def __init__(self, start_mm, end_mm):
        self.start = np.array(start_mm, np.float64, copy=True)
        self.end = np.array(end_mm, np.float64, copy=True)
        if (self.start.ndim != 2 or self.start.shape[1:] != (3,)
                or self.end.shape != self.start.shape or not len(self.start)
                or not np.isfinite(self.start).all() or not np.isfinite(self.end).all()):
            raise ValueError('All finite nonempty N by 3 segment endpoints required')
        self.vector = self.end - self.start
        self.squared = np.einsum('ij,ij->i', self.vector, self.vector)
        half = np.sqrt(self.squared) * .5
        middle = self.start + self.vector * .5
        if not np.isfinite(half).all() or not np.isfinite(middle).all():
            raise ValueError('Unrepresentable finite segment lengths or midpoints')
        self.vertex_tree = cKDTree(np.concatenate((self.start, self.end)), copy_data=True)
        _, exponent = np.frexp(half)
        exponent[half == 0.] = -2048
        self.buckets = []
        for value in np.unique(exponent):
            ids = np.flatnonzero(exponent == value)
            self.buckets.append((ids, cKDTree(middle[ids], copy_data=True), float(half[ids].max())))
        self.coordinate_scale = max(1., float(np.abs(self.start).max()),
                                    float(np.abs(self.end).max()), float(half.max()))
        for value in (self.start, self.end, self.vector, self.squared):
            value.flags.writeable = False

    def query(self, points_mm, chunk_rows=64):
        points = np.asarray(points_mm, np.float64)
        if (points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all()
                or isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1):
            raise ValueError('Finite M by 3 points and positive integer chunk size required')
        distances = np.empty(len(points), np.float64)
        witnesses = np.empty(len(points), np.int64)
        candidates = np.empty(len(points), np.int64)
        for begin in range(0, len(points), chunk_rows):
            batch = points[begin:begin+chunk_rows]
            upper, _ = self.vertex_tree.query(batch, k=1, eps=0., workers=1)
            scale = max(self.coordinate_scale, float(np.abs(batch).max()))
            pad = 64. * np.finfo(np.float64).eps * scale
            eligible = [[] for _ in batch]
            for ids, tree, maximum_half in self.buckets:
                radius = np.nextafter(upper + maximum_half + pad, np.inf)
                rows = tree.query_ball_point(batch, radius, p=2., eps=0., workers=1,
                                             return_sorted=True)
                for j, local_ids in enumerate(rows):
                    if local_ids:
                        eligible[j].append(ids[np.asarray(local_ids, np.int64)])
            for j, blocks in enumerate(eligible):
                if not blocks:
                    raise ValueError('Index lost the segment incident on the nearest vertex')
                ids = np.concatenate(blocks)
                offset = batch[j] - self.start[ids]
                denominator = np.where(self.squared[ids] > 0., self.squared[ids], 1.)
                parameter = np.clip(np.einsum('ij,ij->i', offset, self.vector[ids]) / denominator, 0., 1.)
                residual = offset - parameter[:, None] * self.vector[ids]
                squared = np.einsum('ij,ij->i', residual, residual)
                best = int(np.argmin(squared))
                distances[begin+j] = np.sqrt(squared[best])
                witnesses[begin+j] = ids[best]
                candidates[begin+j] = len(ids)
        return distances, witnesses, candidates
