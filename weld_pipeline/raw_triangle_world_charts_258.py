"""Exact source-recorded world charts, independent of weld/material truth."""
from fractions import Fraction as F
from hashlib import sha256
import math
import struct

from weld_pipeline.concrete_curve_certificate_250 import _sqrt_fraction_bounds

VERSION = 'ORIGINAL_RAW_TRIANGLE_WORLD_CHARTS_AND_EXPLICIT_PORTALS_258_v1'


def exact(value):
    if isinstance(value, F):
        return value
    if type(value) is int:
        return F(value)
    if type(value) is float and math.isfinite(value):
        return F.from_float(value)
    raise ValueError('Finite exact source value or finite float literal required')


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


class RawTriangleChart:
    def __init__(self, points_mm, raw_ids, *, face_id=None, triangle_ordinal=None):
        if len(points_mm) != 3 or len(raw_ids) != 3 or any(len(point) != 3 for point in points_mm):
            raise ValueError('All original three XYZ corners and raw identities required')
        if any(type(raw_id) is not int or raw_id < 0 for raw_id in raw_ids):
            raise ValueError('Nonnegative original raw row IDs required')
        self.points = tuple(tuple(exact(float(value)) for value in point) for point in points_mm)
        self.raw_ids = tuple(raw_ids)
        self.face_id, self.triangle_ordinal = face_id, triangle_ordinal
        self.deltas = sub(self.points[1], self.points[0]), sub(self.points[2], self.points[0])
        self.normal = cross(*self.deltas)
        if any(self.normal):
            self.rank = 2
            dropped = max(range(3), key=lambda axis: abs(self.normal[axis]))
            self.axes = tuple(axis for axis in range(3) if axis != dropped)
        else:
            self.rank = int(any(self.deltas[0]) or any(self.deltas[1]))
            self.axes = None

    def point_exact(self, lambda1, lambda2, *, require_inside=True):
        first, second = exact(lambda1), exact(lambda2)
        weights = 1 - first - second, first, second
        if require_inside and any(weight < 0 for weight in weights):
            raise ValueError('Barycentric point outside its original finite raw triangle')
        return tuple(sum(weight * point[axis] for weight, point in zip(weights, self.points))
                     for axis in range(3))

    def point_record(self, lambda1, lambda2):
        point = self.point_exact(lambda1, lambda2)
        exported = [float(value) for value in point]
        squared = sum((exact(value) - coordinate) ** 2 for value, coordinate in zip(exported, point))
        return dict(point_mm=exported, world_numeric_export_upper_mm=_sqrt_fraction_bounds(squared)[1],
                    original_raw_support_rank=self.rank, material_identity_verified=False)

    def local_mm(self, point):
        if self.rank != 2:
            raise ValueError('Lower-dimensional original support cannot become a2D chart')
        point = tuple(exact(value) for value in point)
        return tuple(point[axis] - self.points[0][axis] for axis in self.axes)

    def point_from_local_mm(self, uv, *, require_inside=True):
        if self.rank != 2 or len(uv) != 2:
            raise ValueError('Exact nondegenerate original3D local chart required')
        i, j = self.axes
        a, b = self.deltas
        determinant = a[i] * b[j] - a[j] * b[i]
        u, v = map(exact, uv)
        lambda1 = (u * b[j] - v * b[i]) / determinant
        lambda2 = (a[i] * v - a[j] * u) / determinant
        return self.point_exact(lambda1, lambda2, require_inside=require_inside)

    def edge_point_exact(self, raw_a, raw_b, alpha):
        if type(raw_a) is not int or type(raw_b) is not int or raw_a not in self.raw_ids or raw_b not in self.raw_ids:
            raise ValueError('Both canonical raw edge endpoints must belong to the declared original triangle')
        value = exact(alpha)
        if not 0 <= value <= 1:
            raise ValueError('Original shared edge alpha must be in the finite unit interval')
        a, b = self.points[self.raw_ids.index(raw_a)], self.points[self.raw_ids.index(raw_b)]
        return tuple((1 - value) * x + value * y for x, y in zip(a, b))


def geometry_alias(source_sha256, raw_ids):
    return sha256(bytes.fromhex(source_sha256) + struct.pack('<3q', *sorted(raw_ids))).hexdigest()


def certify_world_line(points):
    points = [tuple(exact(value) for value in point) for point in points]
    if len(points) < 2 or any(len(point) != 3 for point in points):
        raise ValueError('Complete ordered world points required')
    edges = [sub(b, a) for a, b in zip(points, points[1:])]
    axis = next((edge for edge in edges if any(edge)), None)
    if axis is None:
        raise ValueError('Positive world line required; source zero edges remain separate')
    if any(any(cross(axis, edge)) or (any(edge) and dot(axis, edge) <= 0) for edge in edges):
        raise ValueError('Whole child has a world bend or backtrack and cannot be labelled line')
    return dict(status='PASS_COMPLETE_ORDERED_COLLINEAR_WORLD_LINE',
                original_zero_edges=sum(not any(edge) for edge in edges), continuous_GT_error_verified=False)


def certify_exact_affine_circle_image(u_column, v_column):
    u, v = tuple(map(exact, u_column)), tuple(map(exact, v_column))
    if len(u) != 3 or len(v) != 3 or dot(u, u) <= 0 or dot(u, u) != dot(v, v) or dot(u, v) != 0:
        raise ValueError('General affine UV-circle image lacks an exact world-circle similarity witness')
    return dict(status='PASS_EXACT_RECORDED_AFFINE_CIRCLE_SIMILARITY',
                scale_squared_numerator=str(dot(u, u).numerator),
                scale_squared_denominator=str(dot(u, u).denominator),
                whole_child_arc_witness=False, continuous_GT_error_verified=False)
