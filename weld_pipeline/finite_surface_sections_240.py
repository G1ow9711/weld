"""Analytic sections of sampled triangle unions, independent of weld truth.

These operations describe finite interpolation support. They do not establish
filled material, a solid identity, weld identity, or any customer attribute.
Positive intervals and gaps have no physical length cutoff.
"""
from dataclasses import dataclass

import numpy as np

VERSION = 'FINITE_SAMPLED_TRIANGLE_ANALYTIC_SECTIONS_240_v1'


class UnresolvedSection(ValueError):
    """The supplied geometry cannot support this analytic construction."""


def _vector(value, dimensions):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (dimensions,) or not np.isfinite(result).all():
        raise ValueError('Finite vector of the declared dimension required')
    return result


def _cross2(left, right):
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def _unit_plane(normal, offset):
    value = _vector(normal, 3)
    length = np.linalg.norm(value)
    if length == 0 or not np.isfinite(offset):
        raise ValueError('Nonzero normal and finite plane offset required')
    return value / length, float(offset) / length


@dataclass(frozen=True)
class AnalyticLine:
    origin_mm: np.ndarray
    direction: np.ndarray

    def __post_init__(self):
        origin = _vector(self.origin_mm, 3).copy()
        direction = _vector(self.direction, 3).copy()
        length = np.linalg.norm(direction)
        if length == 0:
            raise ValueError('Nonzero line direction required')
        direction /= length
        origin.flags.writeable = direction.flags.writeable = False
        object.__setattr__(self, 'origin_mm', origin)
        object.__setattr__(self, 'direction', direction)

    def points(self, t_mm):
        t = np.asarray(t_mm, np.float64)
        if not np.isfinite(t).all():
            raise ValueError('Finite physical line coordinates required')
        return self.origin_mm + t[..., None] * self.direction


def plane_intersection(normal_a, offset_a, normal_b, offset_b):
    """Return the common infinite line; finite support must be clipped later."""
    a, da = _unit_plane(normal_a, offset_a)
    b, db = _unit_plane(normal_b, offset_b)
    cross = np.cross(a, b)
    norm = np.linalg.norm(cross)
    if norm <= 64 * np.finfo(np.float64).eps:
        raise UnresolvedSection('Parallel or numerically unresolved plane pair')
    direction = cross / norm
    matrix = np.stack((a, b, direction))
    origin = np.linalg.solve(matrix, [da, db, 0.])
    if not np.isfinite(origin).all():
        raise UnresolvedSection('Nonfinite intersection')
    return AnalyticLine(origin, direction)


def triangle_line_sections(uv_mm, triangles, origin_uv_mm, direction_uv):
    """Clip a parameterized UV line to each accepted triangle.

    The line parameter remains that supplied by the caller. For the projection
    of an AnalyticLine it is physical mm along the original 3D line. Every
    returned row retains its original triangle ordinal. Point contacts remain
    zero-length rows; no short positive row is deleted.
    """
    uv = np.asarray(uv_mm, np.float64)
    indices = np.asarray(triangles)
    origin = _vector(origin_uv_mm, 2)
    direction = _vector(direction_uv, 2)
    if (uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all()
            or indices.ndim != 2 or indices.shape[1] != 3
            or not np.issubdtype(indices.dtype, np.integer)
            or (indices < 0).any() or (indices >= len(uv)).any()):
        raise ValueError('Finite UV rows and valid integer triangle rows required')
    if np.linalg.norm(direction) == 0:
        raise UnresolvedSection('A projected point needs a separate point-support test')
    if not len(indices):
        return dict(intervals=np.empty((0, 2)), triangle_ordinals=np.empty(0, np.int64),
                    total_triangles=0, no_length_threshold=True)
    vertices = uv[indices]
    edge_a = vertices[:, 1] - vertices[:, 0]
    edge_b = vertices[:, 2] - vertices[:, 0]
    determinant = _cross2(edge_a, edge_b)
    if (determinant == 0).any():
        raise UnresolvedSection('Degenerate accepted triangles must be resolved, not removed')
    relative = origin - vertices[:, 0]
    u0 = _cross2(relative, edge_b) / determinant
    v0 = _cross2(edge_a, relative) / determinant
    du = _cross2(direction, edge_b) / determinant
    dv = _cross2(edge_a, direction) / determinant
    values = np.column_stack((u0, v0, 1. - u0 - v0))
    slopes = np.column_stack((du, dv, -du - dv))
    lower = np.full(len(indices), -np.inf)
    upper = np.full(len(indices), np.inf)
    allowed = np.ones(len(indices), bool)
    for axis in range(3):
        a, b = values[:, axis], slopes[:, axis]
        positive, negative = b > 0, b < 0
        lower[positive] = np.maximum(lower[positive], -a[positive] / b[positive])
        upper[negative] = np.minimum(upper[negative], -a[negative] / b[negative])
        allowed &= (b != 0) | (a >= 0)
    allowed &= lower <= upper
    ids = np.flatnonzero(allowed)
    intervals = np.column_stack((lower[ids], upper[ids]))
    if not np.isfinite(intervals).all():
        raise UnresolvedSection('Unbounded section of a finite triangle')
    return dict(intervals=intervals, triangle_ordinals=ids,
                total_triangles=len(indices), no_length_threshold=True)


def union_intervals(intervals):
    """Union intervals within one declared support set; never bridge a gap."""
    rows = np.asarray(intervals, np.float64)
    if rows.ndim != 2 or rows.shape[1] != 2 or not np.isfinite(rows).all():
        raise ValueError('Finite interval rows required')
    if (rows[:, 0] > rows[:, 1]).any():
        raise ValueError('Interval bounds must be ordered')
    if not len(rows):
        return np.empty((0, 2), np.float64)
    order = np.lexsort((rows[:, 1], rows[:, 0]))
    result = []
    lower, upper = rows[order[0]]
    for left, right in rows[order[1:]]:
        if left <= upper:
            upper = max(upper, right)
        else:
            result.append((lower, upper))
            lower, upper = left, right
    result.append((lower, upper))
    return np.asarray(result, np.float64)


def intersect_interval_unions(left, right):
    """Require both finite support sets, retaining even isolated point contact."""
    a, b = union_intervals(left), union_intervals(right)
    i = j = 0
    result = []
    while i < len(a) and j < len(b):
        lower = max(a[i, 0], b[j, 0])
        upper = min(a[i, 1], b[j, 1])
        if lower <= upper:
            result.append((lower, upper))
        if a[i, 1] < b[j, 1]:
            i += 1
        elif b[j, 1] < a[i, 1]:
            j += 1
        else:
            i += 1
            j += 1
    return union_intervals(np.asarray(result, np.float64).reshape(-1, 2))


def projected_face_sections(line, uv_mm, triangles, basis_uv, origin_mm):
    """Section by orthogonal projection onto a supplied finite face.

    Projection alone does not certify coplanarity/contact. A caller must retain
    the plane distances and all supplying face IDs, and declare its hypothesis.
    """
    basis = np.asarray(basis_uv, np.float64)
    origin = _vector(origin_mm, 3)
    if (basis.shape != (2, 3) or not np.isfinite(basis).all()
            or not np.allclose(basis @ basis.T, np.eye(2), rtol=0, atol=1e-9)):
        raise ValueError('Orthonormal finite face UV basis required')
    sections = triangle_line_sections(uv_mm, triangles,
        (line.origin_mm - origin) @ basis.T, line.direction @ basis.T)
    sections['union_intervals_mm'] = union_intervals(sections['intervals'])
    sections['certified_filled_material'] = False
    sections['weld_or_contact_verified'] = False
    return sections
