"""Closed support of every frozen original UV triangle, including rank 0/1.

The domain is a computational PCD observation hypothesis. Float UV projection
collapse does not imply that the original physical material is a line. Keep
the original rows, exact float geometry and separately bounded projection error.
"""
from fractions import Fraction as F
import math

import numpy as np

from .concrete_curve_certificate_250 import ConcreteCurve, UnprovenCurve, POLICY, _up
from .finite_curve_domain_251 import TriangleDomain as PositiveTriangleDomain

VERSION = 'ALL_ORIGINAL_FLOAT_UV_TRIANGLE_POINT_SEGMENT_AREA_SUPPORT_253_v1'


def _f(value):
    return F.from_float(float(value))


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _minus(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _record(value):
    return dict(numerator=str(value.numerator), denominator=str(value.denominator))


def _point_on_segment(value, start, end):
    direction = _minus(end, start)
    offset = _minus(value, start)
    if direction == (0, 0):
        return value == start
    if _cross(direction, offset) != 0:
        return False
    axis = 0 if direction[0] != 0 else 1
    parameter = offset[axis] / direction[axis]
    return 0 <= parameter <= 1


def _segment_parameter_intersection(a, b, p, q):
    """Exact closed parameter interval for one positive query segment."""
    direction, support = _minus(b, a), _minus(q, p)
    offset = _minus(p, a)
    determinant = _cross(direction, support)
    if determinant != 0:
        t = _cross(offset, support) / determinant
        u = _cross(offset, direction) / determinant
        return (t, t) if 0 <= t <= 1 and 0 <= u <= 1 else None
    if _cross(direction, offset) != 0:
        return None
    axis = 0 if direction[0] != 0 else 1
    parameters = sorted(((p[axis] - a[axis]) / direction[axis],
                         (q[axis] - a[axis]) / direction[axis]))
    lower, upper = max(F(0), parameters[0]), min(F(1), parameters[1])
    return (lower, upper) if lower <= upper else None


class TriangleDomain(PositiveTriangleDomain):
    """All original closed triangles; an exactly collinear triangle is a segment."""

    def __init__(self, uv_mm, triangles, domain_id):
        self.uv = np.asarray(uv_mm, float).copy()
        self.triangles = np.asarray(triangles).copy()
        if (not isinstance(domain_id, str) or not domain_id
                or self.uv.ndim != 2 or self.uv.shape[1] != 2
                or not np.isfinite(self.uv).all()
                or self.triangles.ndim != 2 or self.triangles.shape[1] != 3
                or not np.issubdtype(self.triangles.dtype, np.integer)
                or (self.triangles < 0).any() or (self.triangles >= len(self.uv)).any()):
            raise ValueError('Original finite UV rows, integer triangles and domain identity required')
        self.domain_id = domain_id
        self.points = [tuple(map(_f, point)) for point in self.uv]
        self.orientation, self.support_rank, self.closed_support_endpoints = [], [], []
        area_edges, lower_dimensional_edges = {}, {}
        for ordinal, indices in enumerate(self.triangles):
            vertices = [self.points[int(index)] for index in indices]
            a, b, c = vertices
            area = _cross(_minus(b, a), _minus(c, a))
            self.orientation.append(1 if area > 0 else (-1 if area < 0 else 0))
            if area == 0:
                lower, upper = min(vertices), max(vertices)
                self.support_rank.append(0 if lower == upper else 1)
                self.closed_support_endpoints.append((lower, upper))
                for p, q in zip(indices, np.roll(indices, -1)):
                    edge = tuple(sorted((int(p), int(q))))
                    lower_dimensional_edges.setdefault(edge, []).append(ordinal)
            else:
                self.support_rank.append(2)
                self.closed_support_endpoints.append(None)
                for p, q, third in ((indices[0], indices[1], indices[2]),
                                    (indices[1], indices[2], indices[0]),
                                    (indices[2], indices[0], indices[1])):
                    edge = tuple(sorted((int(p), int(q))))
                    side = _cross(_minus(self.points[edge[1]], self.points[edge[0]]),
                                  _minus(self.points[int(third)], self.points[edge[0]]))
                    area_edges.setdefault(edge, []).append((ordinal, 1 if side > 0 else -1))
        self.boundary_superset, self.removed_two_sided_internal_edges = [], []
        for edge in sorted(set(area_edges) | set(lower_dimensional_edges)):
            records = area_edges.get(edge, [])
            two_sided = {side for _, side in records} == {-1, 1}
            if two_sided and edge not in lower_dimensional_edges:
                self.removed_two_sided_internal_edges.append(dict(
                    edge=list(edge), original_triangle_ordinals=[i for i, _ in records]))
            else:
                # Point edges and all rank-1 original edges remain in the superset.
                self.boundary_superset.append(edge)
        if len(self.triangles) and not self.boundary_superset:
            raise UnprovenCurve('No complete finite boundary superset was established')
        vertices = self.uv[self.triangles]
        self.box_lo = vertices.min(axis=1) if len(vertices) else np.empty((0, 2))
        self.box_hi = vertices.max(axis=1) if len(vertices) else np.empty((0, 2))
        self.uv.flags.writeable = self.triangles.flags.writeable = False

    def original_triangle_member(self, ordinal, value):
        if len(value) != 2:
            return False
        if self.support_rank[ordinal] < 2:
            return _point_on_segment(value, *self.closed_support_endpoints[ordinal])
        vertices = [self.points[int(index)] for index in self.triangles[ordinal]]
        sign = self.orientation[ordinal]
        return all(sign * _cross(_minus(b, a), _minus(value, a)) >= 0
                   for a, b in zip(vertices, vertices[1:] + vertices[:1]))

    def point_member(self, point):
        numeric = np.asarray(point, float)
        if numeric.shape != (2,) or not np.isfinite(numeric).all():
            raise ValueError('Finite original UV point required')
        value = tuple(map(_f, numeric))
        candidates = np.flatnonzero(((self.box_lo <= numeric) & (numeric <= self.box_hi)).all(axis=1))
        for ordinal in candidates:
            if self.original_triangle_member(int(ordinal), value):
                return True, int(ordinal)
        return False, None

    def exact_span(self, start, end):
        start, end = np.asarray(start, float), np.asarray(end, float)
        if (start.shape != (2,) or end.shape != (2,)
                or not np.isfinite(start).all() or not np.isfinite(end).all()):
            raise ValueError('Finite two-dimensional span endpoints required')
        a, b = tuple(map(_f, start)), tuple(map(_f, end))
        if a == b:
            inside, ordinal = self.point_member(start)
            return dict(covered=inside, zero_original_span_retained=True,
                        intersections=[] if ordinal is None else [dict(
                            original_triangle_ordinal=ordinal, support_rank=self.support_rank[ordinal],
                            lo=_record(F(0)), hi=_record(F(1)))],
                        union_intervals=[] if not inside else [[_record(F(0)), _record(F(1))]],
                        positive_gap_or_hole_merged=False)
        delta = _minus(b, a)
        segment_lo, segment_hi = np.minimum(start, end), np.maximum(start, end)
        candidates = np.flatnonzero(((self.box_lo <= segment_hi) & (segment_lo <= self.box_hi)).all(axis=1))
        intervals = []
        for ordinal in candidates:
            ordinal = int(ordinal)
            if self.support_rank[ordinal] < 2:
                interval = _segment_parameter_intersection(a, b, *self.closed_support_endpoints[ordinal])
                if interval is not None:
                    intervals.append((*interval, ordinal))
                continue
            vertices = [self.points[int(index)] for index in self.triangles[ordinal]]
            sign, lower, upper, allowed = self.orientation[ordinal], F(0), F(1), True
            for p, q in zip(vertices, vertices[1:] + vertices[:1]):
                edge = _minus(q, p)
                value, slope = sign * _cross(edge, _minus(a, p)), sign * _cross(edge, delta)
                if slope > 0:
                    lower = max(lower, -value / slope)
                elif slope < 0:
                    upper = min(upper, -value / slope)
                elif value < 0:
                    allowed = False
                    break
            if allowed and lower <= upper:
                intervals.append((lower, upper, ordinal))
        union = []
        for lower, upper, _ in sorted(intervals):
            if union and lower <= union[-1][1]:
                union[-1][1] = max(union[-1][1], upper)
            else:
                union.append([lower, upper])
        covered = len(union) == 1 and union[0] == [F(0), F(1)]
        return dict(covered=covered, zero_original_span_retained=False,
                    intersections=[dict(original_triangle_ordinal=i, support_rank=self.support_rank[i],
                                        lo=_record(lo), hi=_record(hi)) for lo, hi, i in intervals],
                    union_intervals=[[_record(lo), _record(hi)] for lo, hi in union],
                    positive_gap_or_hole_merged=False)


def certify_curve_domain(curve, uv_mm, triangles, *, domain_id, policy=None, compiled_domain=None):
    if not isinstance(curve, ConcreteCurve):
        curve = ConcreteCurve(curve)
    if curve.record['dimensions'] != 2:
        raise ValueError('Explicit UV curve required; world binding is separate')
    policy = {**POLICY, **(policy or {})}
    if (not math.isfinite(policy['maximum_step_mm']) or not 0 < policy['maximum_step_mm'] <= 1.
            or not math.isfinite(policy['maximum_chord_mm']) or not 0 < policy['maximum_chord_mm'] <= .1
            or type(policy['max_refinements']) is not int or policy['max_refinements'] < 0):
        raise ValueError('Frozen contract sampling and nonnegative refinement budget required')
    result = dict(version=VERSION, scope='ONE_DECLARED_FROZEN_FLOAT_UV_ORIGINAL_TRIANGLE_DOMAIN', domain_id=domain_id,
                  global_surface_union_substituted_for_allowed_domain=False,
                  world_curve_to_original_domain_binding_verified=False,
                  actual_runtime_slot_permission_and_capacity_verified=False,
                  global_parent_order_and_identity_verified=False, certified_filled_material=False,
                  material_or_weld_identity_verified=False, convex_hull_or_hole_fill_used=False,
                  model_inference_calls=0, actual_weld_optimizer_updates=0,
                  actual_all111_or_all1605_joint_expression_run=False, full_contract_pass=False)
    try:
        domain = compiled_domain if compiled_domain is not None else TriangleDomain(uv_mm, triangles, domain_id)
        if (not isinstance(domain, TriangleDomain) or domain.domain_id != domain_id
                or not np.array_equal(domain.uv, uv_mm) or not np.array_equal(domain.triangles, triangles)):
            raise ValueError('Compiled original domain does not match its declared source rows')
    except UnprovenCurve as error:
        return {**result, 'status': 'UNPROVEN_ORIGINAL_TRIANGLE_DOMAIN', 'reason': str(error)}
    result.update(original_triangle_count=len(domain.triangles), original_uv_row_count=len(domain.uv),
                  original_support_rank_counts={str(rank): domain.support_rank.count(rank) for rank in (0, 1, 2)},
                  original_triangle_ordinals_deleted=0, boundary_superset_edges=len(domain.boundary_superset),
                  two_sided_internal_edges_removed=len(domain.removed_two_sided_internal_edges))
    if not len(domain.triangles):
        return {**result, 'status': 'PROVEN_CONCRETE_CURVE_OUTSIDE_EMPTY_DECLARED_DOMAIN'}
    uncertainty = curve.record['geometry_uncertainty_mm']
    if curve.kind != 'arc' and uncertainty == 0:
        vertices = curve.vertices()
        spans = [domain.exact_span(a, b) for a, b in zip(vertices[:-1], vertices[1:])]
        return {**result, 'status': 'PASS_COMPLETE_EXACT_UV_CURVE_DOMAIN_COVERAGE' if all(x['covered'] for x in spans)
                else 'PROVEN_CONCRETE_CURVE_OUTSIDE_DECLARED_TRIANGLE_DOMAIN',
                'original_geometry_span_count': len(spans),
                'zero_original_spans_retained': sum(x['zero_original_span_retained'] for x in spans),
                'per_original_span': spans,
                'proof': 'EXACT_CLOSED_ORIGINAL_POINT_SEGMENT_TRIANGLE_INTERSECTION_AND_GAP_PRESERVING_UNION'}
    rounds = []
    for refinement in range(policy['max_refinements'] + 1):
        try:
            samples = curve.samples(policy['maximum_step_mm'] / 2**refinement,
                                    policy['maximum_chord_mm'] / 4**refinement, policy['point_budget_per_curve'])
            points = samples['points']
            errors = _up(samples['point_error_upper_mm'] + uncertainty)
            total_chord = _up(samples['analytic_chord_error_upper_mm'] + errors.max())
            if total_chord > .1:
                raise UnprovenCurve('Actual chord plus numeric enclosure not certified at 0.1mm')
            members = [domain.point_member(point) for point in points]
            lower, upper = domain.boundary_distance_bounds(points)
            radius = _up(errors + _up(samples['maximum_step_upper_mm'] / 2))
            membership = np.array([row[0] for row in members])
            proven_outside, proven_inside = (~membership) & (lower > errors), membership & (lower > radius)
            rounds.append(dict(refinement=refinement, point_count=len(points),
                               sample_members_inside=int(membership.sum()), neighborhoods_proven_inside=int(proven_inside.sum()),
                               maximum_step_upper_mm=samples['maximum_step_upper_mm'],
                               total_interpolation_chord_error_upper_mm=float(total_chord),
                               minimum_boundary_distance_lower_mm=float(lower.min()),
                               boundary_edges_form_superset_of_true_union_boundary=True))
            if proven_outside.any():
                index = int(np.flatnonzero(proven_outside)[0])
                return {**result, 'status': 'PROVEN_CONCRETE_CURVE_POINT_OUTSIDE_DECLARED_DOMAIN', 'rounds': rounds,
                        'outside_witness': dict(sample_index=index, point_mm=points[index].tolist(),
                                               point_error_upper_mm=float(errors[index]),
                                               boundary_distance_lower_mm=float(lower[index]),
                                               exact_sample_center_not_in_any_original_triangle=True)}
            if proven_inside.all():
                return {**result, 'status': 'PASS_COMPLETE_CONSERVATIVE_CURVE_UV_DOMAIN_COVERAGE', 'rounds': rounds,
                        'proof': 'EXACT_MEMBERSHIP_BOUNDARY_SUPERSET_DISTANCE_AND_CONTINUOUS_NEIGHBORHOODS'}
        except UnprovenCurve as error:
            return {**result, 'status': 'UNPROVEN_CURVE_DOMAIN_RESOURCE_OR_NUMERIC_LIMIT', 'reason': str(error), 'rounds': rounds}
    return {**result, 'status': 'UNPROVEN_NEAR_DECLARED_DOMAIN_BOUNDARY', 'rounds': rounds,
            'unproven_is_not_deleted_or_scored_as_success': True}
