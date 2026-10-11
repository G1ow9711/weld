"""Offline finite-line child expressivity, never a runtime truth-fed proposal.

Exact rational checks use the supplied IEEE doubles as the declared geometry.
One child's witnesses do not certify instance capacity, alias ownership, parent
order, weld semantics, or learnability. No positive length or gap is discarded.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
import math

import numpy as np

VERSION = 'OFFLINE_FINITE_LINE_ORDERED_CHILD_EXPRESSION_246_v1'
LIMIT_MM = 3.0


def _vec(value):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError('Three finite physical XYZ coordinates required')
    return array


def _fraction_vector(value):
    return tuple(Fraction.from_float(float(x)) for x in value)


def _dot(a, b):
    return sum((x * y for x, y in zip(a, b)), Fraction(0))


def _decimal(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def _square_distance(left, right):
    delta = tuple(x - y for x, y in zip(left, right))
    return _dot(delta, delta)


def _fraction_record(value):
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}


def _ordered_components(intervals):
    rows = np.asarray(intervals, dtype=np.float64)
    if rows.size == 0:
        rows = rows.reshape(0, 2)
    if (rows.ndim != 2 or rows.shape[1] != 2 or not np.isfinite(rows).all()
            or (rows[:, 0] > rows[:, 1]).any()):
        raise ValueError('Finite ordered closed parameter intervals required')
    output = []
    for ordinal in sorted(range(len(rows)), key=lambda i: tuple(rows[i])):
        lo, hi = map(float, rows[ordinal])
        if output and lo <= output[-1]['hi']:
            output[-1]['hi'] = max(output[-1]['hi'], hi)
            output[-1]['original_interval_ordinals'].append(ordinal)
        else:
            output.append(dict(lo=lo, hi=hi, original_interval_ordinals=[ordinal]))
    return rows, output


def ball_parameter_interval(origin_mm, direction, point_mm):
    """Closed 3mm-ball section; unresolved tangencies remain explicit.

    Direction must be a finite nonzero numerical unit vector. The quadratic
    retains its actual squared norm rather than silently assuming it is one.
    Floating interval endpoints are rounded inward and checked exactly.
    """
    origin, vector, point = map(_vec, (origin_mm, direction, point_mm))
    norm2 = float(vector @ vector)
    if not math.isfinite(norm2) or abs(norm2 - 1.0) > 1e-12:
        raise ValueError('Numerical unit line direction required')
    o, d, p = map(_fraction_vector, (origin, vector, point))
    relative = tuple(x - y for x, y in zip(p, o))
    a, b = _dot(d, d), _dot(relative, d)
    c = _dot(relative, relative) - Fraction(9)
    discriminant = b * b - a * c
    result = {'limit_mm': LIMIT_MM, 'quadratic_a': _fraction_record(a),
              'quadratic_b': _fraction_record(b), 'quadratic_c': _fraction_record(c),
              'discriminant_sign': (discriminant > 0) - (discriminant < 0)}
    if discriminant < 0:
        return {**result, 'status': 'BALL_MISSES_INFINITE_LINE', 'interval_mm': None}

    def inside(value):
        t = Fraction.from_float(value)
        return a * t * t - 2 * b * t + c <= 0

    if discriminant == 0:
        exact = b / a
        try:
            value = float(exact)
        except OverflowError:
            value = math.inf
        if not math.isfinite(value) or Fraction.from_float(value) != exact:
            return {**result, 'status': 'UNRESOLVED_FLOAT_TANGENCY',
                    'exact_tangent_parameter': _fraction_record(exact), 'interval_mm': None}
        return {**result, 'status': 'CLOSED_POINT_BALL_SECTION', 'interval_mm': [value, value]}

    with localcontext() as context:
        context.prec = 160
        da, db, dc = map(_decimal, (a, b, c))
        root = _decimal(discriminant).sqrt()
        stable = db + root if b >= 0 else db - root
        endpoints = sorted((stable / da, dc / stable))
        lower, upper = map(float, endpoints)
        if not all(map(math.isfinite, (lower, upper))):
            return {**result, 'status': 'UNRESOLVED_FLOAT_RANGE', 'interval_mm': None}
        if Decimal.from_float(lower) < endpoints[0]:
            lower = math.nextafter(lower, math.inf)
        if Decimal.from_float(upper) > endpoints[1]:
            upper = math.nextafter(upper, -math.inf)
    for _ in range(8):
        if lower <= upper and inside(lower) and inside(upper):
            return {**result, 'status': 'FINITE_CLOSED_BALL_SECTION', 'interval_mm': [lower, upper]}
        if lower > upper:
            break
        if not inside(lower):
            lower = math.nextafter(lower, math.inf)
        if not inside(upper):
            upper = math.nextafter(upper, -math.inf)
    return {**result, 'status': 'UNRESOLVED_FLOAT_BALL_SECTION', 'interval_mm': None}


def _interior_pair(start, end, orientation):
    with localcontext() as context:
        context.prec = 160
        start_center = sum(map(Decimal.from_float, start)) / 2
        end_center = sum(map(Decimal.from_float, end)) / 2
        center_gap = orientation * (end_center - start_center)
        extreme_gap = (Decimal.from_float(end[1]) - Decimal.from_float(start[0])
                       if orientation > 0 else
                       Decimal.from_float(start[1]) - Decimal.from_float(end[0]))
        if center_gap >= extreme_gap / 2:
            a, b = float(start_center), float(end_center)
            if orientation * (b - a) > 0:
                return a, b
        if orientation > 0:
            left, right = map(Decimal.from_float, (start[0], end[1]))
            gap = right - left
            weight = gap / (2 * (gap - center_gap))
            a = min(start[1], max(start[0], float((1 - weight) * left + weight * start_center)))
            b = max(end[0], min(end[1], float((1 - weight) * right + weight * end_center)))
            return (a, b) if a < b else (start[0], end[1])
        left, right = map(Decimal.from_float, (start[1], end[0]))
        gap = left - right
        weight = gap / (2 * (gap - center_gap))
        a = max(start[0], min(start[1], float((1 - weight) * left + weight * start_center)))
        b = min(end[1], max(end[0], float((1 - weight) * right + weight * end_center)))
        return (a, b) if a > b else (start[1], end[0])


def finite_ordered_child_subspans(origin_mm, direction, finite_intervals_mm,
                                 reference_start_mm, reference_end_mm, *, construction_id):
    """Return all feasible components/orientations for ONE offline reference.

    Each witness stays inside one connected finite component. Corresponding
    endpoints bound both entire straight segments by convex interpolation.
    This routine neither generates a runtime proposal ID per GT nor decides
    how many parent/child instances a future trained decoder can emit.
    """
    origin, vector, r0, r1 = map(_vec, (origin_mm, direction, reference_start_mm, reference_end_mm))
    if np.array_equal(r0, r1):
        raise ValueError('A positive reference line child is required')
    rows, components = _ordered_components(finite_intervals_mm)
    balls = [ball_parameter_interval(origin, vector, point) for point in (r0, r1)]
    result = dict(version=VERSION, construction_id=construction_id,
                  scope='OFFLINE_SINGLE_ORDERED_LINE_CHILD_GEOMETRY_ONLY',
                  original_finite_intervals_mm=rows.tolist(), connected_components=components,
                  endpoint_ball_sections=balls, witnesses=[], unresolved_witnesses=[],
                  positive_length_cutoff_applied=False, positive_gap_merged=False,
                  reference_used_for_runtime_candidate_generation=False,
                  simultaneous_parent_child_or_alias_assignment_verified=False,
                  material_or_weld_identity_verified=False, model_inference_calls=0)
    if any(b['status'] == 'BALL_MISSES_INFINITE_LINE' for b in balls):
        return {**result, 'status': 'INFEASIBLE_REFERENCE_ENDPOINT_BALL_MISSES_LINE'}
    if any(b['interval_mm'] is None for b in balls):
        return {**result, 'status': 'UNRESOLVED_ENDPOINT_BALL_SECTION'}
    o, d, references = _fraction_vector(origin), _fraction_vector(vector), list(map(_fraction_vector, (r0, r1)))
    for ordinal, component in enumerate(components):
        lo, hi = component['lo'], component['hi']
        ranges = [[max(lo, b['interval_mm'][0]), min(hi, b['interval_mm'][1])] for b in balls]
        if any(a > b for a, b in ranges) or lo == hi:
            continue
        for orientation, possible in ((1, ranges[0][0] < ranges[1][1]),
                                      (-1, ranges[0][1] > ranges[1][0])):
            if not possible:
                continue
            start_t, end_t = _interior_pair(*ranges, orientation)
            assert lo <= min(start_t, end_t) < max(start_t, end_t) <= hi
            points = [tuple(x + y * Fraction.from_float(t) for x, y in zip(o, d))
                      for t in (start_t, end_t)]
            squared = [_square_distance(point, ref) for point, ref in zip(points, references)]
            assert max(squared) <= 9
            exported = np.array([[float(v) for v in point] for point in points])
            exported_squared = [_square_distance(_fraction_vector(point), ref)
                                for point, ref in zip(exported, references)]
            witness = dict(connected_component_ordinal=ordinal,
                           original_interval_ordinals=component['original_interval_ordinals'],
                           start_t_mm=start_t, end_t_mm=end_t, orientation=orientation,
                           endpoint_parameter_ranges_mm=ranges,
                           start_mm=exported[0].tolist(), end_mm=exported[1].tolist(),
                           exact_analytic_endpoint_squared_errors=list(map(_fraction_record, squared)),
                           exact_exported_endpoint_squared_errors=list(map(_fraction_record, exported_squared)))
            if np.array_equal(*exported) or max(exported_squared) > 9:
                witness['status'] = 'UNRESOLVED_FLOAT_EXPORT_GEOMETRY'
                result['unresolved_witnesses'].append(witness)
                continue
            witness.update(status='FEASIBLE_POSITIVE_ORDERED_LINE_SUBSPAN',
                           endpoint_and_bidirectional_continuous_error_limit_mm=LIMIT_MM,
                           proof='CORRESPONDING_ENDPOINT_BALLS_AND_CONVEX_INTERPOLATION',
                           finite_domain_parameter_membership_verified=True)
            result['witnesses'].append(witness)
    status = ('FEASIBLE_POSITIVE_ORDERED_LINE_SUBSPANS' if result['witnesses'] else
              'UNRESOLVED_FLOAT_EXPORT_GEOMETRY' if result['unresolved_witnesses'] else
              'NO_POSITIVE_ORDERED_SUBSPAN_IN_FINITE_DOMAIN')
    return {**result, 'status': status}
