"""Independent C1/D4 circular-arc representation and sampling in millimetres.

This module does not select welds or infer material-side normals. The plane
normal describes only the circle's oriented geometry. Original model, cache,
and decoder files remain unchanged.
"""
from dataclasses import dataclass
from collections.abc import Mapping
import math

import numpy as np

VERSION = 'contract_3d_circular_arc_chord_v1'
TAU = 2. * math.pi


class CurveGeometryError(ValueError):
    """Invalid or numerically unrepresentable geometry; never silently dropped."""


def _vector(value, name):
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise CurveGeometryError(f'{name}: finite three-vector required')
    return vector


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0.:
        raise CurveGeometryError(f'{name}: positive finite value required')
    return value


@dataclass(frozen=True)
class CircularArc:
    """One ordered 3D circular arc, with no repeated winding.

    Position is nonnegative physical arc length. Positive sweep follows the
    right-hand rule around plane_normal; start_direction fixes the start point.
    A full revolution requires an explicit is_closed=True marker.
    """
    center_mm: object
    radius_mm: float
    plane_normal: object
    start_direction: object
    sweep_rad: float
    is_closed: bool = False

    def __post_init__(self):
        center = _vector(self.center_mm, 'center_mm')
        radius = _positive(self.radius_mm, 'radius_mm')
        normal = _vector(self.plane_normal, 'plane_normal')
        start = _vector(self.start_direction, 'start_direction')
        normal_norm, start_norm = float(np.linalg.norm(normal)), float(np.linalg.norm(start))
        if (not math.isfinite(normal_norm) or not math.isfinite(start_norm)
                or normal_norm == 0. or start_norm == 0.):
            raise CurveGeometryError('DEGENERATE_BASIS: nonzero normal and start direction required')
        normal = normal / normal_norm
        start = start / start_norm
        if abs(float(normal @ start)) > 1e-10:
            raise CurveGeometryError('NONPLANAR_START_DIRECTION: start direction must lie in circle plane')
        # Remove only floating roundoff from an already perpendicular basis.
        start = start - normal * float(normal @ start)
        start = start / np.linalg.norm(start)
        sweep = float(self.sweep_rad)
        if not math.isfinite(sweep) or sweep == 0.:
            raise CurveGeometryError('DEGENERATE_SWEEP: finite nonzero sweep required')
        if abs(sweep) > TAU:
            raise CurveGeometryError('REPEATED_WINDING: absolute sweep must not exceed 2*pi')
        if not isinstance(self.is_closed, (bool, np.bool_)):
            raise CurveGeometryError('CLOSURE_FLAG: explicit boolean required')
        if self.is_closed:
            if not math.isclose(abs(sweep), TAU, rel_tol=0., abs_tol=1e-12):
                raise CurveGeometryError('INCOMPLETE_CLOSED_ARC: closed circle requires one full revolution')
            sweep = math.copysign(TAU, sweep)
        elif abs(sweep) == TAU:
            raise CurveGeometryError('MISSING_CLOSURE_FLAG: full circle requires is_closed=True')
        if not math.isfinite(radius * abs(sweep)) or radius * abs(sweep) == 0.:
            raise CurveGeometryError('UNREPRESENTABLE_LENGTH: finite positive arc length required')
        for name, value in [('center_mm', center), ('radius_mm', radius),
                            ('plane_normal', normal), ('start_direction', start),
                            ('sweep_rad', sweep), ('is_closed', bool(self.is_closed))]:
            object.__setattr__(self, name, value)

    @property
    def length_mm(self):
        return self.radius_mm * abs(self.sweep_rad)

    def point_at(self, position_mm):
        position = np.asarray(position_mm, dtype=np.float64)
        if (not np.isfinite(position).all() or np.any(position < 0.)
                or np.any(position > self.length_mm)):
            raise CurveGeometryError('POSITION_OUTSIDE_ARC: positions must be within [0, length_mm]')
        angle = math.copysign(1., self.sweep_rad) * position / self.radius_mm
        second = np.cross(self.plane_normal, self.start_direction)
        result = self.center_mm + self.radius_mm * (
            np.cos(angle)[..., None] * self.start_direction
            + np.sin(angle)[..., None] * second)
        if self.is_closed:
            result = np.where((position == self.length_mm)[..., None],
                              self.center_mm + self.radius_mm * self.start_direction, result)
        return result

    def to_parameters(self):
        return {'schema_version': VERSION, 'center_mm': self.center_mm.tolist(),
                'radius_mm': self.radius_mm, 'plane_normal': self.plane_normal.tolist(),
                'start_direction': self.start_direction.tolist(), 'sweep_rad': self.sweep_rad,
                'is_closed': self.is_closed, 'units': 'mm', 'angle_units': 'rad',
                'parameterization': 'C + R*(cos(sign(sweep)*s/R)*U + sin(sign(sweep)*s/R)*(N cross U))'}

    @classmethod
    def from_parameters(cls, parameters):
        if not isinstance(parameters, Mapping):
            raise CurveGeometryError('ARC_PARAMETERS: mapping required')
        if parameters.get('units', 'mm') != 'mm' or parameters.get('angle_units', 'rad') != 'rad':
            raise CurveGeometryError('PARAMETER_UNITS: millimetres and radians required')
        try:
            return cls(**{key: parameters[key] for key in
                          ['center_mm', 'radius_mm', 'plane_normal', 'start_direction', 'sweep_rad', 'is_closed']})
        except KeyError as exc:
            raise CurveGeometryError(f'MISSING_ARC_PARAMETER: {exc.args[0]}') from exc

    def subarc(self, start_mm, end_mm):
        start_mm, end_mm = float(start_mm), float(end_mm)
        if not (0. <= start_mm < end_mm <= self.length_mm):
            raise CurveGeometryError('INVALID_SUBARC: require 0 <= start < end <= arc length')
        direction = (self.point_at(start_mm) - self.center_mm) / self.radius_mm
        sweep = math.copysign((end_mm - start_mm) / self.radius_mm, self.sweep_rad)
        return CircularArc(self.center_mm, self.radius_mm, self.plane_normal, direction,
                           sweep, self.is_closed and start_mm == 0. and end_mm == self.length_mm)


def from_parameters(parameters):
    return CircularArc.from_parameters(parameters)


def stable_sagitta(radius_mm, angle_step_rad):
    radius = _positive(radius_mm, 'radius_mm')
    angle = np.asarray(angle_step_rad, dtype=np.float64)
    if not np.isfinite(angle).all() or np.any(np.abs(angle) > math.pi / 2. + 1e-14):
        raise CurveGeometryError('ANGLE_STEP: finite angle step <= pi/2 required')
    return 2. * radius * np.sin(angle / 4.) ** 2


def sample_arc(arc, max_step_mm=1., max_chord_error_mm=.1, breakpoints_mm=(), max_points=1_000_001):
    """Sample at physical arc length, retaining every exact supplied breakpoint.

    Sagitta uses 2*R*sin(delta_angle/4)**2, avoiding cancellation for large R
    or short arcs. The pi/2 cap keeps tiny circles' ordered topology visible.
    Resource or floating-resolution limits raise errors; they never erase arcs.
    """
    if not isinstance(arc, CircularArc):
        raise CurveGeometryError('ARC_TYPE: CircularArc required')
    step = _positive(max_step_mm, 'max_step_mm')
    chord_error = _positive(max_chord_error_mm, 'max_chord_error_mm')
    if not isinstance(max_points, int) or max_points < 2:
        raise CurveGeometryError('MAX_POINTS: integer >=2 required')
    ratio = chord_error / arc.radius_mm / 2.
    angle_cap = math.pi / 2.
    if ratio < math.sin(math.pi / 8.) ** 2:
        angle_cap = 4. * math.asin(math.sqrt(ratio))
    physical_step = min(step, arc.radius_mm * angle_cap)
    if physical_step <= 0.:
        raise CurveGeometryError('UNREPRESENTABLE_SAMPLE_STEP: numeric underflow')
    cuts = np.asarray(breakpoints_mm, dtype=np.float64)
    if (cuts.ndim != 1 or not np.isfinite(cuts).all()
            or np.any(cuts < 0.) or np.any(cuts > arc.length_mm)):
        raise CurveGeometryError('INVALID_BREAKPOINT: finite positions within the arc required')
    divisions = np.unique(np.r_[0., cuts, arc.length_mm])
    intervals = []
    count = 1
    for left, right in zip(divisions[:-1], divisions[1:]):
        needed = (right - left) / physical_step
        if not math.isfinite(needed) or needed > max_points - count:
            raise CurveGeometryError('SAMPLE_LIMIT_EXCEEDED: arc needs more points; geometry was not dropped')
        n = max(1, int(math.ceil(needed)))
        count += n
        if count > max_points:
            raise CurveGeometryError('SAMPLE_LIMIT_EXCEEDED: arc needs more points; geometry was not dropped')
        intervals.append(np.linspace(left, right, n + 1, dtype=np.float64)[:-1])
    positions = np.r_[np.concatenate(intervals), arc.length_mm]
    if np.any(np.diff(positions) <= 0.):
        raise CurveGeometryError('UNREPRESENTABLE_SAMPLE_POSITIONS: positive intervals collapse in float64')
    points = arc.point_at(positions)
    if not np.isfinite(points).all():
        raise CurveGeometryError('UNREPRESENTABLE_POINTS: generated coordinates overflow float64')
    if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) == 0.):
        raise CurveGeometryError('UNREPRESENTABLE_POINTS: ordered points collapse at coordinate precision')
    audit = {'version': VERSION, 'max_arc_step_mm': float(np.diff(positions).max()),
             'max_angle_step_rad': float((np.diff(positions) / arc.radius_mm).max()),
             'max_chord_error_mm': float(stable_sagitta(arc.radius_mm, np.diff(positions) / arc.radius_mm).max()),
             'requested_max_step_mm': step, 'requested_max_chord_error_mm': chord_error,
             'point_count': len(points), 'breakpoints_mm': cuts.tolist(),
             'polyline_length_mm': float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
             'length_basis': 'delivered analytic circular arc: radius_mm * abs(sweep_rad)'}
    return {'kind': 'arc', 'points': points.tolist(), 'arc_parameters': arc.to_parameters(),
            'length_mm': arc.length_mm, 'is_closed': arc.is_closed,
            'sample_positions_mm': positions.tolist(), 'sampling_audit': audit}


def verify_sampled_arc(segment, tolerance_mm=1e-9):
    """Verify delivered points, topology, sampling limits and analytic length."""
    arc = CircularArc.from_parameters(segment['arc_parameters'])
    points = np.asarray(segment['points'], dtype=np.float64)
    positions = np.asarray(segment['sample_positions_mm'], dtype=np.float64)
    if (points.shape != (len(positions), 3) or len(points) < 2 or positions.ndim != 1
            or not np.isfinite(points).all() or positions[0] != 0.
            or positions[-1] != arc.length_mm or np.any(np.diff(positions) <= 0.)):
        raise CurveGeometryError('INVALID_ORDERED_SAMPLES: ordered endpoint-complete positions required')
    error = float(np.linalg.norm(points - arc.point_at(positions), axis=1).max())
    if error > tolerance_mm:
        raise CurveGeometryError(f'ANALYTIC_SAMPLE_MISMATCH: residual {error} mm')
    if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) == 0.):
        raise CurveGeometryError('DUPLICATE_SAMPLE: consecutive points may not coincide')
    if arc.is_closed and not np.array_equal(points[0], points[-1]):
        raise CurveGeometryError('CLOSED_ENDPOINT_MISMATCH: closed endpoint must equal start exactly')
    steps = np.diff(positions)
    sagittas = stable_sagitta(arc.radius_mm, steps / arc.radius_mm)
    if float(steps.max()) > 1. + 1e-12 or float(sagittas.max()) > .1 + 1e-12:
        raise CurveGeometryError('D4_SAMPLING_LIMIT: spacing <=1 mm and sagitta <=0.1 mm required')
    if bool(segment.get('is_closed', arc.is_closed)) != arc.is_closed:
        raise CurveGeometryError('CLOSURE_PARAMETER_MISMATCH')
    if 'length_mm' in segment and not math.isclose(float(segment['length_mm']), arc.length_mm,
                                                 rel_tol=1e-12, abs_tol=tolerance_mm):
        raise CurveGeometryError('DELIVERY_LENGTH_MISMATCH: length must equal analytic arc length')
    return {'point_count': len(points), 'analytic_length_mm': arc.length_mm,
            'max_point_residual_mm': error, 'max_arc_step_mm': float(steps.max()),
            'max_chord_error_mm': float(sagittas.max()),
            'polyline_length_mm': float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())}


def fit_ordered_circle(points, tolerance_mm=1e-6, is_closed=None):
    """Recover an analytic arc only from a strictly verified ordered circle.

    This is representation fitting, never weld discovery or a learned-accuracy
    measurement. Every source point must pass plane and radial residual tests;
    all angular increments must have one sign. A half-turn or larger source
    interval is ambiguous and rejected. Repeated winding is rejected.
    """
    tolerance = _positive(tolerance_mm, 'tolerance_mm')
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 3 or not np.isfinite(points).all():
        raise CurveGeometryError('CIRCLE_FIT_POINTS: at least three finite ordered 3D points required')
    closed = np.array_equal(points[0], points[-1]) if is_closed is None else is_closed
    if not isinstance(closed, (bool, np.bool_)):
        raise CurveGeometryError('CLOSURE_FLAG: boolean required')
    if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) == 0.):
        raise CurveGeometryError('DUPLICATE_SOURCE_POINT: consecutive source points coincide')
    if closed and np.linalg.norm(points[-1] - points[0]) > tolerance:
        raise CurveGeometryError('INCOMPLETE_CLOSED_SOURCE: endpoint residual exceeds tolerance')
    unique = points[:-1] if closed else points
    if len(unique) < 3:
        raise CurveGeometryError('DEGENERATE_CIRCLE_FIT: three distinct circle points required')
    origin = unique.mean(axis=0)
    shifted = unique - origin
    _, singular, basis = np.linalg.svd(shifted, full_matrices=False)
    if len(singular) < 2 or singular[1] <= np.finfo(float).eps * max(shifted.shape) * singular[0]:
        raise CurveGeometryError('DEGENERATE_CIRCLE_FIT: noncollinear points required')
    normal = basis[-1]
    if normal[np.argmax(np.abs(normal))] < 0.:
        normal = -normal
    x_axis = basis[0]
    y_axis = np.cross(normal, x_axis)
    xy = np.column_stack([shifted @ x_axis, shifted @ y_axis])
    matrix = np.column_stack([2. * xy, np.ones(len(xy))])
    solution, _, rank, _ = np.linalg.lstsq(matrix, np.sum(xy ** 2, axis=1), rcond=None)
    if rank != 3:
        raise CurveGeometryError('DEGENERATE_CIRCLE_FIT: circle system is rank deficient')
    radius_squared = solution[2] + solution[:2] @ solution[:2]
    if not math.isfinite(radius_squared) or radius_squared <= 0.:
        raise CurveGeometryError('DEGENERATE_CIRCLE_FIT: positive finite radius required')
    radius = math.sqrt(radius_squared)
    center = origin + solution[0] * x_axis + solution[1] * y_axis
    radial = points - center
    plane_error = np.abs(radial @ normal)
    in_plane = radial - (radial @ normal)[:, None] * normal
    radial_error = np.abs(np.linalg.norm(in_plane, axis=1) - radius)
    if float(np.hypot(plane_error, radial_error).max()) > tolerance:
        raise CurveGeometryError('NONCIRCULAR_SOURCE: strict 3D circle residual exceeds tolerance')
    start = in_plane[0] / np.linalg.norm(in_plane[0])
    tangent = np.cross(normal, start)
    angle = np.arctan2(in_plane @ tangent, in_plane @ start)
    increments = (np.diff(angle) + math.pi) % TAU - math.pi
    if np.any(np.abs(increments) >= math.pi - 1e-12) or np.any(increments == 0.):
        raise CurveGeometryError('AMBIGUOUS_SOURCE_TOPOLOGY: nonzero increments strictly below pi required')
    if not (np.all(increments > 0.) or np.all(increments < 0.)):
        raise CurveGeometryError('REVERSED_SOURCE_TOPOLOGY: source returns over an already traversed arc')
    sweep = float(math.fsum(increments.tolist()))
    # Permit only roundoff in recovering the explicitly declared full circle.
    if closed:
        if not math.isclose(abs(sweep), TAU, rel_tol=0., abs_tol=1e-10):
            raise CurveGeometryError('REPEATED_OR_INCOMPLETE_CLOSED_SOURCE: exactly one revolution required')
        sweep = math.copysign(TAU, sweep)
    elif abs(sweep) >= TAU:
        raise CurveGeometryError('REPEATED_SOURCE_WINDING: open source must cover less than one revolution')
    return CircularArc(center, radius, normal, start, sweep, bool(closed))
