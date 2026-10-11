"""CPU geometry/complete-incidence preparation; no recognition, NN or solver.

Geometry must come from the current point observation or an explicit synthetic
fixture. This module neither discovers primitives nor reads GT/STEP/cache files.
The frozen 195 topology is reproduced without its transitive torch import.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from types import MappingProxyType
import json
import math

import numpy as np

from weld_pipeline.contract_curve_geometry import CircularArc, CurveGeometryError, sample_arc, verify_sampled_arc

VERSION = 'typed_runtime_primitive_incidence_209_v1'
FG_VERSION = 'inclusive_sample_mean_std_actual_q_trapezoid_209_v1_uniform_195_exact'
ROLES = ('start', 'end', 'cut')
CHILD_ROLE_PAIRS = ((0, 1), (0, 2), (2, 2), (2, 1))
RESOLUTION_MM = 8. * 16. * np.finfo(np.float32).eps
PAIR_FEATURE_NAMES = ('query_delta_mm', 'centre_delta_mm', 'xyz_delta_x_mm', 'xyz_delta_y_mm',
    'xyz_delta_z_mm', 'xyz_distance_mm', 'abs_query_delta_mm', 'interval_FG_mean',
    'interval_FG_std', 'interval_FG_trapezoid_integral_mm', 'source_start', 'source_end',
    'source_cut', 'target_start', 'target_end', 'target_cut')


class PrimitiveError(ValueError):
    pass


def ordinal(value, name='ordinal'):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise PrimitiveError(name + ': nonnegative integer required')
    return int(value)


def array_sha(value):
    a = np.ascontiguousarray(value)
    return sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def canonical_json(value):
    return json.dumps(_json_value(value), sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def _json_value(value):
    if isinstance(value, Mapping):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


def _freeze_json(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze_json(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(v) for v in value)
    return value


def _seal(value):
    a = np.ascontiguousarray(value)
    # Immutable bytes backing also rejects setflags(write=True).
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _frozen_array(value, shape, name):
    a = np.array(value, np.float64, copy=True)
    if a.shape != shape or not np.isfinite(a).all():
        raise PrimitiveError(name + ': finite array of shape ' + str(shape) + ' required')
    return _seal(a)


def _positions(value, length, allow_extension):
    q = np.asarray(value, np.float64)
    if not np.isfinite(q).all() or (not allow_extension and (np.any(q < 0.) or np.any(q > length))):
        raise PrimitiveError('POSITION_OUTSIDE_PRIMITIVE: finite q in [0,length] required')
    return q


@dataclass(frozen=True)
class LinePrimitive:
    start_mm: object
    end_mm: object

    def __post_init__(self):
        a = _frozen_array(self.start_mm, (3,), 'start_mm')
        b = _frozen_array(self.end_mm, (3,), 'end_mm')
        length = float(np.linalg.norm(b - a))
        if not math.isfinite(length) or length <= 0.:
            raise PrimitiveError('DEGENERATE_LINE: finite positive length required')
        object.__setattr__(self, 'start_mm', a)
        object.__setattr__(self, 'end_mm', b)

    @property
    def length_mm(self):
        return float(np.linalg.norm(self.end_mm - self.start_mm))

    @property
    def is_closed(self):
        return False

    def point_at(self, q, allow_extension=False):
        q = _positions(q, self.length_mm, allow_extension)
        result = self.start_mm + q[..., None] * ((self.end_mm - self.start_mm) / self.length_mm)
        result = np.where((q == self.length_mm)[..., None], self.end_mm, result)
        return result

    def to_parameters(self):
        return dict(kind='line', start_mm=self.start_mm.tolist(), end_mm=self.end_mm.tolist(),
                    length_mm=self.length_mm, is_closed=False, units='mm', parameterization='physical_arc_length')

    def segment(self, a, b):
        _valid_child_span(self, a, b)
        count = max(1, int(math.ceil(abs(b - a))))
        if count > 1_000_000:
            raise PrimitiveError('EXPORT_SAMPLE_LIMIT_EXCEEDED: child retained in FP ledger')
        q = np.linspace(a, b, count + 1)
        points = self.point_at(q)
        chord_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        physical_steps = np.abs(np.diff(q))
        if (not np.isfinite(points).all() or not np.isfinite(chord_lengths).all()
                or np.any(chord_lengths <= 0.) or not np.allclose(chord_lengths, physical_steps, rtol=1e-9, atol=1e-9)):
            raise PrimitiveError('UNREPRESENTABLE_LINE_SAMPLES: XYZ steps differ from physical q; child retained')
        return dict(kind='line', points=points.tolist(), length_mm=abs(b - a), is_closed=False,
                    sample_positions_mm=np.abs(q - a).tolist(),
                    sampling_audit=dict(max_arc_step_mm=float(np.abs(np.diff(q)).max()), max_chord_error_mm=0.))


@dataclass(frozen=True)
class ArcPrimitive:
    arc: CircularArc

    def __post_init__(self):
        if not isinstance(self.arc, CircularArc):
            raise PrimitiveError('CircularArc required')
        # Copy the frozen 178 kernel's array payload and make it immutable here.
        value = CircularArc.from_parameters(self.arc.to_parameters())
        for name in ('center_mm', 'plane_normal', 'start_direction'):
            object.__setattr__(value, name, _seal(getattr(value, name)))
        object.__setattr__(self, 'arc', value)

    @property
    def length_mm(self):
        return self.arc.length_mm

    @property
    def is_closed(self):
        return self.arc.is_closed

    def point_at(self, q, allow_extension=False):
        q = _positions(q, self.length_mm, allow_extension)
        if not allow_extension:
            return self.arc.point_at(q)
        angle = math.copysign(1., self.arc.sweep_rad) * q / self.arc.radius_mm
        xyz = self.arc.center_mm + self.arc.radius_mm * (
            np.cos(angle)[..., None] * self.arc.start_direction + np.sin(angle)[..., None] *
            np.cross(self.arc.plane_normal, self.arc.start_direction))
        if self.is_closed:
            xyz = np.where((q == self.length_mm)[..., None], self.arc.point_at(0.), xyz)
        return xyz

    def to_parameters(self):
        return dict(kind='arc', arc_parameters=self.arc.to_parameters(), length_mm=self.length_mm,
                    is_closed=self.is_closed, units='mm', parameterization='physical_arc_length')

    def segment(self, a, b):
        _valid_child_span(self, a, b)
        closed = self.is_closed and abs(b - a) == self.length_mm
        sweep = math.copysign(1., self.arc.sweep_rad) * (b - a) / self.arc.radius_mm
        if closed:
            sweep = math.copysign(2. * math.pi, sweep)
        arc = CircularArc(self.arc.center_mm, self.arc.radius_mm, self.arc.plane_normal,
            (self.point_at(a) - self.arc.center_mm) / self.arc.radius_mm,
            sweep, closed)
        result = sample_arc(arc)
        result['points'][0] = self.point_at(a).tolist()
        result['points'][-1] = self.point_at(b).tolist()
        verify_sampled_arc(result)
        return result


@dataclass(frozen=True)
class PolylinePrimitive:
    knots_mm: object
    is_closed: bool = False

    def __post_init__(self):
        a = np.array(self.knots_mm, np.float64, copy=True)
        if a.ndim != 2 or a.shape[1:] != (3,) or len(a) < 2 or not np.isfinite(a).all():
            raise PrimitiveError('POLYLINE_KNOTS: at least two finite 3D knots required')
        if not isinstance(self.is_closed, (bool, np.bool_)):
            raise PrimitiveError('explicit boolean closure required')
        closed = bool(self.is_closed)
        if closed != np.array_equal(a[0], a[-1]):
            raise PrimitiveError('CLOSURE_FLAG: exact repeated endpoint iff declared closed required')
        if closed and len(a) < 4:
            raise PrimitiveError('CLOSED_POLYLINE: at least three traversed edges required')
        distances = np.linalg.norm(np.diff(a, axis=0), axis=1)
        cumulative = np.r_[0., np.cumsum(distances)]
        if not np.isfinite(cumulative).all() or np.any(distances <= 0.) or np.any(np.diff(cumulative) <= 0.):
            raise PrimitiveError('DEGENERATE_POLYLINE: zero or unrepresentable knot span')
        object.__setattr__(self, 'knots_mm', _seal(a))
        object.__setattr__(self, 'is_closed', closed)
        object.__setattr__(self, 'knot_positions_mm', _seal(cumulative))

    @property
    def length_mm(self):
        return float(self.knot_positions_mm[-1])

    def point_at(self, q, allow_extension=False):
        q = _positions(q, self.length_mm, allow_extension)
        flat = q.reshape(-1)
        i = np.clip(np.searchsorted(self.knot_positions_mm, flat, side='right') - 1,
                    0, len(self.knots_mm) - 2)
        factor = (flat - self.knot_positions_mm[i]) / np.diff(self.knot_positions_mm)[i]
        xyz = self.knots_mm[i] + factor[:, None] * (self.knots_mm[i + 1] - self.knots_mm[i])
        # Retain supplied knots exactly, including the final logical endpoint.
        for k, position in enumerate(self.knot_positions_mm):
            xyz[flat == position] = self.knots_mm[k]
        return xyz.reshape(q.shape + (3,))

    def to_parameters(self):
        return dict(kind='polyline', knots_mm=self.knots_mm.tolist(),
            knot_positions_mm=self.knot_positions_mm.tolist(), length_mm=self.length_mm,
            is_closed=self.is_closed, units='mm', parameterization='physical_arc_length')

    def segment(self, a, b):
        _valid_child_span(self, a, b)
        lo, hi = sorted((a, b))
        interior = self.knot_positions_mm[(self.knot_positions_mm > lo) & (self.knot_positions_mm < hi)]
        breaks = np.r_[a, interior if b > a else interior[::-1], b]
        sampled = []
        for left, right in zip(breaks[:-1], breaks[1:]):
            n = max(1, int(math.ceil(abs(right - left))))
            if len(sampled) + n > 1_000_000:
                raise PrimitiveError('EXPORT_SAMPLE_LIMIT_EXCEEDED: child retained in FP ledger')
            sampled.extend(np.linspace(left, right, n + 1)[:-1].tolist())
        sampled.append(b)
        q = np.asarray(sampled, np.float64)
        points = self.point_at(q)
        physical_steps = np.abs(np.diff(q))
        chord_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if (not np.isfinite(points).all() or not np.isfinite(chord_lengths).all()
                or np.any(chord_lengths <= 0.) or physical_steps.max() > 1. + 1e-12
                or not np.allclose(chord_lengths, physical_steps, rtol=1e-9, atol=1e-9)):
            raise PrimitiveError('UNREPRESENTABLE_POLYLINE_SAMPLES')
        length = float(chord_lengths.sum())
        if not math.isclose(length, abs(b - a), rel_tol=1e-10, abs_tol=1e-9):
            raise PrimitiveError('POLYLINE_LENGTH_MISMATCH')
        return dict(kind='polyline', points=points.tolist(), length_mm=abs(b - a),
            is_closed=self.is_closed and abs(b - a) == self.length_mm,
            ordered_knots_mm=self.point_at(breaks).tolist(), source_knot_positions_mm=breaks.tolist(),
            sample_positions_mm=np.abs(q - a).tolist(),
            sampling_audit=dict(max_arc_step_mm=float(physical_steps.max()), max_chord_error_mm=0.,
                all_supplied_corners_retained=True, length_basis='sum of exact ordered polyline edge lengths'))


def _valid_child_span(primitive, a, b):
    if not all(math.isfinite(v) for v in (a, b)) or a == b:
        raise PrimitiveError('ZERO_OR_NONFINITE_CHILD_SPAN')
    _positions([a, b], primitive.length_mm, False)


def parse_primitive(parameters):
    if not isinstance(parameters, Mapping) or parameters.get('units', 'mm') != 'mm':
        raise PrimitiveError('typed millimetre geometry mapping required')
    kind = parameters.get('kind')
    if kind == 'line':
        return LinePrimitive(parameters['start_mm'], parameters['end_mm'])
    if kind == 'arc':
        return ArcPrimitive(CircularArc.from_parameters(parameters['arc_parameters']))
    if kind == 'polyline':
        return PolylinePrimitive(parameters['knots_mm'], parameters['is_closed'])
    raise PrimitiveError('NOT_IMPLEMENTED: primitive kind ' + str(kind))


@dataclass(frozen=True)
class RuntimeProvenance:
    observation_kind: str
    observation_sha256: str
    geometry_provider_version: str
    feature_provider_version: str
    raw_provider_version: str
    raw_output_kind: str

    def __post_init__(self):
        if self.observation_kind not in ('runtime_pcd', 'synthetic_fixture'):
            raise PrimitiveError('GT/STEP/name/runtime-cache geometry provenance is forbidden')
        if self.raw_output_kind not in ('actual_head_output', 'synthetic_fields'):
            raise PrimitiveError('skipped/zero-placeholder fields are not actual head output')
        if (self.observation_kind == 'synthetic_fixture') != (self.raw_output_kind == 'synthetic_fields'):
            raise PrimitiveError('raw output kind must match observation provenance')
        if len(self.observation_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.observation_sha256):
            raise PrimitiveError('observation SHA256 required')
        if any(not isinstance(v, str) or not v for v in (self.geometry_provider_version,
                self.feature_provider_version, self.raw_provider_version)):
            raise PrimitiveError('explicit provider versions required')

    def to_dict(self):
        return dict(self.__dict__)


@dataclass(frozen=True)
class RuntimePrimitiveInput:
    path_id: int
    weld_type: str
    geometry_parameters: object
    query_positions_mm: object
    raw7: object
    feature_provider: object
    provenance: RuntimeProvenance

    def __post_init__(self):
        object.__setattr__(self, 'path_id', ordinal(self.path_id, 'path_id'))
        if self.weld_type not in ('horizontal', 'vertical'):
            raise PrimitiveError('explicit weld_type horizontal/vertical required')
        if not isinstance(self.geometry_parameters, Mapping):
            raise PrimitiveError('geometry_parameters mapping required')
        # Strict JSON copy; typed invalid geometry remains a ledger-visible path.
        geometry = _freeze_json(json.loads(canonical_json(self.geometry_parameters)))
        q = np.array(self.query_positions_mm, np.float64, copy=True)
        raw = np.array(self.raw7, copy=True)
        if q.ndim != 1 or not np.isfinite(q).all() or np.any(q < 0.) or (len(q) and q[0] != 0.) or np.any(np.diff(q) <= 0.):
            raise PrimitiveError('strictly increasing finite physical queries starting at zero required')
        if raw.dtype != np.float32 or raw.shape != (len(q), 7) or not np.isfinite(raw).all():
            raise PrimitiveError('finite float32 raw[len(q),7] required')
        if not callable(self.feature_provider) or not isinstance(self.provenance, RuntimeProvenance):
            raise PrimitiveError('explicit feature provider and runtime provenance required')
        object.__setattr__(self, 'geometry_parameters', geometry)
        object.__setattr__(self, 'query_positions_mm', _seal(q))
        object.__setattr__(self, 'raw7', _seal(raw))


@dataclass(frozen=True)
class PrimitiveProposal:
    runtime_id: tuple
    path_id: int
    role: str
    query_position_mm: float
    predicted_position_mm: float
    predicted_xyz_mm: object
    support: float
    raw7: object
    pcd_features425: object
    geometry_error: str = ''

    def __post_init__(self):
        identity = tuple(ordinal(v, 'runtime_id') for v in self.runtime_id)
        if len(identity) != 3 or self.role not in ROLES or identity[0] != ordinal(self.path_id) or identity[2] != ROLES.index(self.role):
            raise PrimitiveError('runtime_id=(path,query,role) must match proposal')
        if not all(math.isfinite(float(v)) for v in (self.query_position_mm, self.predicted_position_mm, self.support)) or not 0. <= self.support <= 1.:
            raise PrimitiveError('finite q/centre/probability required')
        xyz = self.predicted_xyz_mm
        if xyz is None:
            if not self.geometry_error:
                raise PrimitiveError('missing XYZ requires explicit geometry error')
        else:
            xyz = tuple(_frozen_array(xyz, (3,), 'predicted XYZ'))
        for key, size in (('raw7', 7), ('pcd_features425', 425)):
            a = np.asarray(getattr(self, key))
            if a.dtype != np.float32 or a.shape != (size,) or not np.isfinite(a).all():
                raise PrimitiveError('finite float32 ' + key + ' required')
            object.__setattr__(self, key, _seal(a))
        object.__setattr__(self, 'runtime_id', identity)
        object.__setattr__(self, 'predicted_xyz_mm', xyz)


@dataclass(frozen=True)
class PrimitiveContext:
    path_id: int
    weld_type: str
    primitive: object
    supplied_geometry: object
    geometry_error: str
    query_positions_mm: object
    raw7: object
    provenance: RuntimeProvenance
    fg_prefix: object
    fg_square_prefix: object
    fg_integral_prefix: object

    @property
    def query_count(self):
        return len(self.query_positions_mm)

    @property
    def origin_mm(self):
        if self.primitive is None:
            raise PrimitiveError('INVALID_PRIMITIVE_HAS_NO_ORIGIN')
        return tuple(self.primitive.point_at(0.))

    def bound_metadata(self):
        return dict(path_id=self.path_id, weld_type=self.weld_type, geometry=self.supplied_geometry,
            canonical_geometry=None if self.primitive is None else self.primitive.to_parameters(),
            geometry_error=self.geometry_error, provenance=self.provenance.to_dict(),
            FG_feature_version=FG_VERSION,
            query_positions_sha256=array_sha(self.query_positions_mm), raw_sha256=array_sha(self.raw7),
            fg_prefix_sha256=array_sha(self.fg_prefix), fg_square_prefix_sha256=array_sha(self.fg_square_prefix),
            fg_integral_prefix_sha256=array_sha(self.fg_integral_prefix))


class PrimitiveQueryGraph:
    """Complete role relations, independent of q order/geometry/GT validity."""
    def __init__(self, proposals, contexts, receipts):
        proposals, contexts = tuple(proposals), tuple(contexts)
        if not all(isinstance(p, PrimitiveContext) for p in contexts) or not all(isinstance(n, PrimitiveProposal) for n in proposals):
            raise PrimitiveError('typed primitive contexts and proposals required')
        self.nodes = tuple(sorted(proposals, key=lambda n: n.runtime_id))
        if len({n.runtime_id for n in self.nodes}) != len(self.nodes):
            raise PrimitiveError('duplicate explicit runtime ID')
        self.path_contexts = MappingProxyType({p.path_id: p for p in contexts})
        if len(self.path_contexts) != len(contexts):
            raise PrimitiveError('duplicate primitive path ID')
        for n in self.nodes:
            if n.path_id not in self.path_contexts:
                raise PrimitiveError('proposal path context missing; cannot omit from full relations')
            context = self.path_contexts[n.path_id]
            query = n.runtime_id[1]
            if query >= context.query_count or n.query_position_mm != context.query_positions_mm[query]:
                raise PrimitiveError('proposal query ID/physical position differs from context')
            if not np.array_equal(n.raw7, context.raw7[query]):
                raise PrimitiveError('proposal raw7 differs from bound full runtime fields')
            role = ROLES.index(n.role)
            expected_c = math.floor(n.query_position_mm / 64.) * 64. + float(n.raw7[4 + role]) * 16.
            if n.predicted_position_mm != expected_c:
                raise PrimitiveError('proposal centre differs from actual-q raw phase')
            if not n.geometry_error:
                if context.primitive is None or not np.array_equal(context.primitive.point_at(expected_c), n.predicted_xyz_mm):
                    raise PrimitiveError('proposal XYZ differs from typed runtime primitive')
        self.admission_receipts = _freeze_json(receipts)
        self.index = MappingProxyType({node.runtime_id: i for i, node in enumerate(self.nodes)})
        self.paths = MappingProxyType({path: _seal(np.asarray([i for i, n in enumerate(self.nodes) if n.path_id == path], np.int64))
                      for path in sorted(self.path_contexts)})
        self.roles = MappingProxyType({path: tuple(_seal(np.asarray([i for i in ids if self.nodes[i].role == r], np.int64))
                      for r in ROLES) for path, ids in self.paths.items()})
        self._child_buckets = []
        offset = 0
        for path, roles in self.roles.items():
            for a, b in CHILD_ROLE_PAIRS:
                count = len(roles[a]) * len(roles[b]) - (len(roles[a]) if a == b else 0)
                if count:
                    self._child_buckets.append((offset, offset + count, path, a, b, roles[a], roles[b]))
                offset += count
        self._child_buckets = tuple(self._child_buckets)

    def pair_count(self, kind='child'):
        if kind == 'child':
            return sum(sum(len(r[a]) * len(r[b]) - (len(r[a]) if a == b else 0)
                           for a, b in CHILD_ROLE_PAIRS) for r in self.roles.values())
        if kind == 'message':
            return sum(len(ids) * (len(ids) - 1) for ids in self.paths.values())
        if kind == 'duplicate':
            return sum(len(ids) * (len(ids) - 1) // 2 for ids in self.paths.values())
        raise PrimitiveError('child/message/duplicate pair kind required')

    def iter_pairs(self, kind='child', chunk_size=65536):
        if kind not in ('child', 'message', 'duplicate') or ordinal(chunk_size) == 0:
            raise PrimitiveError('valid kind and positive chunk size required')
        for path, ids in self.paths.items():
            groups = [(self.roles[path][a], self.roles[path][b], a == b) for a, b in CHILD_ROLE_PAIRS] if kind == 'child' else [(ids, ids, True)]
            for source, target, may_self in groups:
                if not len(source) or not len(target):
                    continue
                for start in range(0, len(source) * len(target), chunk_size):
                    linear = np.arange(start, min(start + chunk_size, len(source) * len(target)), dtype=np.int64)
                    pairs = np.c_[source[linear // len(target)], target[linear % len(target)]]
                    if kind == 'duplicate':
                        pairs = pairs[pairs[:, 0] < pairs[:, 1]]
                    elif may_self:
                        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
                    if len(pairs):
                        yield pairs

    def iter_child_pairs(self, chunk_size=65536):
        return self.iter_pairs('child', chunk_size)

    def iter_message_pairs(self, chunk_size=65536):
        return self.iter_pairs('message', chunk_size)

    def iter_duplicate_pairs(self, chunk_size=65536):
        return self.iter_pairs('duplicate', chunk_size)

    def has_child_pair(self, source, target):
        try:
            source, target = ordinal(source), ordinal(target)
        except PrimitiveError:
            return False
        if source == target or source >= len(self.nodes) or target >= len(self.nodes):
            return False
        a, b = self.nodes[source], self.nodes[target]
        return a.path_id == b.path_id and (ROLES.index(a.role), ROLES.index(b.role)) in CHILD_ROLE_PAIRS

    def child_edge_id(self, source, target):
        if not self.has_child_pair(source, target):
            raise PrimitiveError('pair absent from full child graph')
        node = self.nodes[source]
        a, b = ROLES.index(node.role), ROLES.index(self.nodes[target].role)
        for begin, end, path, ra, rb, sources, targets in self._child_buckets:
            if (path, ra, rb) == (node.path_id, a, b):
                i, j = int(np.searchsorted(sources, source)), int(np.searchsorted(targets, target))
                local = i * (len(targets) - 1) + (j if j < i else j - 1) if a == b else i * len(targets) + j
                return begin + local
        raise PrimitiveError('missing edge bucket')

    def child_edge_pair(self, identity):
        identity = ordinal(identity)
        for begin, end, path, a, b, source, target in self._child_buckets:
            if begin <= identity < end:
                local = identity - begin
                if a == b:
                    i, j = divmod(local, len(target) - 1)
                    j = j if j < i else j + 1
                else:
                    i, j = divmod(local, len(target))
                return int(source[i]), int(target[j])
        raise PrimitiveError('child edge ID outside full graph')

    def duplicate_pair_id(self, left, right):
        left, right = sorted((ordinal(left), ordinal(right)))
        if left == right or right >= len(self.nodes) or self.nodes[left].path_id != self.nodes[right].path_id:
            raise PrimitiveError('unordered distinct same-path pair required')
        offset = 0
        for path, ids in self.paths.items():
            if path == self.nodes[left].path_id:
                i, j, n = int(np.searchsorted(ids, left)), int(np.searchsorted(ids, right)), len(ids)
                return offset + i * (2 * n - i - 1) // 2 + j - i - 1
            offset += len(ids) * (len(ids) - 1) // 2
        raise PrimitiveError('missing duplicate path')

    def node_arrays(self):
        n = len(self.nodes)
        result = dict(runtime_ids=np.asarray([v.runtime_id for v in self.nodes], np.int64).reshape(n, 3),
            features425=np.asarray([v.pcd_features425 for v in self.nodes], np.float32).reshape(n, 425),
            raw7=np.asarray([v.raw7 for v in self.nodes], np.float32).reshape(n, 7),
            roles=np.asarray([ROLES.index(v.role) for v in self.nodes], np.int64),
            query_position_mm=np.asarray([v.query_position_mm for v in self.nodes], np.float64),
            predicted_position_mm=np.asarray([v.predicted_position_mm for v in self.nodes], np.float64),
            predicted_xyz_mm=np.asarray([v.predicted_xyz_mm if v.predicted_xyz_mm is not None else (np.nan,) * 3
                                       for v in self.nodes], np.float64).reshape(n, 3),
            support=np.asarray([v.support for v in self.nodes], np.float64))
        return {key: _seal(a) for key, a in result.items()}

    def node_tensors(self):
        # This is a NumPy interface, not an assertion of trained 425 semantics.
        if any(n.geometry_error for n in self.nodes):
            raise PrimitiveError('INVALID_GEOMETRY_NUMERIC_FEATURES_UNAVAILABLE: candidates retained in ledger')
        a = self.node_arrays()
        relative = a['predicted_xyz_mm'].copy()
        for path, ids in self.paths.items():
            relative[ids] -= np.asarray(self.path_contexts[path].origin_mm)
        geometry = np.c_[a['query_position_mm'], a['predicted_position_mm'],
            a['predicted_position_mm'] - a['query_position_mm'], relative, a['support']]
        return dict(node_features=a['features425'], raw7=a['raw7'], roles=a['roles'], node_geometry=geometry)

    def pair_features(self, pairs):
        p = _pair_array(pairs)
        if len(p) and (p.min() < 0 or p.max() >= len(self.nodes)):
            raise PrimitiveError('node index outside graph')
        out = np.empty((len(p), 16), np.float32)
        for k, (s, t) in enumerate(p):
            a, b = self.nodes[s], self.nodes[t]
            if a.path_id != b.path_id:
                raise PrimitiveError('same-path pairs required')
            if a.geometry_error or b.geometry_error:
                raise PrimitiveError('INVALID_GEOMETRY_PAIR_FEATURES_UNAVAILABLE: full pair retained')
            dq, dc = b.query_position_mm - a.query_position_mm, b.predicted_position_mm - a.predicted_position_mm
            delta = np.asarray(b.predicted_xyz_mm) - a.predicted_xyz_mm
            context = self.path_contexts[a.path_id]
            lo, hi = sorted((a.runtime_id[1], b.runtime_id[1]))
            count = hi - lo + 1
            total = context.fg_prefix[hi + 1] - context.fg_prefix[lo]
            square = context.fg_square_prefix[hi + 1] - context.fg_square_prefix[lo]
            mean = total / count
            std = 0. if count == 1 else math.sqrt(max(square / count - mean * mean, 0.))
            integral = float(context.fg_integral_prefix[hi] - context.fg_integral_prefix[lo])
            # Retain frozen 195's uniform-grid numerical definition exactly.
            q = context.query_positions_mm
            if len(q) >= 2 and np.array_equal(q, np.arange(len(q), dtype=np.float64) * (q[1] - q[0])):
                first = context.fg_prefix[lo + 1] - context.fg_prefix[lo]
                last = context.fg_prefix[hi + 1] - context.fg_prefix[hi]
                integral = (total - .5 * (first + last)) * (q[1] - q[0])
            out[k] = np.r_[dq, dc, delta, np.linalg.norm(delta), abs(dq), mean, std, integral,
                          np.eye(3)[ROLES.index(a.role)], np.eye(3)[ROLES.index(b.role)]]
        if not np.isfinite(out).all():
            raise PrimitiveError('finite pair features required')
        return out

    def complete_hashes(self, chunk_size=65536):
        arrays = self.node_arrays()
        identities = array_sha(arrays['runtime_ids'])
        h = sha256()
        for key in sorted(arrays):
            h.update(key.encode())
            h.update(array_sha(arrays[key]).encode())
        h.update(canonical_json([n.geometry_error for n in self.nodes]).encode())
        metadata = [self.path_contexts[p].bound_metadata() for p in sorted(self.path_contexts)]
        result = dict(schema_version=VERSION, node_count=len(self.nodes), node_identity_sha256=identities,
            node_payload_sha256=h.hexdigest(), typed_geometry_sha256=sha256(canonical_json(metadata).encode()).hexdigest(), pairs={})
        for kind in ('child', 'message', 'duplicate'):
            digest = sha256((VERSION + '|' + kind + '|' + identities).encode())
            count = 0
            for pairs in self.iter_pairs(kind, chunk_size):
                digest.update(pairs.astype('<i8', copy=False).tobytes())
                count += len(pairs)
            if count != self.pair_count(kind):
                raise PrimitiveError('full enumeration disagrees with analytic count')
            result['pairs'][kind] = dict(count=count, sha256=digest.hexdigest(), complete_enumeration=True,
                                        no_geometry_or_GT_truncation=True)
        return result

    def fingerprint(self, chunk_size=65536):
        return sha256(canonical_json(self.complete_hashes(chunk_size)).encode()).hexdigest()


def build_primitive_graph(inputs):
    nodes, contexts, receipts = [], [], {}
    for path in inputs:
        if not isinstance(path, RuntimePrimitiveInput) or path.path_id in receipts:
            raise PrimitiveError('unique RuntimePrimitiveInput paths required')
        geometry_error = ''
        try:
            primitive = parse_primitive(path.geometry_parameters)
        except (PrimitiveError, CurveGeometryError, KeyError, TypeError, ValueError) as exc:
            primitive = None
            geometry_error = 'INVALID_RUNTIME_PRIMITIVE: ' + str(exc)
        q, raw = path.query_positions_mm, path.raw7
        if primitive is not None and len(q) >= 2 and q[-1] != primitive.length_mm:
            geometry_error = 'QUERY_COVERAGE_MISMATCH: last query must equal actual primitive length exactly'
        centres = np.floor(q[:, None] / 64.) * 64. + raw[:, 4:7].astype(np.float64) * 16.
        support = 1. / (1. + np.exp(-np.clip(raw[:, 1:4].astype(np.float64), -60., 60.)))
        span = float(q[-1]) if len(q) else 0.
        flags = dict(support=support >= .5, radius=np.abs((centres - q[:, None]) / 16.) <= 1. + 1e-6,
                     span=(centres >= -RESOLUTION_MM) & (centres <= span + RESOLUTION_MM))
        admitted = flags['support'] & flags['radius'] & flags['span']
        ids = np.flatnonzero(admitted.any(axis=1))
        features = np.asarray(path.feature_provider(ids.copy()))
        if features.dtype != np.float32 or features.shape != (len(ids), 425) or not np.isfinite(features).all():
            raise PrimitiveError('provider must return finite float32[len(admitted_unique_queries),425]')
        for row, query in enumerate(ids):
            for role in np.flatnonzero(admitted[query]):
                c = float(centres[query, role])
                error = geometry_error
                xyz = None
                if primitive is not None:
                    xyz = tuple(float(v) for v in primitive.point_at(c, allow_extension=True))
                    if not all(math.isfinite(v) for v in xyz):
                        xyz = None
                        error = error or 'NONFINITE_GEOMETRY_POSITION'
                    if not 0. <= c <= primitive.length_mm:
                        error = error or 'PREDICTED_CENTRE_OUTSIDE_GEOMETRY: original centre retained'
                feature = features[row].copy()
                feature.setflags(write=False)
                nodes.append(PrimitiveProposal((path.path_id, int(query), int(role)), path.path_id, ROLES[role],
                    float(q[query]), c, xyz, float(support[query, role]), raw[query], feature, error))
        fg = 1. / (1. + np.exp(-np.clip(raw[:, 0].astype(np.float64), -60., 60.)))
        prefix = np.r_[0., np.cumsum(fg)]
        square = np.r_[0., np.cumsum(fg * fg)]
        integral = np.r_[0., np.cumsum(.5 * (fg[:-1] + fg[1:]) * np.diff(q))] if len(q) else np.empty(0, np.float64)
        prefix, square, integral = (_seal(a) for a in (prefix, square, integral))
        contexts.append(PrimitiveContext(path.path_id, path.weld_type, primitive, path.geometry_parameters,
            geometry_error, q, raw, path.provenance, prefix, square, integral))
        receipts[path.path_id] = dict(queries=len(q), geometry_error=geometry_error,
            raw_output_kind=path.provenance.raw_output_kind, FG_feature_version=FG_VERSION,
            admitted_by_role={r: int(admitted[:, i].sum()) for i, r in enumerate(ROLES)},
            raw_sha256=array_sha(raw), query_positions_sha256=array_sha(q), provider425_sha256=array_sha(features),
            gates_sha256={k: array_sha(v) for k, v in flags.items()}, admitted_sha256=array_sha(admitted),
            provider_admitted_unique_queries=len(ids), support_threshold=.5, radius_mm=16.,
            radius_offset_tolerance=1e-6, span_tolerance_mm=RESOLUTION_MM,
            no_clip_or_rint_of_predicted_centres=True, all_admitted_queries_retained=True,
            invalid_geometry_candidates_retained=True, actual_PCD_provider_verified=False,
            V_curve_feature_semantics='NOT_IMPLEMENTED', V_curve_raw7_model='UNTRAINED')
    return PrimitiveQueryGraph(nodes, contexts, receipts)


def _pair_array(value):
    a = np.asarray(value)
    if a.size and a.dtype.kind not in 'iu':
        raise PrimitiveError('integer pair indices required')
    if not a.size:
        return np.empty((0, 2), np.int64)
    if a.ndim != 2 or a.shape[1] != 2:
        raise PrimitiveError('pairs[N,2] required')
    return a.astype(np.int64, copy=False)


def validate_selected(graph, pairs):
    """195 degree/DAG semantics; coordinates never prune selected relations."""
    p = _pair_array(pairs)
    selected, incoming, outgoing = set(), {}, {}
    for s, t in p.tolist():
        if not graph.has_child_pair(s, t) or (s, t) in selected:
            raise PrimitiveError('absent or repeated selected child edge')
        selected.add((s, t))
        incoming.setdefault(t, []).append(s)
        outgoing.setdefault(s, []).append(t)
    used = set(incoming) | set(outgoing)
    for i in used:
        ni, no, role = len(incoming.get(i, ())), len(outgoing.get(i, ())), graph.nodes[i].role
        legal = (ni == 0 and no == 1) if role == 'start' else ((ni == 1 and no == 0) if role == 'end' else ni == no == 1)
        if not legal:
            raise PrimitiveError('illegal S/E/C selected degree')
    chains, visited, edges = [], set(), set()
    for start in sorted(i for i in used if graph.nodes[i].role == 'start'):
        chain, current = [start], start
        while graph.nodes[current].role != 'end':
            target = outgoing[current][0]
            if target in chain or target in visited:
                raise PrimitiveError('cycle or reused selected boundary')
            edges.add((current, target))
            chain.append(target)
            current = target
        visited.update(chain)
        chains.append(tuple(chain))
    if visited != used or edges != selected:
        raise PrimitiveError('disconnected cut cycle')
    return dict(chains=chains, selected_nodes=len(used), selected_children=len(selected),
                selected_parents=len(chains), empty_graph_legal=True, acyclic=True)


def export_selected(graph, pairs):
    """Every submitted child stays in the ledger; invalid geometry is an FP.

    Valid sampled exports do not establish matching TP or contract accuracy.
    Polyline exports use a new typed representation, not 193's line/arc schema.
    """
    p = _pair_array(pairs)
    structural_error = ''
    try:
        validation = validate_selected(graph, p)
    except PrimitiveError as exc:
        structural_error = str(exc)
        validation = dict(chains=[], selected_children=len(p), selected_parents=0, structural_error=structural_error)
    edge_order = {}
    direction_errors = {}
    for parent, chain in enumerate(validation['chains']):
        spans = [graph.nodes[b].predicted_position_mm - graph.nodes[a].predicted_position_mm for a, b in zip(chain[:-1], chain[1:])]
        direction = 1 if spans[0] > 0. else -1
        wrong = any(not math.isfinite(v) or direction * v <= 0. for v in spans)
        for order, pair in enumerate(zip(chain[:-1], chain[1:])):
            edge_order[pair] = (parent, order)
            if wrong:
                direction_errors[pair] = 'INCONSISTENT_PARENT_PARAMETER_DIRECTION'
    ledger, welds = [], {}
    for submitted_ordinal, (s, t) in enumerate(p.tolist()):
        row = dict(submitted_child_ordinal=submitted_ordinal, source_node_ordinal=s, target_node_ordinal=t,
                   status='INVALID_GEOMETRY_FP', counts_as_prediction=True, invalid_geometry_FP=True,
                   edge_id=None, segment=None)
        if 0 <= s < len(graph.nodes):
            node = graph.nodes[s]
            row['start_node_id'] = list(node.runtime_id)
            row['start_physical_q_mm'] = node.predicted_position_mm
            row['start_xyz_mm'] = node.predicted_xyz_mm
        if 0 <= t < len(graph.nodes):
            node = graph.nodes[t]
            row['end_node_id'] = list(node.runtime_id)
            row['end_physical_q_mm'] = node.predicted_position_mm
            row['end_xyz_mm'] = node.predicted_xyz_mm
        if graph.has_child_pair(s, t):
            row['edge_id'] = graph.child_edge_id(s, t)
        try:
            if structural_error:
                raise PrimitiveError('INVALID_SELECTED_STRUCTURE: ' + structural_error)
            if not graph.has_child_pair(s, t):
                raise PrimitiveError('absent role-legal child')
            a, b = graph.nodes[s], graph.nodes[t]
            row['edge_id'] = graph.child_edge_id(s, t)
            row['start_node_id'], row['end_node_id'] = list(a.runtime_id), list(b.runtime_id)
            context = graph.path_contexts[a.path_id]
            if a.geometry_error or b.geometry_error:
                raise PrimitiveError(a.geometry_error or b.geometry_error)
            if (s, t) in direction_errors:
                raise PrimitiveError(direction_errors[(s, t)])
            segment = context.primitive.segment(a.predicted_position_mm, b.predicted_position_mm)
            segment.update(start_node_id=list(a.runtime_id), end_node_id=list(b.runtime_id),
                           runtime_child_edge_id=row['edge_id'], geometry_version=VERSION)
            row.update(status='VALID_GEOMETRY_UNSCORED', invalid_geometry_FP=False, segment=segment)
            parent, order = edge_order[(s, t)]
            weld = welds.setdefault(parent, dict(runtime_parent_id=parent, type=context.weld_type,
                segments=[], primitive_geometry=context.primitive.to_parameters(), geometry_version=VERSION,
                recognition_status='NOT_RUN', attributes_status='UNKNOWN'))
            weld['segments'].append((order, segment))
        except (PrimitiveError, CurveGeometryError, ValueError, IndexError, OverflowError) as exc:
            row['reason'] = str(exc)
        ledger.append(row)
    output = []
    parent_ledger = []
    rows_by_pair = {(r['source_node_ordinal'], r['target_node_ordinal']): r for r in ledger}
    for parent, chain in enumerate(validation['chains']):
        rows = [rows_by_pair[pair] for pair in zip(chain[:-1], chain[1:])]
        state = dict(runtime_parent_id=parent, selected_child_edge_ids=[r['edge_id'] for r in rows],
            delivered_child_edge_ids=[r['edge_id'] for r in rows if not r['invalid_geometry_FP']],
            invalid_child_edge_ids=[r['edge_id'] for r in rows if r['invalid_geometry_FP']],
            complete_parent_geometry=all(not r['invalid_geometry_FP'] for r in rows),
            start_node_id=list(graph.nodes[chain[0]].runtime_id), end_node_id=list(graph.nodes[chain[-1]].runtime_id))
        parent_ledger.append(state)
        if parent in welds:
            welds[parent].update(state)
    for parent in sorted(welds):
        weld = welds[parent]
        weld['segments'] = [s for _, s in sorted(weld['segments'], key=lambda x: x[0])]
        output.append(weld)
    invalid = sum(r['invalid_geometry_FP'] for r in ledger)
    return dict(welds=output, child_ledger=ledger, parent_ledger=parent_ledger, graph_fingerprint=graph.fingerprint(),
        selected_children=len(p), delivered_geometry_children=len(p) - invalid, invalid_geometry_FP=invalid,
        prediction_count_including_invalid=len(p), no_submitted_child_removed=True,
        structural_validation=validation, learned_accuracy_measured=False, full_contract_pass=False)


def discover_primitives_from_pcd(*args, **kwargs):
    raise NotImplementedError('STANDBY: pure-PCD generic 3D primitive discovery provider NOT_IMPLEMENTED')


def build_V_curve_features425(*args, **kwargs):
    raise NotImplementedError('STANDBY: V/curve 425 feature semantics/provider NOT_IMPLEMENTED')


def predict_V_curve_raw7(*args, **kwargs):
    raise NotImplementedError('STANDBY: V/curve raw7 model UNTRAINED; Hline weights cannot establish this scope')
