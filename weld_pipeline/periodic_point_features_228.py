"""Query-centred observed inputs for lines, arcs and piecewise linear paths.

World XY/XZ/YZ grids do not switch a tangent-dependent reference axis. They
describe sampling only; no grid, covariance or curve normal is material truth.
This schema needs a new FIT-only normalizer and a newly trained prediction head.
It cannot be passed as the previous 425 channels or normalized with their stats.
"""
from hashlib import sha256
import math

import numpy as np

from .primitive_incidence_209 import (
    LinePrimitive, ArcPrimitive, PolylinePrimitive, array_sha, canonical_json,
)
from .runtime_primitive_provider_adapter_212 import RuntimeQueryPath, ProviderError
from .typed_point_features_217 import PointObservation, _sealed, _unit_rows

VERSION = 'query_centred_periodic_world_observation_228_v1'
LOCAL_OFFSETS_MM = _sealed(np.array([-25., -12., -6., 0., 6., 12., 25.]))
REFERENCE_OFFSETS_MM = _sealed(np.array([-16., -8., 0., 8., 16.]))
PLANES = (('xy', 0, 1), ('xz', 0, 2), ('yz', 1, 2))
RADII_MM = (5., 20.)
DENSITY_STEP_MM = 2.
DISTANCE_LIMIT_MM = 50.


def _grid_names(prefix, size):
    return tuple(f'{prefix}_{field}_{plane}_{i:03d}'
                 for field in ('distance_over50', 'within50')
                 for plane, _, _ in PLANES for i in range(size * size))


MOMENT_NAMES = ('log_count_over20', 'mean_x', 'mean_y', 'mean_z',
                'cov_xx', 'cov_xy', 'cov_xz', 'cov_yy', 'cov_yz', 'cov_zz',
                'eigen_0', 'eigen_1', 'eigen_2', 'has_points', 'count_ge4')
CONTEXT_NAMES = (
    'relative_mean_x_over1000', 'relative_mean_y_over1000', 'relative_mean_z_over1000',
    'relative_mean_x_over_std', 'relative_mean_y_over_std', 'relative_mean_z_over_std',
    'global_cov_xx_over_trace', 'global_cov_xy_over_trace', 'global_cov_xz_over_trace',
    'global_cov_yy_over_trace', 'global_cov_yz_over_trace', 'global_cov_zz_over_trace',
    'observed_min_dx_over1000', 'observed_max_dx_over1000',
    'observed_min_dy_over1000', 'observed_max_dy_over1000',
    'observed_min_dz_over1000', 'observed_max_dz_over1000',
    'observed_global_z_fraction', 'world_x_over10000', 'world_y_over10000', 'world_z_over10000',
)
REFERENCE_TAIL_NAMES = ('log_count_over20', 'eigen_0_over400', 'eigen_1_over400',
                        'eigen_2_over400', 'radius_mean_over20', 'radius_median_over20',
                        'radius_max_over20', 'has_points', 'count_ge4')
GEOMETRY_NAMES = ('length_over1000', 'kind_line', 'kind_arc', 'kind_polyline', 'closed',
                  'horizontal_hypothesis', 'vertical_hypothesis',
                  'tangent_before_x', 'tangent_before_y', 'tangent_before_z',
                  'tangent_after_x', 'tangent_after_y', 'tangent_after_z',
                  'within_segment_curvature_per_mm', 'corner_turn_over_pi',
                  'is_true_corner', 'is_true_open_endpoint')
PROXIMITY_NAMES = ('before_distance_over128', 'after_distance_over128',
                   'has_before_event', 'has_after_event', 'log_events_within16_over20')
DENSITY_NAMES = ('before_log_count_over20', 'query_log_count_over20', 'after_log_count_over20',
                 'log_count_derivative_per_mm_over20', 'log_count_second_derivative_per_mm2_over20',
                 'valid_before_probe', 'valid_after_probe', 'three_probe_local_max', 'three_probe_local_min')

LOCAL_NAMES = _grid_names('local', 7) + tuple(
    f'moment_r{int(radius)}_{name}' for radius in RADII_MM for name in MOMENT_NAMES) + CONTEXT_NAMES
REFERENCE_NAMES = _grid_names('reference', 5) + REFERENCE_TAIL_NAMES
NODE_NAMES = tuple(f'reference_{j}_{name}' for j in range(7) for name in REFERENCE_NAMES) + GEOMETRY_NAMES
FEATURE_NAMES = LOCAL_NAMES + NODE_NAMES + PROXIMITY_NAMES + DENSITY_NAMES
FEATURE_DIM = len(FEATURE_NAMES)
BLOCK_SLICES = dict(local=(0, 346), node=(346, 1476), geometry_events=(1476, 1481), density=(1481, 1490))
assert FEATURE_DIM == 1490 and len(set(FEATURE_NAMES)) == FEATURE_DIM
SCHEMA = dict(version=VERSION, dimension=FEATURE_DIM, fields=FEATURE_NAMES, blocks=BLOCK_SLICES,
              frame='FIXED_WORLD_XY_XZ_YZ_NOT_MATERIAL_NORMAL',
              closed_reference='(physical_query_phase + j*L/7) mod L; j=0..6',
              open_reference='j*L/6; j=0..6; genuine open ends included',
              density='query-centred real-mm probes; closed periodic; log1p counts; actual delta',
              minimum_covariance_sample_flag=4,
              count_ge4_meaning='algebraic sample-count flag; not covariance reliability or material truth',
              unsupported_fields=('absolute_q', 'L_minus_q', 'q_over_L', 'fixed64_phase', 'material_truth'))
SCHEMA_SHA256 = sha256(canonical_json(SCHEMA).encode()).hexdigest()


def physical_phase(primitive, query):
    q = np.asarray(query, np.float64)
    if q.ndim != 1 or not np.isfinite(q).all() or np.any(q < 0.) or np.any(q > primitive.length_mm):
        raise ProviderError('Finite physical query vector inside the complete primitive required')
    if primitive.is_closed:
        q = np.remainder(q, primitive.length_mm)
        q = np.where(q == primitive.length_mm, 0., q)
    return q


def _periodic(value, length):
    phase = np.remainder(value, length)
    return np.where(phase == length, 0., phase)


def reference_positions(primitive, query):
    phase = physical_phase(primitive, query)
    length = primitive.length_mm
    if primitive.is_closed:
        positions = _periodic(phase[:, None] + np.arange(7)[None, :] * (length / 7.), length)
    else:
        positions = np.broadcast_to(np.linspace(0., length, 7), (len(phase), 7)).copy()
    if len(positions) and np.any(np.diff(np.sort(positions, axis=1), axis=1) <= 0.):
        raise ProviderError('Seven reference phases not representable; retain whole path as unsupported')
    return positions


def one_sided_geometry(primitive, phase):
    """Forward tangents on both sides; a real corner has two distinct tangents."""
    phase = physical_phase(primitive, phase)
    hit = np.zeros(len(phase), bool)
    if isinstance(primitive, ArcPrimitive):
        radial = _unit_rows(primitive.point_at(phase) - primitive.arc.center_mm)
        tangent = math.copysign(1., primitive.arc.sweep_rad) * np.cross(primitive.arc.plane_normal, radial)
        before, after = tangent, tangent.copy()
        curvature = np.full(len(phase), 1. / primitive.arc.radius_mm)
    elif isinstance(primitive, LinePrimitive):
        tangent = (primitive.end_mm - primitive.start_mm) / primitive.length_mm
        before = np.broadcast_to(tangent, (len(phase), 3)).copy()
        after, curvature = before.copy(), np.zeros(len(phase))
    elif isinstance(primitive, PolylinePrimitive):
        positions = primitive.knot_positions_mm
        tangents = _unit_rows(np.diff(primitive.knots_mm, axis=0))
        left = np.searchsorted(positions, phase, side='left') - 1
        right = np.searchsorted(positions, phase, side='right') - 1
        if primitive.is_closed:
            left, right = left % len(tangents), right % len(tangents)
        else:
            left, right = np.clip(left, 0, len(tangents)-1), np.clip(right, 0, len(tangents)-1)
        before, after = tangents[left], tangents[right]
        hit = np.any(phase[:, None] == positions[None, :], axis=1)
        curvature = np.zeros(len(phase))  # Curvature inside each linear segment.
    else:
        raise ProviderError('Observed line/arc/polyline required')
    corner = hit & np.any(before != after, axis=1)
    turn = np.where(corner, np.arccos(np.clip((before * after).sum(1), -1., 1.)), 0.)
    endpoint = (not primitive.is_closed) & ((phase == 0.) | (phase == primitive.length_mm))
    return before, after, curvature, turn, corner, endpoint


def geometry_event_positions(primitive):
    ends = np.empty(0) if primitive.is_closed else np.array([0., primitive.length_mm])
    if isinstance(primitive, PolylinePrimitive):
        vertices = primitive.knot_positions_mm[:-1] if primitive.is_closed else primitive.knot_positions_mm
        _, _, _, _, corner, _ = one_sided_geometry(primitive, vertices)
        ends = np.r_[ends, vertices[corner]]
    return _sealed(np.unique(ends))


class PeriodicPointFeatureProvider:
    def __init__(self, observation, path, chunk_rows=32):
        if not isinstance(observation, PointObservation) or not isinstance(path, RuntimeQueryPath):
            raise ProviderError('Current observed XYZ and immutable query path required')
        if observation.binding != path.binding:
            raise ProviderError('Observation/primitive binding mismatch')
        if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
            raise ProviderError('Positive bounded feature chunk required')
        self.observation, self.path, self.chunk_rows = observation, path, chunk_rows
        self.geometry_events = geometry_event_positions(path.primitive)
        self.calls = dict(local=0, node=0, geometry_events=0, density=0)
        self.rows_produced = 0
        self.fingerprint = sha256(canonical_json(dict(schema=SCHEMA_SHA256,
            observation_xyz_sha256=observation.xyz_sha256, path=path.fingerprint,
            geometry_events_sha256=array_sha(self.geometry_events))).encode()).hexdigest()

    def nearest_world_grids(self, xyz, offsets):
        a, b = np.meshgrid(offsets, offsets)
        displacements = []
        for _, ia, ib in PLANES:
            delta = np.zeros((len(offsets)**2, 3))
            delta[:, ia], delta[:, ib] = a.ravel(), b.ravel()
            displacements.append(delta)
        delta = np.concatenate(displacements)
        probes = xyz[:, None, :] + delta[None, :, :]
        distances = self.observation.tree.query(probes.reshape(-1, 3), workers=1,
            distance_upper_bound=np.nextafter(DISTANCE_LIMIT_MM, np.inf))[0].reshape(len(xyz), len(delta))
        valid = np.isfinite(distances) & (distances <= DISTANCE_LIMIT_MM)
        return np.c_[np.minimum(distances, DISTANCE_LIMIT_MM)/DISTANCE_LIMIT_MM, valid.astype(float)]

    def _counts(self, xyz, radius):
        return np.asarray(self.observation.tree.query_ball_point(xyz, radius, workers=1, return_length=True))

    @staticmethod
    def _world_frames(count):
        return np.broadcast_to(np.eye(3), (count, 3, 3))

    def _moment15(self, xyz, radius):
        count = self._counts(xyz, radius)
        moments = self.observation.moment13(xyz, self._world_frames(len(xyz)), radius)
        return np.c_[moments, count > 0, count >= 4]

    def _reference159(self, xyz):
        distance = self.nearest_world_grids(xyz, REFERENCE_OFFSETS_MM)
        count = self._counts(xyz, 20.)
        # Last seven old reference fields are world-covariance/radius observations.
        tails = self.observation.reference32(xyz, self._world_frames(len(xyz)))[:, 25:]
        return np.c_[distance, tails, count > 0, count >= 4]

    def _local(self, xyz):
        self.calls['local'] += 1
        context = self.observation.workpiece22(xyz, self._world_frames(len(xyz)))[:, :19]
        return np.c_[self.nearest_world_grids(xyz, LOCAL_OFFSETS_MM),
                     self._moment15(xyz, 5.), self._moment15(xyz, 20.), context, xyz/10000.]

    def _node(self, phase):
        self.calls['node'] += 1
        primitive = self.path.primitive
        references = reference_positions(primitive, phase)
        points = primitive.point_at(references.reshape(-1))
        refs = self._reference159(points).reshape(len(phase), 7*159)
        before, after, curvature, turn, corner, endpoint = one_sided_geometry(primitive, phase)
        kind = [isinstance(primitive, t) for t in (LinePrimitive, ArcPrimitive, PolylinePrimitive)]
        fixed = np.broadcast_to([primitive.length_mm/1000., *kind, primitive.is_closed,
            self.path.weld_type == 'horizontal', self.path.weld_type == 'vertical'], (len(phase), 7))
        geometry = np.c_[fixed, before, after, curvature, turn/math.pi, corner, endpoint]
        return np.c_[refs, geometry]

    def _proximity(self, phase):
        self.calls['geometry_events'] += 1
        rows, pool = [], self.geometry_events
        for query in phase:
            if self.path.primitive.is_closed:
                left = _periodic(query-pool, self.path.primitive.length_mm)
                right = _periodic(pool-query, self.path.primitive.length_mm)
                nearby = int((np.minimum(left, right) <= 16.).sum())
            else:
                left, right = query-pool[pool <= query], pool[pool >= query]-query
                nearby = int((np.abs(pool-query) <= 16.).sum())
            rows.append([min(float(left.min()), 128.)/128. if len(left) else 0.,
                         min(float(right.min()), 128.)/128. if len(right) else 0.,
                         bool(len(left)), bool(len(right)), np.log1p(nearby)/20.])
        return np.asarray(rows, float).reshape(len(phase), 5)

    def _density(self, phase):
        self.calls['density'] += 1
        primitive, length = self.path.primitive, self.path.primitive.length_mm
        if primitive.is_closed:
            h = min(DENSITY_STEP_MM, length/3.)
            left, right = _periodic(phase-h, length), _periodic(phase+h, length)
            dl, dr = _periodic(phase-left, length), _periodic(right-phase, length)
        else:
            left, right = np.maximum(0., phase-DENSITY_STEP_MM), np.minimum(length, phase+DENSITY_STEP_MM)
            dl, dr = phase-left, right-phase
        lv, rv = dl > 0., dr > 0.
        probes = np.stack((left, phase, right), axis=1)
        counts = self._counts(primitive.point_at(probes.reshape(-1)), 5.).reshape(-1, 3)
        values = np.log1p(counts)/20.
        slope_left = np.divide(values[:, 1]-values[:, 0], dl, out=np.zeros(len(phase)), where=lv)
        slope_right = np.divide(values[:, 2]-values[:, 1], dr, out=np.zeros(len(phase)), where=rv)
        both, span = lv & rv, dl+dr
        derivative = np.where(lv, slope_left, slope_right)
        derivative[both] = (dr[both]*slope_left[both] + dl[both]*slope_right[both])/span[both]
        second = np.divide(2.*(slope_right-slope_left), span, out=np.zeros(len(phase)), where=both)
        maximum = both & (counts[:, 1] >= counts[:, 0]) & (counts[:, 1] >= counts[:, 2]) & (
            (counts[:, 1] > counts[:, 0]) | (counts[:, 1] > counts[:, 2]))
        minimum = both & (counts[:, 1] <= counts[:, 0]) & (counts[:, 1] <= counts[:, 2]) & (
            (counts[:, 1] < counts[:, 0]) | (counts[:, 1] < counts[:, 2]))
        return np.c_[values, derivative, second, lv, rv, maximum, minimum]

    def features_at(self, query):
        """Current physical positions; original query metadata remains separate."""
        phase = physical_phase(self.path.primitive, query)
        if len(phase) > self.chunk_rows:
            raise ProviderError('Feature request exceeds bounded chunk; use iter_features')
        if not len(phase):
            return np.empty((0, FEATURE_DIM), np.float32)
        xyz = self.path.primitive.point_at(phase)
        result = np.c_[self._local(xyz), self._node(phase), self._proximity(phase), self._density(phase)]
        if result.shape != (len(phase), FEATURE_DIM) or not np.isfinite(result).all():
            raise ProviderError('Full feature request unrepresentable; no row/primitive discarded')
        with np.errstate(over='ignore'):
            result = result.astype(np.float32)
        if not np.isfinite(result).all():
            raise ProviderError('Float32 feature range exceeded; retain whole path as unsupported')
        self.rows_produced += len(phase)
        return result

    def iter_features(self, indices):
        indices = self.path.indices(indices)
        for begin in range(0, len(indices), self.chunk_rows):
            i = indices[begin:begin+self.chunk_rows]
            yield i, self.features_at(self.path.q[i])

    def receipt(self):
        return dict(version=VERSION, feature_schema_sha256=SCHEMA_SHA256, feature_dimension=FEATURE_DIM,
            feature_provider_fingerprint=self.fingerprint, observed_xyz_sha256=self.observation.xyz_sha256,
            observation_binding=dict(kind=self.path.binding.kind,
                observation_sha256=self.path.binding.observation_sha256,
                proposal_provider_version=self.path.binding.proposal_provider_version),
            primitive_and_query_binding=self.path.fingerprint, complete_queries=len(self.path.q),
            geometry_event_positions_mm=self.geometry_events.tolist(),
            artificial_closed_endpoints_in_event_pool=False, callbacks_actually_called=dict(self.calls),
            rows_actually_produced=self.rows_produced, grid_axes_are_material_normals=False,
            count_ge4_is_reliability_or_material_truth=False, observation_only=True,
            density_feature_can_identify_all_process_events=False,
            original_absolute_q_retained_in_geometry_metadata=True,
            new_normalizer='NOT_FITTED', new_prediction_head='UNTRAINED',
            legacy_425_adapter_compatible=False, model_accuracy_measured=False, full_contract_pass=False)
