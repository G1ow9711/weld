"""Observed-point 425 features for true 3D parameterized paths.

New feature semantics require a new FIT-only normalizer and trained raw head.
Frames below are sampling axes, never measured material-side normals.
No GT, STEP, filename, checkpoint, optimizer or weld decoder is read here.
"""
from hashlib import sha256
import math

import numpy as np
from scipy.spatial import cKDTree

from .primitive_incidence_209 import (
    LinePrimitive, ArcPrimitive, PolylinePrimitive, array_sha, canonical_json,
)
from .runtime_primitive_provider_adapter_212 import (
    ObservationBinding, RuntimeQueryPath, FeatureBlockProviders,
    Chunked425Provider, ProviderError,
)

VERSION = 'observed_typed_path_distance_moments_context_425_217_v1'
OFFSETS_MM = np.asarray([-25., -12., -6., 0., 6., 12., 25.])
NODE_OFFSETS_MM = np.asarray([-16., -8., 0., 8., 16.])
REFERENCE_FRACTIONS = np.asarray([0., .125, .25, .5, .75, .875, 1.])
LOCAL_RADII_MM = (5., 20.)
UPPER = np.triu_indices(3)


def _sealed(a):
    a = np.ascontiguousarray(a)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _unit_rows(values):
    length = np.linalg.norm(values, axis=1)
    if not np.isfinite(length).all() or np.any(length <= 0.):
        raise ProviderError('Invalid sampling frame')
    return values / length[:, None]


def sampling_frames(primitive, q):
    """Rows are tangent, transverse sampling axis A, axis B; no material sign."""
    q = np.asarray(q, np.float64)
    xyz = primitive.point_at(q)
    if isinstance(primitive, ArcPrimitive):
        radial = _unit_rows(xyz - primitive.arc.center_mm)
        tangent = math.copysign(1., primitive.arc.sweep_rad) * np.cross(primitive.arc.plane_normal, radial)
        transverse = radial
    elif isinstance(primitive, (LinePrimitive, PolylinePrimitive)):
        if isinstance(primitive, LinePrimitive):
            tangent = np.broadcast_to((primitive.end_mm - primitive.start_mm) / primitive.length_mm,
                                      (len(q), 3)).copy()
        else:
            index = np.minimum(np.searchsorted(primitive.knot_positions_mm, q, side='right') - 1,
                               len(primitive.knots_mm) - 2)
            tangent = _unit_rows(np.diff(primitive.knots_mm, axis=0)[index])
        # Deterministic world-axis frame. The rule is shared across sources,
        # contains no lookup coordinates, and cannot imply outward orientation.
        reference = np.eye(3)[np.argmin(np.abs(tangent), axis=1)]
        transverse = _unit_rows(reference - (reference * tangent).sum(1)[:, None] * tangent)
    else:
        raise ProviderError('Typed line/arc/polyline primitive required')
    tangent = _unit_rows(tangent)
    third = _unit_rows(np.cross(tangent, transverse))
    frames = np.stack((tangent, transverse, third), axis=1)
    if not np.allclose(frames @ frames.transpose(0, 2, 1), np.eye(3), rtol=0., atol=1e-12):
        raise ProviderError('Nonorthonormal sampling frame')
    return frames


class PointObservation:
    """Numerical observation binding; provenance still needs an entry receipt."""
    def __init__(self, xyz_mm, binding):
        if not isinstance(binding, ObservationBinding):
            raise ProviderError('Current observation binding required')
        points = np.asarray(xyz_mm, np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ProviderError('Nonempty finite observed XYZ[vertices,3] in mm required')
        self.xyz = _sealed(points)
        self.binding = binding
        self.xyz_sha256 = array_sha(self.xyz)
        self.tree = cKDTree(self.xyz)
        self.mean = self.xyz.mean(0)
        centered = self.xyz - self.mean
        self.covariance = centered.T @ centered / max(len(self.xyz) - 1, 1)
        self.trace = max(float(np.trace(self.covariance)), 1e-12)
        self.min_z = float(self.xyz[:, 2].min())
        self.span_z = float(np.ptp(self.xyz[:, 2]))
        self._bounds_cache = {}

    def nearest_grid(self, xyz, frames, offsets):
        a, b = np.meshgrid(offsets, offsets)
        probes = xyz[:, None, :] + a.ravel()[None, :, None] * frames[:, 1, None, :]
        probes += b.ravel()[None, :, None] * frames[:, 2, None, :]
        distances = self.tree.query(probes.reshape(-1, 3), workers=1, distance_upper_bound=50.)[0]
        # Saturation is a numerical observation feature, never endpoint clipping.
        return (np.minimum(distances, 50.) / 50.).reshape(len(xyz), len(offsets)**2)

    def moment13(self, xyz, frames, radius):
        rows = []
        for point, frame in zip(xyz, frames):
            ids = self.tree.query_ball_point(point, radius, workers=1)
            if not ids:
                rows.append(np.zeros(13))
                continue
            local = (self.xyz[ids] - point) @ frame.T / radius
            mean = local.mean(0)
            centered = local - mean
            covariance = centered.T @ centered / len(local)
            values = np.maximum(np.linalg.eigvalsh(covariance), 0.)
            rows.append(np.r_[np.log1p(len(local))/20., mean, covariance[UPPER], values])
        return np.asarray(rows, np.float64).reshape(len(xyz), 13)

    def workpiece22(self, xyz, frames):
        rows = []
        for point, frame in zip(xyz, frames):
            relative = (point - self.mean) @ frame.T
            covariance = frame @ self.covariance @ frame.T
            scale = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
            # Bounded temporary memory; all observed vertices participate.
            key = frame.tobytes()
            if key in self._bounds_cache:
                low, high = self._bounds_cache[key]
            else:
                low, high = np.full(3, np.inf), np.full(3, -np.inf)
                for begin in range(0, len(self.xyz), 8192):
                    values = (self.xyz[begin:begin+8192] - self.mean) @ frame.T
                    low = np.minimum(low, values.min(0))
                    high = np.maximum(high, values.max(0))
                if len(self._bounds_cache) < 32:
                    self._bounds_cache[key] = (low.copy(), high.copy())
            bounds = np.stack((low-relative, high-relative), axis=1)
            height_context = (point[2]-self.min_z)/self.span_z if self.span_z > 0. else 0.
            rows.append(np.r_[relative/1000., relative/scale,
                (covariance/self.trace)[UPPER], bounds.ravel()/1000., height_context, frame[:, 2]])
        return np.asarray(rows, np.float64).reshape(len(xyz), 22)

    def reference32(self, xyz, frames):
        distance = self.nearest_grid(xyz, frames, NODE_OFFSETS_MM)
        values = []
        for point in xyz:
            ids = self.tree.query_ball_point(point, 20., workers=1)
            if not ids:
                values.append(np.zeros(7))
                continue
            delta = self.xyz[ids] - point
            centered = delta - delta.mean(0)
            covariance = centered.T @ centered / len(delta)
            eigen = np.maximum(np.linalg.eigvalsh(covariance), 0.) / 20.**2
            r = np.linalg.norm(delta, axis=1) / 20.
            values.append(np.r_[np.log1p(len(delta))/20., eigen, r.mean(), np.median(r), r.max()])
        return np.c_[distance, np.asarray(values, np.float64).reshape(len(xyz), 7)]


def _curvature_context(primitive):
    if isinstance(primitive, ArcPrimitive):
        return math.copysign(1., primitive.arc.sweep_rad) / primitive.arc.radius_mm
    if isinstance(primitive, PolylinePrimitive):
        tangents = _unit_rows(np.diff(primitive.knots_mm, axis=0))
        left, right = tangents[:-1], tangents[1:]
        if primitive.is_closed:
            left = np.r_[left, tangents[-1:]]
            right = np.r_[right, tangents[:1]]
        turns = np.arccos(np.clip((left * right).sum(1), -1., 1.))
        return float(turns.sum()) / primitive.length_mm
    return 0.


class TypedPointFeatureBlocks:
    """Execute all three observed blocks; complete query density pool prepared."""
    def __init__(self, observation, path, density_chunk_rows=1024):
        if not isinstance(observation, PointObservation) or not isinstance(path, RuntimeQueryPath):
            raise ProviderError('Bound observation and true query path required')
        if observation.binding != path.binding:
            raise ProviderError('Observation and primitive provider binding differ')
        if isinstance(density_chunk_rows, bool) or not isinstance(density_chunk_rows, int) or density_chunk_rows < 1:
            raise ProviderError('Positive full-density chunk size required')
        self.observation, self.path = observation, path
        self.calls = {'local97': 0, 'node255': 0, 'proximity11': 0}
        self.node = None
        self.geometry_pool = _sealed(np.unique(np.r_[0., path.primitive.length_mm,
            path.primitive.knot_positions_mm if isinstance(path.primitive, PolylinePrimitive) else np.empty(0)]))
        counts = np.empty(len(path.q), np.int64)
        for begin in range(0, len(path.q), density_chunk_rows):
            ids = np.arange(begin, min(begin+density_chunk_rows, len(path.q)))
            _, _, xyz = path.query_batch(ids)
            counts[ids] = observation.tree.query_ball_point(xyz, 5., workers=1, return_length=True)
        rates = np.empty(len(counts), np.float64)
        rates[0] = abs(float(counts[1]-counts[0])) / (path.q[1]-path.q[0])
        rates[-1] = abs(float(counts[-1]-counts[-2])) / (path.q[-1]-path.q[-2])
        rates[1:-1] = np.abs(counts[2:]-counts[:-2]) / (path.q[2:]-path.q[:-2])
        left = np.r_[rates[0], rates[:-1]]
        right = np.r_[rates[1:], rates[-1]]
        maxima = (rates > 0.) & (rates >= left) & (rates >= right)
        self.density_counts, self.density_rates = _sealed(counts), _sealed(rates)
        self.observation_pool = _sealed(path.q[maxima])
        self.fingerprint = sha256(canonical_json(dict(version=VERSION,
            observation_sha256=observation.binding.observation_sha256,
            actual_xyz_array_sha256=observation.xyz_sha256, query_path_fingerprint=path.fingerprint,
            geometry_pool_sha256=array_sha(self.geometry_pool),
            density_counts_sha256=array_sha(self.density_counts),
            density_rates_sha256=array_sha(self.density_rates),
            observation_pool_sha256=array_sha(self.observation_pool))).encode()).hexdigest()

    def _validated(self, indices, q, xyz):
        i, actual_q, actual_xyz = self.path.query_batch(indices)
        if not np.array_equal(actual_q, q) or not np.array_equal(actual_xyz, xyz):
            raise ProviderError('Callback q or XYZ differs from same complete primitive')
        return i, actual_q, actual_xyz, sampling_frames(self.path.primitive, actual_q)

    def local97(self, indices, q, xyz):
        _, _, xyz, frames = self._validated(indices, q, xyz)
        self.calls['local97'] += 1
        blocks = [self.observation.nearest_grid(xyz, frames, OFFSETS_MM)]
        blocks.extend(self.observation.moment13(xyz, frames, radius) for radius in LOCAL_RADII_MM)
        blocks.append(self.observation.workpiece22(xyz, frames))
        return np.column_stack(blocks).astype(np.float32)

    def node255(self, indices, q, xyz):
        self._validated(indices, q, xyz)
        self.calls['node255'] += 1
        if self.node is None:
            primitive = self.path.primitive
            positions = REFERENCE_FRACTIONS * primitive.length_mm
            reference_xyz = primitive.point_at(positions)
            reference_frames = sampling_frames(primitive, positions)
            descriptor = self.observation.reference32(reference_xyz, reference_frames).ravel()
            midpoint = np.asarray([primitive.length_mm/2.])
            mid_xyz = primitive.point_at(midpoint)
            mid_frame = sampling_frames(primitive, midpoint)
            kind = [isinstance(primitive, cls) for cls in (LinePrimitive, ArcPrimitive, PolylinePrimitive)]
            geometry = np.r_[primitive.length_mm/1000., mid_frame[0, 0], _curvature_context(primitive),
                kind, primitive.is_closed, self.path.weld_type=='horizontal', self.path.weld_type=='vertical']
            context = self.observation.workpiece22(mid_xyz, mid_frame)[0, :20]
            result = np.r_[descriptor, geometry, context].astype(np.float32)
            if result.shape != (255,) or not np.isfinite(result).all():
                raise ProviderError('Invalid actual node255 descriptor')
            self.node = _sealed(result)
        return self.node

    def proximity11(self, indices, q, xyz):
        self._validated(indices, q, xyz)
        self.calls['proximity11'] += 1
        columns, counts = [], []
        for pool in (self.geometry_pool, self.observation_pool):
            insertion = np.searchsorted(pool, q)
            left_exists, right_exists = insertion > 0, insertion < len(pool)
            left, right = np.full(len(q), -1.), np.full(len(q), 1.)
            if len(pool):
                left[left_exists] = np.maximum(pool[insertion[left_exists]-1]-q[left_exists], -128.)/128.
                right[right_exists] = np.minimum(pool[insertion[right_exists]]-q[right_exists], 128.)/128.
            columns.extend((left, right, left_exists.astype(float), right_exists.astype(float)))
            count = np.searchsorted(pool, q+16., side='right') - np.searchsorted(pool, q-16.)
            counts.append(np.minimum(count, 16)/16.)
        return np.column_stack(columns + counts + [q/self.path.primitive.length_mm]).astype(np.float32)

    def registered_callbacks(self):
        return FeatureBlockProviders(self.observation.binding.observation_sha256, self.path.fingerprint,
            VERSION, self.local97, self.node255, self.proximity11)

    def provider425(self, chunk_rows=1024):
        return Chunked425Provider(self.path, self.registered_callbacks(), chunk_rows=chunk_rows)

    def receipt(self):
        return dict(version=VERSION, fingerprint=self.fingerprint, observed_vertices=len(self.observation.xyz),
            observation_kind=self.observation.binding.kind,
            observation_sha256=self.observation.binding.observation_sha256,
            actual_xyz_array_sha256=self.observation.xyz_sha256, query_path_fingerprint=self.path.fingerprint,
            geometry=self.path.primitive.to_parameters(), queries=len(self.path.q),
            geometry_pool_mm=self.geometry_pool.tolist(), observation_density_pool_mm=self.observation_pool.tolist(),
            density_counts_sha256=array_sha(self.density_counts), density_rates_sha256=array_sha(self.density_rates),
            callbacks_actually_called=self.calls.copy(), complete_query_density_evaluated=True,
            local97_recipe='49_PCD_distances+13_R5_moments+13_R20_moments+22_workpiece_context',
            node255_recipe='7_true_arc_length_reference_positions_x32+11_geometry+20_workpiece_context',
            proximity11_recipe='geometry_end_corner_and_observed_density_extrema_2x4+2_counts+actual_q_over_L',
            sampling_axes_are_material_normals=False, whole_height_context_is_plate_height=False,
            pools_are_weld_events=False, no_candidate_query_or_child_pruning=True,
            requires_new_FIT_only_normalizer=True, compatible_with_175_mean_std_or_weights=False,
            actual_runtime_PCD_replay_verified=False, actual_raw7_head='UNTRAINED',
            actual_model_precision_claim=False, full_contract_pass=False)


def trace_observed_boundary_paths(vertex_xyz_mm, boundary_edges, raw_representative_ids):
    """Every observed boundary edge in one maximal nonbranching path or loop.

    No branch continuation is guessed and no material/weld identity is assigned.
    Original rows, edges, corners and all short spans remain in the ledger.
    """
    xyz = np.asarray(vertex_xyz_mm, np.float64)
    edges = np.asarray(boundary_edges)
    raw_ids = np.asarray(raw_representative_ids)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all()
            or edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in 'iu'
            or raw_ids.shape != (len(xyz),) or raw_ids.dtype.kind not in 'iu'
            or np.any(raw_ids < 0) or len(set(raw_ids.tolist())) != len(raw_ids)):
        raise ProviderError('Finite raw vertices, integer edges and unique original row IDs required')
    if len(edges) and (edges.min() < 0 or edges.max() >= len(xyz) or np.any(edges[:, 0] == edges[:, 1])):
        raise ProviderError('Observed boundary edge IDs invalid')
    keys = [tuple(sorted(map(int, pair))) for pair in edges]
    if len(set(keys)) != len(keys):
        raise ProviderError('Repeated physical boundary edge requires explicit multigraph semantics')
    adjacency = {}
    for i, (a, b) in enumerate(keys):
        adjacency.setdefault(a, []).append((b, i))
        adjacency.setdefault(b, []).append((a, i))
    for values in adjacency.values():
        values.sort()
    used, rows = set(), []
    def trace(start, neighbour, edge_id):
        vertices, source_edges = [start], []
        current, target, identity = start, neighbour, edge_id
        while True:
            if identity in used:
                raise ProviderError('Boundary traversal tried to reuse an observed edge')
            used.add(identity)
            source_edges.append(identity)
            vertices.append(target)
            if target == start or len(adjacency[target]) != 2:
                break
            candidates = [(v, e) for v, e in adjacency[target] if e not in used]
            if not candidates:
                break
            current, (target, identity) = target, candidates[0]
        points = xyz[vertices]
        closed = vertices[0] == vertices[-1]
        error, geometry = '', None
        try:
            primitive = PolylinePrimitive(points, closed)
            geometry = primitive.to_parameters()
        except (ValueError, OverflowError) as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        rows.append(dict(path_ordinal=len(rows), original_boundary_edge_ordinals=source_edges,
            original_vertex_ordinals=vertices, original_raw_row_ids=raw_ids[vertices].tolist(),
            original_points_mm=points.tolist(), is_closed=closed, geometry=geometry,
            status='OBSERVED_BOUNDARY_PRIMITIVE_UNKNOWN_WELD' if geometry else 'UNREPRESENTABLE_BOUNDARY_RETAINED',
            geometry_error=error, branch_at_start=len(adjacency[start]) > 2,
            branch_at_end=len(adjacency[vertices[-1]]) > 2,
            no_branch_connection_guessed=True, material_or_weld_identity_verified=False))
    for start in sorted(v for v in adjacency if len(adjacency[v]) != 2):
        for neighbour, identity in adjacency[start]:
            if identity not in used:
                trace(start, neighbour, identity)
    for identity, (a, b) in enumerate(keys):
        if identity not in used:
            trace(a, b, identity)
    if used != set(range(len(edges))):
        raise ProviderError('Incomplete observed boundary traversal')
    return dict(paths=rows, observed_boundary_edges=len(edges), accounted_boundary_edges=len(used),
        closed_paths=sum(r['is_closed'] for r in rows),
        branch_vertices=[v for v in sorted(adjacency) if len(adjacency[v]) > 2],
        unrepresentable_paths=sum(r['geometry'] is None for r in rows),
        every_original_edge_retained_exactly_once=True, no_short_edge_or_corner_removed=True,
        no_GT_STEP_material_or_weld_truth_used=True, full_contract_pass=False)
