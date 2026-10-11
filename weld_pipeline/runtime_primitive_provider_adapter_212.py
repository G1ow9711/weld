"""Pure CPU current-observation proposal/query/425 provider adapter.

No point-file discovery, model inference, GT, STEP or cache loading occurs here.
Blocks must be produced by registered providers from the same current point
observation. Correct dimensions do not establish V/curve feature validity.
"""
from dataclasses import dataclass
from hashlib import sha256
import math
from types import MappingProxyType

import numpy as np

from weld_pipeline.contract_curve_geometry import CircularArc
from weld_pipeline.primitive_incidence_209 import (
    RuntimePrimitiveInput, RuntimeProvenance, parse_primitive, canonical_json, array_sha)

VERSION = 'current_observation_primitive_query_provider_212_v1'
FEATURE_VERSION = 'runtime97_node255_proximity11_physical5_actualq_phase57_212_v1'
PERIODS_MM = np.array([16., 32., 64., 128., 256., 512., 1024.])
PHYSICAL5_NAMES = ('q_over_1000', 'remaining_q_over_1000', 'x_over_10000', 'y_over_10000', 'z_over_10000')
PHASE57_NAMES = ('cell_q_mod64_over64',) + tuple(
    f'{function}_{axis}_period{int(period)}mm'
    for function in ('sin', 'cos') for axis in ('q', 'x', 'y', 'z') for period in PERIODS_MM)
FEATURE_BLOCK_SLICES = dict(local_point97=(0, 97), node255=(97, 352), proximity11=(352, 363),
                            physical5=(363, 368), phase57=(368, 425))


class ProviderError(ValueError):
    pass


def _sealed(a):
    a = np.ascontiguousarray(a)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _freeze_metadata(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_metadata(v) for key, v in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_metadata(v) for v in value)
    return value


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ProviderError('SHA256 lowercase hex required')
    return value


@dataclass(frozen=True)
class ObservationBinding:
    kind: str
    observation_sha256: str
    proposal_provider_version: str

    def __post_init__(self):
        if self.kind not in ('runtime_pcd', 'synthetic_fixture'):
            raise ProviderError('only current runtime PCD or explicit synthetic fixture allowed; no GT/STEP/cache/name')
        _sha(self.observation_sha256)
        if not isinstance(self.proposal_provider_version, str) or not self.proposal_provider_version:
            raise ProviderError('explicit proposal provider version required')


def query_grid(length_mm, step_mm, max_queries=1_000_001):
    length, step = float(length_mm), float(step_mm)
    if not math.isfinite(length) or length <= 0. or not math.isfinite(step) or step <= 0.:
        raise ProviderError('finite positive physical length/step required')
    if isinstance(max_queries, bool) or not isinstance(max_queries, int) or max_queries < 2:
        raise ProviderError('query resource limit >=2 required')
    intervals = length / step
    if not math.isfinite(intervals) or intervals > max_queries - 1:
        raise ProviderError('QUERY_RESOURCE_LIMIT: path not truncated; use a reviewed streaming-query provider')
    q = np.arange(int(math.floor(intervals)) + 1, dtype=np.float64) * step
    q = q[q < length]
    q = np.r_[q, length]
    if len(q) > max_queries or q[0] != 0. or np.any(np.diff(q) <= 0.):
        raise ProviderError('QUERY_PRECISION_OR_RESOURCE_FAILURE: no endpoint clamping/truncation')
    return _sealed(q)


class RuntimeQueryPath:
    def __setattr__(self, name, value):
        if getattr(self, '_locked', False):
            raise ProviderError('runtime geometry/query binding is immutable')
        object.__setattr__(self, name, value)

    def __init__(self, geometry_parameters, weld_type, binding, step_mm=2., max_queries=1_000_001):
        if not isinstance(binding, ObservationBinding) or weld_type not in ('horizontal', 'vertical'):
            raise ProviderError('observation binding and explicit weld type required')
        self.primitive = parse_primitive(geometry_parameters)
        self.weld_type, self.binding = weld_type, binding
        self.geometry_parameters = _freeze_metadata(self.primitive.to_parameters())
        self.q = query_grid(self.primitive.length_mm, step_mm, max_queries)
        self.geometry_sha256 = sha256(canonical_json(self.geometry_parameters).encode()).hexdigest()
        self.query_sha256 = array_sha(self.q)
        self.fingerprint = sha256(canonical_json(dict(version=VERSION, weld_type=weld_type,
            observation_kind=binding.kind, observation_sha256=binding.observation_sha256,
            proposal_provider_version=binding.proposal_provider_version,
            geometry_sha256=self.geometry_sha256, query_sha256=self.query_sha256)).encode()).hexdigest()
        self._locked = True

    def indices(self, value):
        i = np.asarray(value)
        if i.ndim != 1 or (i.size and i.dtype.kind not in 'iu'):
            raise ProviderError('one-dimensional integer query indices required')
        i = i.astype(np.int64, copy=False)
        if i.size and (i.min() < 0 or i.max() >= len(self.q)):
            raise ProviderError('query index outside runtime path')
        return i

    def query_batch(self, indices):
        i = self.indices(indices)
        q = self.q[i]
        xyz = self.primitive.point_at(q)
        return i, q, xyz


def adapt_current_proposal(proposal, binding, step_mm=2., max_queries=1_000_001):
    """Consume a current proposal's actual geometry, never cached XYZ/GT fits.

    Existing straight proposals have one line with two endpoint points. Existing
    arc_surface_candidates expose observed XY arc metadata plus contact height.
    Full signed-3D/closed/polyline providers can supply typed_geometry explicitly.
    This adapter does not enlarge the existing proposal generator's scope.
    """
    if 'typed_geometry' in proposal:
        geometry = proposal['typed_geometry']
    elif 'arc' in proposal:
        arc = proposal['arc']
        angle = float(arc['start_angle_rad'])
        geometry = dict(kind='arc', units='mm', arc_parameters=CircularArc(
            [*arc['center_xy_mm'], float(proposal['height_mm'])], arc['radius_mm'],
            [0., 0., 1.], [math.cos(angle), math.sin(angle), 0.], arc['sweep_rad'],
            bool(arc.get('is_closed', False))).to_parameters())
    else:
        segments = proposal.get('segments', ())
        if len(segments) != 1 or segments[0].get('kind') != 'line':
            raise ProviderError('one true line proposal or explicit typed_geometry required; curve is not a chord')
        points = np.asarray(segments[0]['points'], np.float64)
        if points.shape != (2, 3):
            raise ProviderError('true double precision line endpoints required')
        geometry = dict(kind='line', start_mm=points[0].tolist(), end_mm=points[1].tolist(), units='mm')
    return RuntimeQueryPath(geometry, proposal['type'], binding, step_mm, max_queries)


def physical5(q, xyz, length_mm):
    return np.c_[q / 1000., (length_mm - q) / 1000., xyz / 10000.].astype(np.float32)


def actual_q_phase57(q, xyz):
    coordinates = np.c_[q, xyz]
    angles = coordinates[:, :, None] * (2. * np.pi / PERIODS_MM)[None, None, :]
    return np.c_[np.mod(q, 64.) / 64., np.sin(angles).reshape(len(q), 28),
                 np.cos(angles).reshape(len(q), 28)].astype(np.float32)


@dataclass(frozen=True)
class FeatureBlockProviders:
    observation_sha256: str
    query_path_fingerprint: str
    version: str
    local_point97: object
    node255: object
    proximity11: object

    def __post_init__(self):
        _sha(self.observation_sha256)
        _sha(self.query_path_fingerprint)
        if not isinstance(self.version, str) or not self.version or not all(callable(v) for v in
                (self.local_point97, self.node255, self.proximity11)):
            raise ProviderError('all actual block callbacks97/255/11 and their version required; no zero-fill fallback')


class Chunked425Provider:
    """Callbacks get (indices,actual_q,actual_XYZ); no all-cloud/NN imports.

    node255 returns one path descriptor or query-local [K,255], explicitly.
    A complete request above max_materialized_rows raises a resource error; it
    never drops queries. iter_features supports full streamed requests.
    """
    def __init__(self, path, blocks, chunk_rows=1024, max_materialized_rows=8192):
        if not isinstance(path, RuntimeQueryPath) or not isinstance(blocks, FeatureBlockProviders):
            raise ProviderError('runtime path and complete bound blocks required')
        if blocks.observation_sha256 != path.binding.observation_sha256 or blocks.query_path_fingerprint != path.fingerprint:
            raise ProviderError('provider observation/query/complete geometry binding mismatch')
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (chunk_rows, max_materialized_rows)):
            raise ProviderError('positive chunk/resource sizes required')
        self.path, self.blocks = path, blocks
        self.chunk_rows, self.max_materialized_rows = chunk_rows, max_materialized_rows

    def _batch(self, indices):
        i, q, xyz = self.path.query_batch(indices)
        values = []
        for callback, count in [(self.blocks.local_point97, 97), (self.blocks.node255, 255), (self.blocks.proximity11, 11)]:
            block = np.asarray(callback(i.copy(), q.copy(), xyz.copy()))
            if count == 255 and block.shape == (255,):
                block = np.broadcast_to(block, (len(i), 255))
            if block.dtype != np.float32 or block.shape != (len(i), count) or not np.isfinite(block).all():
                raise ProviderError(f'actual finite float32[K,{count}] block required')
            values.append(block)
        result = np.column_stack(tuple(values) + (physical5(q, xyz, self.path.primitive.length_mm), actual_q_phase57(q, xyz)))
        if result.shape != (len(i), 425) or result.dtype != np.float32 or not np.isfinite(result).all():
            raise ProviderError('finite425 runtime feature assembly failed')
        return result

    def iter_features(self, indices):
        i = self.path.indices(indices)
        for start in range(0, len(i), self.chunk_rows):
            batch = i[start:start + self.chunk_rows]
            yield batch.copy(), self._batch(batch)

    def __call__(self, indices):
        i = self.path.indices(indices)
        if len(i) > self.max_materialized_rows:
            raise ProviderError('FEATURE_MATERIALIZATION_LIMIT: complete request rejected, no query truncation')
        result = np.empty((len(i), 425), np.float32)
        start = 0
        for _, values in self.iter_features(i):
            result[start:start + len(values)] = values
            start += len(values)
        return result


def bind_209_input(path_id, path, provider, raw7, raw_receipt):
    """Bind externally executed matching-scope head output, without inference."""
    expected = 'synthetic_fields' if path.binding.kind == 'synthetic_fixture' else 'actual_head_output'
    if (provider.path is not path or raw_receipt.get('output_kind') != expected or
            raw_receipt.get('observation_sha256') != path.binding.observation_sha256 or
            raw_receipt.get('query_path_fingerprint') != path.fingerprint or
            raw_receipt.get('executed_for_all_queries') is not True):
        raise ProviderError('actual matching observation/query head output required; skipped placeholder forbidden')
    scope = (path.weld_type, path.geometry_parameters['kind'], path.primitive.is_closed)
    if scope not in tuple(tuple(v) for v in raw_receipt.get('trained_scope', ())):
        raise ProviderError('head trained scope does not include this type/geometry/closure')
    provenance = RuntimeProvenance(path.binding.kind, path.binding.observation_sha256,
        path.binding.proposal_provider_version, provider.blocks.version,
        raw_receipt['head_version'], expected)
    return RuntimePrimitiveInput(path_id, path.weld_type, path.geometry_parameters, path.q,
                                 raw7, provider, provenance)
