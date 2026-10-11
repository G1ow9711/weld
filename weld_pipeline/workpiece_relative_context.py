"""PCD-only relative workpiece context. Whole height is a context, not plate height."""
import numpy as np

from weld_pipeline.signed_material_support import unit, frame_views as material_views

VERSION = 'workpiece_relative_context_v1'
FEATURE_COUNT = 22
COMBINED_FEATURE_COUNT = 188
GRAVITY_FEATURE_COUNT = 169
PARAMETERS = dict(length_scale_mm=1000., variance_floor_mm2=1e-12,
                  gravity_axis=[0., 0., 1.], coordinate_round_decimals=5,
                  equivalent_frames=16)
UPPER = np.triu_indices(3)


class WorkpieceRelativeContext:
    def __init__(self, xyz):
        self.xyz = np.asarray(xyz, float)
        if (self.xyz.ndim != 2 or self.xyz.shape[1] != 3 or len(self.xyz) == 0
                or not np.isfinite(self.xyz).all()):
            raise ValueError('Nonempty finite PCD XYZ required')
        self.mean = self.xyz.mean(0)
        self.centered = self.xyz - self.mean
        self.covariance = self.centered.T @ self.centered / max(len(self.xyz)-1, 1)
        self.trace = max(float(np.trace(self.covariance)), 1e-12)
        self.lower_z = float(self.xyz[:, 2].min())
        self.span_z = float(np.ptp(self.xyz[:, 2]))
        self.bounds_cache = {}

    def _bounds(self, axis):
        key = tuple(axis.tolist())
        if key not in self.bounds_cache:
            values = np.round(self.centered @ axis, 5)
            self.bounds_cache[key] = np.array([values.min(), values.max()])
        return self.bounds_cache[key]

    def describe(self, query, normals, tangent):
        queries = np.asarray(query, float)
        scalar = queries.ndim == 1
        queries = np.atleast_2d(queries)
        if queries.shape[1] != 3 or not np.isfinite(queries).all():
            raise ValueError('Finite XYZ query required')
        basis = np.stack([unit(tangent), unit(normals[0]), unit(normals[1])])
        relative = np.round((queries-self.mean) @ basis.T, 5)
        covariance = basis @ self.covariance @ basis.T
        scale = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
        bounds = np.stack([self._bounds(axis) for axis in basis])
        query_bounds = bounds[None, :, :] - relative[:, :, None]
        height = ((queries[:, 2]-self.lower_z)/self.span_z
                  if self.span_z > 1e-8 else np.zeros(len(queries)))
        constant = np.tile((covariance/self.trace)[UPPER], (len(queries), 1))
        gravity = np.tile(basis[:, 2], (len(queries), 1))
        result = np.concatenate((relative/1000., relative/scale,
            constant, query_bounds.reshape(-1, 6)/1000., height[:, None], gravity), 1).astype(np.float32)
        if result.shape != (len(queries), FEATURE_COUNT) or not np.isfinite(result).all():
            raise ValueError('Invalid relative workpiece context')
        return result[0] if scalar else result

    def describe_sequence(self, sequence, indices):
        indices = np.asarray(indices, int)
        features = np.asarray(sequence['features'], float)
        normals = features[indices, 60:66].reshape(-1, 2, 3)
        if len(indices) == 0:
            return np.empty((0, FEATURE_COUNT), np.float32)
        # Straight candidates have constant face normals; retain a general fallback.
        if np.array_equal(normals, np.broadcast_to(normals[0], normals.shape)):
            return self.describe(sequence['xyz'][indices], normals[0], sequence['direction'])
        return np.stack([self.describe(sequence['xyz'][i], n, sequence['direction'])
                         for i, n in zip(indices, normals)])


def transform_context(features, exchange=False, flip_a=False, flip_b=False, reverse=False):
    x = np.asarray(features, np.float32)
    if x.shape[-1] != FEATURE_COUNT:
        raise ValueError('Invalid relative context width')
    order = [0, 2, 1] if exchange else [0, 1, 2]
    signs = np.array([-1. if reverse else 1., -1. if flip_a else 1.,
                      -1. if flip_b else 1.], np.float32)
    result = x.copy()
    result[..., :3] = x[..., :3][..., order]*signs
    result[..., 3:6] = x[..., 3:6][..., order]*signs
    covariance = np.zeros((*x.shape[:-1], 3, 3), np.float32)
    covariance[..., UPPER[0], UPPER[1]] = x[..., 6:12]
    covariance[..., UPPER[1], UPPER[0]] = x[..., 6:12]
    covariance = covariance[..., order, :][..., :, order]*signs[:, None]*signs[None, :]
    result[..., 6:12] = covariance[..., UPPER[0], UPPER[1]]
    bounds = x[..., 12:18].reshape(*x.shape[:-1], 3, 2)[..., order, :].copy()
    for axis, sign in enumerate(signs):
        if sign < 0:
            bounds[..., axis, :] = -bounds[..., axis, ::-1]
    result[..., 12:18] = bounds.reshape(*x.shape[:-1], 6)
    result[..., 19:22] = x[..., 19:22][..., order]*signs
    return result


def frame_views(features, representation='whole'):
    """Same 16 frame combinations for old, gravity-only, and whole-context inputs."""
    x = np.asarray(features, np.float32)
    count = {'baseline': 166, 'gravity': GRAVITY_FEATURE_COUNT, 'whole': COMBINED_FEATURE_COUNT}[representation]
    if x.shape[-1] != count:
        raise ValueError('Invalid workpiece representation width')
    local = material_views(x[..., :166])
    views = []
    i = 0
    for exchange in (False, True):
        for flip_a, flip_b in ((False, False), (True, False), (False, True), (True, True)):
            for reverse in (False, True):
                if representation == 'baseline':
                    views.append(local[i].copy())
                elif representation == 'whole':
                    context = transform_context(x[..., 166:], exchange, flip_a, flip_b, reverse)
                    views.append(np.concatenate((local[i], context), -1))
                else:
                    order = [0, 2, 1] if exchange else [0, 1, 2]
                    signs = np.array([-1. if reverse else 1., -1. if flip_a else 1.,
                                      -1. if flip_b else 1.], np.float32)
                    gravity = x[..., 166:][..., order]*signs
                    views.append(np.concatenate((local[i], gravity), -1))
            i += 1
    return views


def predict_equivalent_views(model, features, representation='whole'):
    return np.mean([model.predict_proba(view) for view in frame_views(features, representation)], axis=0)
