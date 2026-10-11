"""Point-only cross-section context for vertical candidates."""
import numpy as np
from scipy.spatial import cKDTree

FEATURE_VERSION = 'typed_contact_wide_vertical_v1'
BASE_CHANNELS = 84
OFFSETS = np.array([-150., -100., -50., -25., 0., 25., 50., 100., 150.])
EXTRA_CHANNELS = len(OFFSETS) ** 2
CHANNELS = BASE_CHANNELS + EXTRA_CHANNELS
DISTANCE_SCALE_MM = 150.


def make_vertical_context(xyz, sequences, workers=4, block_points=4096):
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Finite nonempty XYZ in millimeters required')
    tree = cKDTree(xyz)
    transverse = np.array([(x, y) for y in OFFSETS for x in OFFSETS])
    result = []
    for seq in sequences:
        if seq['type'] != 'vertical':
            result.append(None)
            continue
        features = seq['features']
        q = np.asarray(seq['xyz'], dtype=float)
        n1 = np.asarray(features[:, 60:63], dtype=float)
        n2 = np.asarray(features[:, 63:66], dtype=float)
        if not (features[:, 66] > .5).all():
            raise ValueError('Candidate geometry and vertical feature disagree')
        if not (np.allclose(np.linalg.norm(n1, axis=1), 1., atol=1e-4) and
                np.allclose(np.linalg.norm(n2, axis=1), 1., atol=1e-4) and
                np.allclose((n1 * n2).sum(1), 0., atol=1e-4)):
            raise ValueError('Orthonormal transverse frame required')
        extra = np.empty((len(q), EXTRA_CHANNELS), np.float32)
        for start in range(0, len(q), block_points):
            stop = min(len(q), start + block_points)
            queries = (q[start:stop, None] + n1[start:stop, None] * transverse[None, :, :1] +
                       n2[start:stop, None] * transverse[None, :, 1:])
            distances = tree.query(queries.reshape(-1, 3), workers=workers)[0]
            extra[start:stop] = np.minimum(distances.reshape(stop - start, -1), DISTANCE_SCALE_MM) / DISTANCE_SCALE_MM
        result.append(extra)
    return result


class ContextFeatures:
    """Append immutable context at slice time, without copying entire H arrays."""
    def __init__(self, base, extra=None):
        if base.ndim != 2 or base.shape[1] != BASE_CHANNELS or base.dtype != np.float32:
            raise ValueError('Expected original float32 84-channel features')
        if extra is not None and (extra.shape != (len(base), EXTRA_CHANNELS) or extra.dtype != np.float32):
            raise ValueError('Invalid context array')
        self.base, self.extra = base, extra
        self.shape = (len(base), CHANNELS)
        self.dtype = np.dtype('float32')
        self.ndim = 2

    def __len__(self):
        return len(self.base)

    def __getitem__(self, key):
        rows, columns = key if isinstance(key, tuple) else (key, slice(None))
        if isinstance(columns, (int, np.integer)):
            column = int(columns) % CHANNELS
            if not -CHANNELS <= int(columns) < CHANNELS:
                raise IndexError('Feature column out of bounds')
            if column < BASE_CHANNELS:
                return self.base[rows, column]
        if isinstance(columns, slice):
            start, stop, step = columns.indices(CHANNELS)
            if step > 0 and stop <= BASE_CHANNELS:
                return self.base[rows, slice(start, stop, step)]
        base = self.base[rows]
        extra = self.extra[rows] if self.extra is not None else np.zeros(base.shape[:-1] + (EXTRA_CHANNELS,), np.float32)
        return np.concatenate([base, extra], axis=-1)[..., columns]

    def __array__(self, dtype=None):
        array = self[:]
        return array.astype(dtype, copy=False) if dtype is not None else array
