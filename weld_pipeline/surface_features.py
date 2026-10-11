"""Point-only distances restricted to the two contact surfaces of a sequence."""
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.typed_sequences import OFFSETS, DEPTHS

FEATURE_VERSION = 'typed_contact_surface_v1'


def append_surface_features(xyz, sequences):
    cache = {}
    stats = {'plane_trees': 0, 'empty_planes': 0, 'query_points': 0}

    def restricted_distance(q, axis, offsets, cross):
        axis = np.asarray(axis, float)
        axis /= np.linalg.norm(axis)
        offset = float(q[0] @ axis)
        if axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
            offset = -offset
        # Quantization only indexes repeated exactly constructed planes. The
        # actual double precision axis and offset are used for point selection.
        key = tuple(np.round(axis, 9)) + (round(offset, 4),)
        if key not in cache:
            mask = np.abs(xyz @ axis - offset) < .15
            p = xyz[mask]
            cache[key] = cKDTree(p) if len(p) else None
            stats['plane_trees'] += 1
            stats['empty_planes'] += int(not len(p))
        queries = (q[:, None] + offsets[None, :, None]*cross).reshape(-1, 3)
        tree = cache[key]
        distance = tree.query(queries, workers=2, distance_upper_bound=50.)[0] if tree is not None else np.full(len(queries), 50.)
        stats['query_points'] += len(queries)
        return np.minimum(distance, 50.).reshape(len(q), -1)/50.

    output = []
    for seq in sequences:
        base = seq['features']
        if base.shape[1] != 70:
            raise ValueError('Only append to original 70-channel point features')
        q = seq['xyz'].astype(float)
        n1, n2 = base[len(base)//2, 60:63].astype(float), base[len(base)//2, 63:66].astype(float)
        first = restricted_distance(q, n1, DEPTHS if seq['type'] == 'horizontal' else OFFSETS, n2)
        second = restricted_distance(q, n2, OFFSETS, n1)
        output.append({**seq, 'features': np.column_stack([base, first, second]).astype(np.float32)})
    return output, stats
