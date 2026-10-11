"""Reuse a worker pool for exact contact-plane feature queries."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.pooled_shared_mixed_sequences import make_fast_sequences
from weld_pipeline.indexed_surface_features import PlaneIndex
from weld_pipeline.typed_sequences import OFFSETS, DEPTHS
from weld_pipeline.shared_curved_contacts import append_curved_contacts


def append_fast_surface_features(xyz, sequences):
    cache, jobs = {}, []
    index = PlaneIndex(xyz)
    stats = {'plane_trees': 0, 'empty_planes': 0, 'query_points': 0}
    def prepare(q, axis, offsets, cross):
        # Keep the original normalization order, including its effects on n1/n2.
        axis = np.asarray(axis, float)
        axis /= np.linalg.norm(axis)
        offset = float(q[0] @ axis)
        if axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
            offset = -offset
        key = tuple(np.round(axis, 9)) + (round(offset, 4),)
        if key not in cache:
            points = index.select(axis, offset)
            cache[key] = cKDTree(points) if len(points) else None
            stats['plane_trees'] += 1
            stats['empty_planes'] += int(not len(points))
        # The first query must retain n2 before the second query normalizes it.
        jobs.append((q, offsets, cross.copy(), cache[key]))
        stats['query_points'] += len(q) * len(offsets)
    for seq in sequences:
        base = seq['features']
        if base.shape[1] != 70:
            raise ValueError('Expected original 70-channel features')
        q = seq['xyz'].astype(float)
        n1, n2 = base[len(base)//2, 60:63].astype(float), base[len(base)//2, 63:66].astype(float)
        prepare(q, n1, DEPTHS if seq['type'] == 'horizontal' else OFFSETS, n2)
        prepare(q, n2, OFFSETS, n1)
    def query(job):
        q, offsets, cross, tree = job
        queries = (q[:, None] + offsets[None, :, None] * cross).reshape(-1, 3)
        distance = tree.query(queries, workers=1, distance_upper_bound=50.)[0] if tree is not None else np.full(len(queries), 50.)
        return np.minimum(distance, 50.).reshape(len(q), -1) / 50.
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(query, jobs))
    output = [{**seq, 'features': np.column_stack([seq['features'], values[2*i], values[2*i+1]]).astype(np.float32)}
              for i, seq in enumerate(sequences)]
    stats['projection_indexes'] = len(index.indexes)
    stats['points_examined_for_plane_masks'] = index.examined_points
    return output, stats


def make_fast_mixed_sequences(xyz):
    straight, geometry = make_fast_sequences(xyz)
    straight, surface = append_fast_surface_features(xyz, straight)
    sequences, curves = append_curved_contacts(xyz, straight)
    return sequences, {'straight_geometry': geometry, 'surface': surface, 'curves': curves,
                       'optimization': 'reused_pool_for_global_and_plane_queries'}
