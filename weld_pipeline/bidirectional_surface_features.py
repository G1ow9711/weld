"""Inspect either side of the base plate without changing input coordinates."""
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.surface_features import append_surface_features
from weld_pipeline.typed_sequences import DEPTHS

FEATURE_VERSION = 'typed_contact_surface_v2'


def append_bidirectional_surface_features(xyz, sequences):
    if all(s['features'].shape[1] == 84 for s in sequences):
        enriched = sequences
        stats = {'reused_point_only_surface_cache': True}
    else:
        enriched, stats = append_surface_features(xyz, sequences)
    cache = {}
    output = []
    queries_count = 0
    for seq in enriched:
        features = seq['features'].copy()
        if seq['type'] == 'horizontal':
            q = seq['xyz'].astype(float)
            normal = features[len(features)//2, 60:63].astype(float)
            normal /= np.linalg.norm(normal)
            cross = features[len(features)//2, 63:66].astype(float)
            offset = float(q[0] @ normal)
            if normal[np.argmax(np.abs(normal))] < 0:
                normal, offset = -normal, -offset
            key = tuple(np.round(normal, 9))+(round(offset, 4),)
            if key not in cache:
                pts = xyz[np.abs(xyz @ normal - offset) < .15]
                cache[key] = cKDTree(pts) if len(pts) else None
            queries = (q[:, None] - DEPTHS[None, :, None]*cross).reshape(-1, 3)
            tree = cache[key]
            distance = tree.query(queries, workers=2, distance_upper_bound=50.)[0] if tree is not None else np.full(len(queries), 50.)
            opposite = np.minimum(distance, 50.).reshape(len(q), 7)/50.
            features[:, 70:77] = np.minimum(features[:, 70:77], opposite)
            queries_count += len(queries)
        output.append({**seq, 'features':features})
    return output, {**stats, 'opposite_side_plane_trees':len(cache), 'opposite_side_query_points':queries_count}
