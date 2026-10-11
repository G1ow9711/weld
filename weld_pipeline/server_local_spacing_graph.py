"""Observed-point candidate graph; edges do not certify boundary or contact."""
import numpy as np
from scipy.spatial import cKDTree


def local_spacing_graph(points):
    """Exact-unique points, edges <= twice BOTH endpoints' nearest spacing.

    No fixed millimetre radius, reference labels, or synthetic vertices. Isolated
    sparse pairs can still connect: absence of samples is not absence of material.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or not np.isfinite(points).all():
        raise ValueError('Expected finite point matrix')
    vertices = np.unique(points, axis=0)
    if len(vertices) < 2:
        return vertices, np.empty((0, 2), np.int64), np.zeros(len(vertices))
    tree = cKDTree(vertices)
    spacing = tree.query(vertices, k=2, workers=2)[0][:, 1]
    edges = []
    for start in range(0, len(vertices), 1024):
        stop = min(start + 1024, len(vertices))
        # Relative epsilon only accommodates floating point boundary comparisons.
        neighbors = tree.query_ball_point(vertices[start:stop], 2 * spacing[start:stop] * (1 + 1e-12), workers=2)
        for i, neighbors_i in enumerate(neighbors, start):
            js = np.asarray(neighbors_i, np.int64)
            js = js[js > i]
            distances = np.linalg.norm(vertices[js] - vertices[i], axis=1)
            js = js[distances <= 2 * np.minimum(spacing[i], spacing[js]) * (1 + 1e-12)]
            if len(js):
                edges.append(np.column_stack((np.full(len(js), i, np.int64), js)))
    return vertices, np.concatenate(edges) if edges else np.empty((0, 2), np.int64), spacing
