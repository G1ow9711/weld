"""Circle fits to unbranched observed graph chains, not weld decisions."""
import numpy as np
from scipy.spatial import cKDTree
from .contract_curve_geometry import CircularArc


def prune_and_trace(vertices, edges):
    """Drop edges whose open diameter disk contains another observed vertex."""
    if not len(edges):
        return [], np.empty((0, 2), np.int64)
    midpoint = vertices[edges].mean(axis=1)
    half_length = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1) / 2
    nearest = cKDTree(vertices).query(midpoint, workers=2)[0]
    kept = edges[nearest >= half_length * (1 - 1e-10)]
    adjacency = [[] for _ in vertices]
    for ei, (a, b) in enumerate(kept):
        adjacency[a].append((b, ei)); adjacency[b].append((a, ei))
    used = np.zeros(len(kept), bool); chains = []

    def trace(start, neighbor, edge):
        chain = [start]; current = neighbor; used[edge] = True
        while True:
            chain.append(current)
            if current == start or len(adjacency[current]) != 2:
                break
            remaining = [(n, e) for n, e in adjacency[current] if not used[e]]
            if not remaining:
                break
            current, edge = remaining[0]; used[edge] = True
        chains.append(np.asarray(chain, np.int64))

    for start, neighbors in enumerate(adjacency):
        if len(neighbors) == 2:
            continue
        for neighbor, edge in neighbors:
            if not used[edge]:
                trace(start, neighbor, edge)
    for ei, (a, b) in enumerate(kept):
        if not used[ei]:
            trace(a, b, ei)
    assert used.all()
    return chains, kept


def fit_chain_arc(uv, chain, origin, basis):
    closed = chain[0] == chain[-1]
    points = uv[chain[:-1] if closed else chain]
    if len(points) < 5:
        return None
    mean = points.mean(0); local = points - mean
    matrix = np.column_stack((2 * local, np.ones(len(local))))
    solution, _, rank, _ = np.linalg.lstsq(matrix, np.sum(local * local, axis=1), rcond=1e-10)
    if rank < 3:
        return None
    center = mean + solution[:2]
    radial = np.linalg.norm(points - center, axis=1)
    radius = float(radial.mean())
    ordered = uv[chain] - center
    angles = np.unwrap(np.arctan2(ordered[:, 1], ordered[:, 0]))
    increments = np.diff(angles); sweep = float(angles[-1] - angles[0])
    if not np.isfinite(radius) or radius <= 0 or abs(sweep) < 1e-10:
        return None
    if np.any(increments * np.sign(sweep) <= 0):
        return None
    if closed:
        if not np.isclose(abs(sweep), 2 * np.pi, atol=1e-10, rtol=0):
            return None
        sweep = float(np.sign(sweep) * 2 * np.pi)
    elif abs(sweep) >= 2 * np.pi:
        return None
    start = ordered[0] / np.linalg.norm(ordered[0])
    arc = CircularArc(origin + basis @ center, radius, np.cross(basis[:, 0], basis[:, 1]), basis @ start, sweep, closed)
    return dict(arc_parameters=arc.to_parameters(), support_vertices=len(points),
                radial_error_max_mm=float(abs(radial-radius).max()),
                radial_error_rms_mm=float(np.sqrt(np.mean((radial-radius)**2))),
                observed_chain_length_mm=float(np.linalg.norm(np.diff(uv[chain], axis=0), axis=1).sum()))
