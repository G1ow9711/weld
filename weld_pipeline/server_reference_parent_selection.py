"""Input-only, nullable whole-parent selection with learned soft competition.

The graph is a bounded nearest-centre search, not a physical weld-identity test.
Scores are reference-policy utility estimates; they are not material occupancy
probabilities. No annotation, owner, or source identity is accepted at runtime.
"""
import collections
import heapq
import numpy as np
from scipy.spatial import cKDTree
from .server_parent_geometry_sampling import parent_geometry, endpoint_correspondence


def build_competition_graph(parents, metadata, parent_children, neighbors=16,
                            max_edges=4_000_000):
    """29 symmetric features; full distance ties, no radius-based hard merge."""
    n = len(parents)
    if n < 2:
        return np.empty((0, 2), np.int32), np.empty((0, 29), np.float32)
    clouds, ends, centres, attrs, faces, profiles = [], [], [], [], [], []
    for p, ids in zip(parents, parent_children):
        segments = p['segments']
        length = np.asarray([s['length_mm'] for s in segments])
        total = length.sum()
        sample, endpoints, centre = parent_geometry(p)
        clouds.append(sample); ends.append(endpoints); centres.append(centre)
        attrs.append([p['child_count'], total, float(p['is_closed']),
                      p['gaps'].count('SHARED'), p['gaps'].count('POSITIVE_GAP'),
                      sum(s['length_mm'] for s in segments if s['kind'] == 'line') / total,
                      sum(s['length_mm'] for s in segments if s['weld_type'] == 'horizontal') / total])
        faces.append({tuple(sorted(pair)) for i in ids for pair in metadata[i]['face_pairs']
                      if len(pair) == 2 and min(pair) >= 0})
        profiles.append({s['profile'] for s in segments})
    centres = np.asarray(centres); attrs = np.asarray(attrs)
    tree = cKDTree(centres)
    distances, _ = tree.query(centres, k=min(neighbors + 1, n))
    edges = set()
    for a, distance in enumerate(distances):
        for b in tree.query_ball_point(centres[a], float(np.max(distance)) + 1e-7):
            if a != b:
                edges.add(tuple(sorted((a, b))))
        if len(edges) > max_edges:
            raise ValueError('Competition graph exceeds declared edge capacity')
    pairs = np.asarray(sorted(edges), np.int32).reshape(-1, 2)
    features = []
    for a, b in pairs:
        distance = np.linalg.norm(clouds[a][:, None, :] - clouds[b][None, :, :], axis=2)
        quantiles = np.asarray([np.quantile(distance.min(1), [0, .25, .5, .75, 1]),
                                np.quantile(distance.min(0), [0, .25, .5, .75, 1])])
        symmetric = np.r_[quantiles.min(0), quantiles.max(0)]
        equal = len(ends[a]) == len(ends[b])
        ordered = endpoint_correspondence(ends[a], ends[b], parents[a]['is_closed'] or
                                          parents[b]['is_closed']) if equal else 0.
        features.append(np.r_[np.log1p(np.linalg.norm(centres[a] - centres[b])),
                               np.log1p(symmetric), np.log1p(abs(attrs[a] - attrs[b])),
                               np.log1p(np.minimum(attrs[a], attrs[b])), float(equal),
                               np.log1p(ordered), len(faces[a] & faces[b]) / max(1, len(faces[a] | faces[b])),
                               len(profiles[a] & profiles[b]) / max(1, len(profiles[a] | profiles[b]))])
    x = np.round(np.asarray(features, np.float64), 6).astype(np.float32)
    if x.shape != (len(pairs), 29) or not np.isfinite(x).all():
        raise ValueError('Invalid competition feature matrix')
    return pairs, x


def greedy_nullable_set(unary, pairs, penalties, parent_children):
    """Greedy positive marginal utility with exact shared-child exclusion.

    Soft pair penalties decrease the gain of later choices. Different nearby
    curves remain selectable; only an identical input child cannot be reused.
    Empty input/all-nonpositive utility produces the empty set. This is not an
    exact global optimizer, and row order breaks exactly equal utility ties.
    """
    unary = np.asarray(unary, np.float64)
    pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
    penalties = np.asarray(penalties, np.float64)
    n = len(unary)
    if len(parent_children) != n or penalties.shape != (len(pairs),):
        raise ValueError('Selection shapes disagree')
    if not np.isfinite(unary).all() or not np.isfinite(penalties).all() or (penalties < 0).any():
        raise ValueError('Finite unary and nonnegative penalties required')
    if pairs.size and ((pairs < 0).any() or (pairs >= n).any() or (pairs[:, 0] >= pairs[:, 1]).any()):
        raise ValueError('Unique unordered candidate-pair indices required')
    if len({tuple(p) for p in pairs}) != len(pairs):
        raise ValueError('Repeated competition edge')
    adjacency = [[] for _ in range(n)]
    for (a, b), cost in zip(pairs, penalties):
        adjacency[a].append((b, cost)); adjacency[b].append((a, cost))
    owners = collections.defaultdict(list)
    available = np.ones(n, bool)
    for p, children in enumerate(parent_children):
        if len(set(children)) != len(children):
            available[p] = False
        for child in set(children):
            owners[child].append(p)
    gains = unary.copy(); version = np.zeros(n, np.int64)
    heap = [(-gains[i], i, 0) for i in range(n) if available[i] and gains[i] > 0]
    heapq.heapify(heap); selected = []
    while heap:
        negative, index, stamp = heapq.heappop(heap)
        if not available[index] or version[index] != stamp:
            continue
        if -negative <= 0:
            break
        selected.append(index); available[index] = False
        for child in parent_children[index]:
            available[owners[child]] = False
        for neighbor, penalty in adjacency[index]:
            if available[neighbor]:
                gains[neighbor] -= penalty; version[neighbor] += 1
                if gains[neighbor] > 0:
                    heapq.heappush(heap, (-gains[neighbor], neighbor, version[neighbor]))
    return selected


def set_model_utility(unary, pairs, penalties, selected):
    selected = np.asarray(selected, np.int64)
    mask = np.zeros(len(unary), bool); mask[selected] = True
    pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
    active = mask[pairs[:, 0]] & mask[pairs[:, 1]]
    return float(np.asarray(unary)[selected].sum() - np.asarray(penalties)[active].sum())
