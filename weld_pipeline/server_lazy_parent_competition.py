"""Exact active subgraphs of the original nonnegative parent competition.

The neighbourhood is still defined over every predicted parent. Only edges
whose two ends can affect a selected runtime or TRAIN comparator set need
features. Negative unary nodes cannot become selectable after nonnegative
penalties. TRAIN comparator edges are requested separately, not omitted.
"""
import numpy as np
from scipy.spatial import cKDTree

from .server_predicted_parent_pool import parent_center


class LazyParentCompetition:
    def __init__(self, parents, priors, neighbors=16, max_active_pairs=4_000_000,
                 feature_chunk_size=2048):
        if min(neighbors, max_active_pairs, feature_chunk_size) < 1:
            raise ValueError('Positive explicit competition resources required')
        self.parents, self.priors = parents, priors
        self.centers = np.asarray([parent_center(p) for p in parents], float).reshape(-1, 3)
        self.tree = cKDTree(self.centers) if len(parents) else None
        self.neighbors, self.max_active_pairs = neighbors, max_active_pairs
        self.feature_chunk_size = feature_chunk_size
        self.counts = np.asarray([p['child_count'] for p in parents], np.int64)
        self.cache = {}

    def pairs_for(self, active):
        ids = np.asarray(active, np.int64)
        if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(self.parents)):
            raise ValueError('Unique valid active parent indices required')
        if len(ids) < 2:
            return np.empty((0, 2), np.int64)
        active_set = set(ids.tolist())
        distances, _ = self.tree.query(self.centers[ids], k=min(self.neighbors+1, len(self.parents)))
        distances = np.asarray(distances).reshape(len(ids), -1)
        pairs = set()
        for a, distance in zip(ids, distances):
            # Same full-input neighbour radius and numerical tie guard as
            # parent_competition_pairs; no subset-relative neighbour graph.
            for b in self.tree.query_ball_point(self.centers[a], float(distance.max())+1e-7):
                if a != b and b in active_set:
                    pairs.add(tuple(sorted((int(a), int(b)))))
            if len(pairs) > self.max_active_pairs:
                raise RuntimeError('Exact active competition exceeds declared resources; refuse truncation')
        return np.asarray(sorted(pairs), np.int64).reshape(-1, 2)

    def features(self, pairs):
        pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
        if (len(pairs) > self.max_active_pairs or np.any(pairs < 0)
                or np.any(pairs >= len(self.parents)) or np.any(pairs[:, 0] >= pairs[:, 1])
                or len(np.unique(pairs, axis=0)) != len(pairs)):
            raise ValueError('Unique valid unordered competition pairs within declared resources required')
        missing = [tuple(pair) for pair in pairs.tolist() if tuple(pair) not in self.cache]
        for start in range(0, len(missing), self.feature_chunk_size):
            block = missing[start:start+self.feature_chunk_size]
            indices = np.asarray(block, np.int64).reshape(-1, 2)
            feature, probability = self.priors.duplicate_probabilities(self.parents, indices)
            scale = 2*np.minimum(self.counts[indices[:, 0]], self.counts[indices[:, 1]])
            for pair, row, base in zip(block, feature, probability*scale):
                self.cache[pair] = (row, float(base))
        x = np.asarray([self.cache[tuple(pair)][0] for pair in pairs], np.float32).reshape(-1, 32)
        base = np.asarray([self.cache[tuple(pair)][1] for pair in pairs], float)
        return x, base

    def relation_arrays(self, active):
        pairs = self.pairs_for(active)
        return self.arrays_for_pairs(pairs)

    def arrays_for_pairs(self, pairs):
        pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
        features, base = self.features(pairs)
        scale = 2*np.minimum(self.counts[pairs[:, 0]], self.counts[pairs[:, 1]])
        return dict(pairs=pairs, pair_features=features, base_penalties=base, pair_scale=scale)

    def relations_for_selections(self, *selections):
        # Internal edges of each complete set are sufficient for its score.
        # Comparator nodes may have negative unary values, so they cannot be
        # inferred from the runtime positive-unary graph.
        blocks = [self.pairs_for(ids) for ids in selections]
        pairs = np.unique(np.concatenate(blocks), axis=0) if blocks else np.empty((0, 2), np.int64)
        return self.arrays_for_pairs(pairs)
