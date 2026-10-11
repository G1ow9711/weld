"""Offline reference-policy unary utility and pairwise matching loss."""
import numpy as np
from .server_offline_parent_set_reference import evaluate_reference_subset


def reference_pair_targets(table, pairs):
    counts = np.asarray(table['child_counts'], np.int64)
    tp = np.asarray(table['tp'], np.int64)
    best = tp.max(axis=1) if tp.shape[1] else np.zeros(len(counts), np.int64)
    unary = (2 * best - counts) / counts
    penalties = np.zeros(len(pairs), np.float64)
    for index, (a, b) in enumerate(pairs):
        # Without a shared compatible reference, both individual assignments
        # remain simultaneously available. Otherwise retain original D3 costs.
        if np.any((tp[a] > 0) & (tp[b] > 0)):
            actual = evaluate_reference_subset(table, [int(a), int(b)])['TP']
            penalty = 2 * (int(best[a] + best[b]) - actual)
            if not 0 <= penalty <= 2 * min(counts[a], counts[b]):
                raise ValueError('Pair reference-loss bound violated')
            penalties[index] = penalty / (2 * min(counts[a], counts[b]))
    return unary.astype(np.float32), penalties.astype(np.float32)
