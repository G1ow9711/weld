"""TRAIN-only dynamic oracle for complete-parent set responsibility.

Compatibility edges come from independent full-parent scoring, never proximity.
This module labels actions. Its GT counts, matches and allowed masks must never
be used as predictor features or be imported by label-free inference.
"""
from collections import deque
import numpy as np


class ParentSetOracle:
    """Incremental maximum bipartite matching for any selected prefix."""

    def __init__(self, compatible_targets, target_count):
        if not isinstance(target_count, int) or target_count < 0:
            raise ValueError('Target count must be a nonnegative integer')
        self.target_count = target_count
        self.edges = tuple(tuple(sorted(set(map(int, row)))) for row in compatible_targets)
        if any(g < 0 or g >= target_count for row in self.edges for g in row):
            raise ValueError('Compatibility target outside original denominator')
        self.reverse = [[] for _ in range(target_count)]
        for candidate, row in enumerate(self.edges):
            for g in row:
                self.reverse[g].append(candidate)
        self.selected = set()
        self.selection_order = []
        self.left_match = [-1] * len(self.edges)
        self.right_match = [-1] * target_count
        self.match_count = 0

    def _augment(self, candidate):
        queue = deque([candidate])
        seen_candidates = {candidate}
        parent_of_target = {}
        while queue:
            left = queue.popleft()
            for target in self.edges[left]:
                if target in parent_of_target:
                    continue
                parent_of_target[target] = left
                matched = self.right_match[target]
                if matched < 0:
                    while target >= 0:
                        left = parent_of_target[target]
                        previous = self.left_match[left]
                        self.left_match[left] = target
                        self.right_match[target] = left
                        target = previous
                    self.match_count += 1
                    return True
                if matched not in seen_candidates:
                    seen_candidates.add(matched)
                    queue.append(matched)
        return False

    def select(self, candidate):
        """Add one candidate, preserving maximum matching, even for bad prefixes."""
        if not isinstance(candidate, (int, np.integer)) or not 0 <= candidate < len(self.edges):
            raise ValueError('Candidate index out of range')
        candidate = int(candidate)
        if candidate in self.selected:
            raise ValueError('The same candidate cannot be selected twice')
        self.selected.add(candidate)
        self.selection_order.append(candidate)
        increased = self._augment(candidate)
        assert sum(g >= 0 for g in self.left_match) == self.match_count
        assert sum(p >= 0 for p in self.right_match) == self.match_count
        return increased

    def allowed_candidates(self):
        """Find every augmenting next action in one reverse alternating traversal."""
        queue = deque(g for g, candidate in enumerate(self.right_match) if candidate < 0)
        reachable = set(queue)
        allowed = set()
        while queue:
            target = queue.popleft()
            for candidate in self.reverse[target]:
                if candidate not in self.selected:
                    allowed.add(candidate)
                    continue
                matched_target = self.left_match[candidate]
                if matched_target >= 0 and matched_target not in reachable:
                    reachable.add(matched_target)
                    queue.append(matched_target)
        return tuple(sorted(allowed))

    def action_masks(self):
        """Return available actions and supervised legal actions; last index STOP."""
        available = np.ones(len(self.edges) + 1, dtype=bool)
        if self.selected:
            available[list(self.selected)] = False
        allowed = np.zeros_like(available)
        candidates = self.allowed_candidates()
        if candidates:
            allowed[list(candidates)] = True
        else:
            allowed[-1] = True
        assert np.any(allowed) and not np.any(allowed & ~available)
        return available, allowed


def set_action_loss(logits, available, allowed):
    """Negative log probability of the legal action set, with exact gradient."""
    logits = np.asarray(logits, dtype=np.float64)
    available = np.asarray(available, dtype=bool)
    allowed = np.asarray(allowed, dtype=bool)
    if logits.ndim != 1 or logits.shape != available.shape or allowed.shape != logits.shape:
        raise ValueError('Logits and masks must be equally sized vectors')
    if not np.isfinite(logits).all() or not np.any(allowed) or np.any(allowed & ~available):
        raise ValueError('Nonfinite scores or invalid legal-action mask')
    def distribution(mask):
        indices = np.flatnonzero(mask)
        values = logits[indices]
        peak = float(values.max())
        exponentials = np.exp(values - peak)
        normalizer = float(exponentials.sum())
        probabilities = np.zeros_like(logits)
        probabilities[indices] = exponentials / normalizer
        return peak + np.log(normalizer), probabilities
    log_all, probabilities = distribution(available)
    log_allowed, legal_probabilities = distribution(allowed)
    return float(log_all - log_allowed), probabilities - legal_probabilities
