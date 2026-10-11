"""Bound cached input geometry while keeping original run446 pair features.

Eviction only recomputes the same input geometry on its next use. No parent,
relation or candidate is removed. The original feature and probability methods
are inherited unchanged. Each cache remains specific to one complete input.
"""
from collections import OrderedDict

from .server_cached_parent_relation_prior import CachedRun446RelationPrior


class BoundedCachedRun446RelationPrior(CachedRun446RelationPrior):
    def __init__(self, priors, max_cached_parents=4096):
        if type(max_cached_parents) is not int or max_cached_parents < 2:
            raise ValueError('At least two cached parents are required')
        self.max_cached_parents = max_cached_parents
        super().__init__(priors)

    def clear(self):
        super().clear()
        self.cache = OrderedDict()
        self.geometry_cache_evictions = 0
        self.peak_cached_parents = 0

    def entry(self, index):
        value = super().entry(index)
        self.cache.move_to_end(int(index))
        while len(self.cache) > self.max_cached_parents:
            self.cache.popitem(last=False)
            self.geometry_cache_evictions += 1
        self.peak_cached_parents = max(self.peak_cached_parents, len(self.cache))
        return value
