"""Reuse primitive distances within the original two directions of one pair.

Both original ParentRelations.pair calls, including their float64 arithmetic,
still run. Only identical distance_to_primitive calls reuse their arrays. The
cache exists for one immutable pair context and is released in finally; it
cannot carry distances into another pair or workpiece. No graph edges, parent
candidates, samples, features or learned probabilities are removed.
"""
from contextvars import ContextVar
from types import FunctionType

from .server_bounded_cached_relation_prior import BoundedCachedRun446RelationPrior
from .server_cached_parent_relation_prior import CachedRun446RelationPrior
from .sjg_additive_parent_precision_1254 import pair_features
from .sjg_parent_set_features_1160 import ParentRelations


_DISTANCES = ContextVar('weld_pair_primitive_distances', default=None)
_METRICS = ContextVar('weld_pair_primitive_metrics', default=None)
_ORIGINAL_DISTANCE = ParentRelations.pair.__globals__['distance_to_primitive']


def _distance(points, segment):
    state = _DISTANCES.get()
    if state is None:
        return _ORIGINAL_DISTANCE(points, segment)
    # The context retains both arrays and all segment objects for both calls.
    # Identity reuse therefore cannot alias a different live primitive here.
    key = (id(points), id(segment))
    cache, metrics = state
    if key not in cache:
        cache[key] = _ORIGINAL_DISTANCE(points, segment)
        metrics['computed'] += 1
    else:
        metrics['reused'] += 1
    return cache[key]


def _clone(function, replacements):
    return FunctionType(function.__code__, dict(function.__globals__, **replacements),
                        function.__name__, function.__defaults__, function.__closure__)


class _PairContext(ParentRelations):
    pair = _clone(ParentRelations.pair, {'distance_to_primitive': _distance})


def _features(context, a, b):
    metrics = dict(computed=0, reused=0)
    token = _DISTANCES.set(({}, metrics))
    try:
        return pair_features(context, a, b)
    finally:
        _DISTANCES.reset(token)
        aggregate = _METRICS.get()
        if aggregate is not None:
            for key in metrics:
                aggregate[key] += metrics[key]


_DUPLICATES = _clone(CachedRun446RelationPrior.duplicate_probabilities,
                     {'ParentRelations': _PairContext, 'pair_features': _features})


class ReciprocalCachedRun446RelationPrior(BoundedCachedRun446RelationPrior):
    def clear(self):
        super().clear()
        self.primitive_distance_computations = 0
        self.reused_primitive_distances = 0

    def duplicate_probabilities(self, parents, pairs):
        metrics = dict(computed=0, reused=0)
        token = _METRICS.set(metrics)
        try:
            return _DUPLICATES(self, parents, pairs)
        finally:
            _METRICS.reset(token)
            self.primitive_distance_computations += metrics['computed']
            self.reused_primitive_distances += metrics['reused']
