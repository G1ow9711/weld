"""Memoize unchanged TRAIN preparation within one immutable input case.

The original computations remain the only producers of values. Full query
batches are keys, so their order, vectorization and floating-point values are
preserved. Training responsibility targets never enter the raw XYZ cache.
"""
import json

import numpy as np

from .server_deployed_query_assignments import DeployedGridTrainingAssignments
from .server_query_point_neighborhood import RawQueryNeighborhood


class MemoizedRawQueryNeighborhood(RawQueryNeighborhood):
    def __init__(self, xyz, inp):
        super().__init__(xyz)
        self.bound_input = inp
        self.geometry_bindings = {}
        self.values = {}
        self.cache_hits = 0
        self.actual_computations = 0

    def features(self, inp, pid, positions_mm):
        if inp is not self.bound_input:
            raise ValueError('Raw query memoization belongs to another input case')
        pid = int(pid)
        representation = json.dumps(inp.profiles[pid]['representation'],
            sort_keys=True, separators=(',', ':'), allow_nan=False)
        if pid in self.geometry_bindings and self.geometry_bindings[pid] != representation:
            raise ValueError('The memoized input curve changed')
        self.geometry_bindings[pid] = representation
        positions = np.asarray(positions_mm, np.float64)
        curve = inp.curve(pid)
        if (positions.ndim != 1 or not np.isfinite(positions).all()
                or np.any(positions < 0) or np.any(positions > curve.length)):
            raise ValueError('Actual finite input query positions are required')
        key = (pid, positions.shape, positions.tobytes())
        if key not in self.values:
            value = np.array(super().features(inp, pid, positions), copy=True)
            value.flags.writeable = False
            self.values[key] = value
            self.actual_computations += 1
        else:
            self.cache_hits += 1
        # A caller cannot modify the stored values through its returned array.
        return self.values[key].copy()


class MemoizedTrainingAssignments(DeployedGridTrainingAssignments):
    """Cache only the original deterministic weak TRAIN responsibility rule."""
    def __init__(self, inp):
        super().__init__(inp)
        self.closest_values = {}
        self.closest_cache_hits = 0
        self.actual_closest_computations = 0

    def closest(self, child):
        # These are exactly the two fields read by the original closest method.
        key = json.dumps(dict(original_kind=child['original_kind'],
            representation=child['representation']), sort_keys=True,
            separators=(',', ':'), allow_nan=False)
        if key not in self.closest_values:
            self.closest_values[key] = super().closest(child)
            self.actual_closest_computations += 1
        else:
            self.closest_cache_hits += 1
        return self.closest_values[key]
