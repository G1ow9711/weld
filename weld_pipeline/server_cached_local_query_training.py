"""Freeze genuine TRAIN query inputs for alternating complete workpiece views.

Only input arrays used by the preserved model are cached. Reference targets
remain ordinary separate TRAIN supervision in the sample. A cache is not a
prediction and cannot enter deployment or any held evaluation.
"""
import hashlib

import numpy as np

from tools.train_full_weld_reference_recall import TrainingTensors
from .server_deployed_query_assignments import eligible_query_positions


def array_binding(array):
    value = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(value.dtype.str.encode())
    h.update(repr(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


class CachedLocalQueryInput:
    def __init__(self, inp, sample, neighborhoods, shared_tokens=None):
        self.tokens_by_profile = {}
        self.local = np.empty((len(sample['profile_ids']), 72), np.float32)
        shared_tokens = {} if shared_tokens is None else shared_tokens
        for pid in np.unique(sample['profile_ids']):
            pid = int(pid)
            rows = np.flatnonzero(sample['profile_ids'] == pid)
            positions = sample['input_query_positions_mm'][rows]
            if not np.isin(positions, eligible_query_positions(inp, pid)).all():
                raise ValueError('A cached query is absent from the deployed input grid')
            actual_tokens = inp.tokens(pid)[0]
            if pid not in shared_tokens:
                tokens = np.array(actual_tokens, copy=True, order='C')
                tokens.flags.writeable = False
                shared_tokens[pid] = tokens
            elif array_binding(actual_tokens) != array_binding(shared_tokens[pid]):
                raise ValueError('Shared query tokens belong to a different actual input')
            self.tokens_by_profile[pid] = shared_tokens[pid]
            self.local[rows] = neighborhoods.features(
                inp, pid, positions)
        if not np.isfinite(self.local).all() or len(self.local) != sample['report']['children']:
            raise ValueError('Actual complete child query contexts are required')
        self.local.flags.writeable = False
        self.context_report = dict(input_neighborhood_rows=len(self.local),
            local_feature_columns=72, queries_belong_to_deployed_input_grid=True,
            raw_point_tree_size=len(neighborhoods.xyz),
            references_or_source_ids_as_features=False,
            neighborhood_computation='ACTUAL_COMPLETE_RAW_XYZ_BEFORE_INPUT_CACHE',
            cached_local_features_sha256=array_binding(self.local))
        self.bindings = dict(
            local_features=array_binding(self.local),
            profile_ids=array_binding(sample['profile_ids']),
            query_positions=array_binding(sample['input_query_positions_mm']),
            features=array_binding(sample['features']),
            geometry=array_binding(sample['geometry']),
            tokens={pid:array_binding(tokens) for pid,tokens in self.tokens_by_profile.items()})
        self.nbytes = self.local.nbytes

    def tokens(self, pid):
        return self.tokens_by_profile[int(pid)], None

    def verify(self, sample):
        expected = dict(local_features=array_binding(self.local),
            profile_ids=array_binding(sample['profile_ids']),
            query_positions=array_binding(sample['input_query_positions_mm']),
            features=array_binding(sample['features']), geometry=array_binding(sample['geometry']),
            tokens={pid:array_binding(tokens) for pid,tokens in self.tokens_by_profile.items()})
        if expected != self.bindings:
            raise ValueError('A cached genuine input or its assigned query changed')
        return True


class CachedLocalTrainingTensors(TrainingTensors):
    def batch(self, inp, sample):
        if not isinstance(inp, CachedLocalQueryInput):
            raise ValueError('A frozen genuine raw-XYZ TRAIN input is required')
        # Binding verification is performed at construction and epoch admission;
        # immutable local arrays and ordinary separate targets retain their roles.
        self.last_context_report = dict(inp.context_report)
        return dict(super().batch(inp, sample), query_local_features=self.tensor(inp.local))
