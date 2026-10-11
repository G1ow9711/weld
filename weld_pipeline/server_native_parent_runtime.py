"""Original native forests, full sparse path decode and frozen parent selection.

Inputs are genuine complete shard/context data, never a reference graph. All
conditional edges are streamed without sampling. This is the original native
branch under run446; it does not replace the separate incremental 526-column
validity/link/duplicate branch or certify process attributes and signed frames.
"""
import hashlib
from pathlib import Path
import pickle
import shutil
import time

import numpy as np

from .sjg_bounded_gap_runtime_1164 import GapResolver
from .sjg_compact_resources_1169 import compact_resources
from .sjg_duplicate_head_1209 import DuplicateRelations
from .sjg_incompatibility_head_1216 import IncompatibilityScorer
from .sjg_interval_forest_runtime_1129 import predict_instances
from .sjg_joint_forest_infer_1111 import decode_paths
from .sjg_parent_set_features_1160 import unary_features
from .sjg_precision_residual_1225 import SCHEMA, probability, deduction
from .sjg_relative_forest_runtime_1139 import relative_link_blocks
from .sjg_sparse_path_solver_1168 import select_paths
from .sjg_local_runtime_904 import digest, read


class FrozenNativeSelector:
    def __init__(self, root):
        root = Path(root)
        folder = root / 'weld_pipeline/reports/goal_contract_training_20261005/sjg_pilot_776/run379'
        training = read(folder / 'selector_training.json')
        names = list(training['frozen_files']) + ['precision_model.pkl']
        self.bindings = {name: digest(folder / name) for name in names}
        for name, sha in training['frozen_files'].items():
            if self.bindings[name] != sha:
                raise ValueError('Original frozen native selector weights changed')
        if self.bindings['precision_model.pkl'] != training['precision_model_sha256']:
            raise ValueError('Original frozen native precision forest changed')
        with (folder / 'precision_model.pkl').open('rb') as stream:
            correction = pickle.load(stream)
        if correction['schema'] != SCHEMA or correction['frozen_files'] != training['frozen_files']:
            raise ValueError('Original precision forest is not bound to this selector')
        self.correction = correction['model']
        self.model = IncompatibilityScorer.load(folder / 'selector.npz', folder / 'frozen_duplicate.npz',
                                               folder / 'frozen_residual.npz', folder / 'base_selector.npz')
        with np.load(folder / 'selector_normalization.npz', allow_pickle=False) as arrays:
            self.mean, self.scale = arrays['mean'], arrays['scale']
        self.training_receipt_sha256 = digest(folder / 'selector_training.json')

    def select(self, decoded, unary, metadata):
        if not decoded['parents']:
            if unary.shape != (0, 160) or metadata:
                raise ValueError('Empty native pool has inconsistent unary/metadata rows')
            return dict(decoded, runtime_model='FROZEN376_WITH_GEOMETRIC_PRECISION_RESIDUAL', GT_reads=0), dict(
                selected_candidate_indices=[], probability=np.empty(0), deduction=np.empty(0),
                selector_weight_bindings=self.bindings,
                selector_training_receipt_sha256=self.training_receipt_sha256,
                invalid_ledger_unchanged=True, no_candidate_precision_evaluations=0,
                GT_reads=0, STEP_reads=0, optimizer_updates=0)
        u = (unary - self.mean) / self.scale
        logits, _ = self.model.logits(u)
        q = probability(self.correction, u)
        penalty = deduction(q)
        context = DuplicateRelations(decoded['parents'], metadata, self.model.frozen)
        for _ in range(len(decoded['parents']) + 1):
            scores = self.model.forward(u, context.features(), context.selected, logits)
            scores[:-1] -= penalty
            chosen = self.model.choose_next(scores, context.selected)
            if chosen is None:
                break
            context.add(chosen)
        else:
            raise RuntimeError('Original native selector failed to choose STOP')
        indices = list(map(int, context.selected))
        result = dict(decoded, parents=[decoded['parents'][i] for i in indices])
        result['total_prediction_children'] = (sum(p['child_count'] for p in result['parents'])
            + sum(v['counted_false_children'] for v in decoded['invalid_predictions']))
        result.update(runtime_model='FROZEN376_WITH_GEOMETRIC_PRECISION_RESIDUAL', GT_reads=0)
        return result, dict(selected_candidate_indices=indices, probability=q, deduction=penalty,
            selector_weight_bindings=self.bindings, selector_training_receipt_sha256=self.training_receipt_sha256,
            invalid_ledger_unchanged=True, GT_reads=0, STEP_reads=0, optimizer_updates=0)


def complete_native_parents(case, models, cache, solve_seconds=60., progress=None):
    """Original interval/path/decode arithmetic with a complete streamed graph."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    screen = predict_instances(case, models)
    nodes = screen['nodes']
    if progress:
        progress('ORIGINAL_NATIVE_INTERVALS_COMPLETE', nodes=len(nodes), invalid=len(screen['invalid']))
    resolver = GapResolver(case['graph'], [n['fragment'] for n in nodes], audit=False)
    names = ('edges.i32', 'legal.u8', 'probabilities.f64')
    streams = {name: (cache / name).open('wb') for name in names}
    hashes = {name: hashlib.sha256() for name in names}
    features_hash = hashlib.sha256()
    edge_count = 0
    try:
        for block, (links, features, legal) in enumerate(relative_link_blocks(nodes, case, resolver)):
            values = np.zeros((len(links), 4))
            values[:, models['connection'].classes_.astype(int)] = models['connection'].predict_proba(features)
            next_bytes = links.nbytes + legal.nbytes + values.nbytes
            if shutil.disk_usage(cache).free - next_bytes < 5 * 1024**3:
                raise RuntimeError('Complete native graph cannot retain five GiB temporary headroom')
            for name, array in zip(names, (links.astype('<i4', copy=False), legal.astype('u1', copy=False),
                                           values.astype('<f8', copy=False))):
                data = array.tobytes()
                streams[name].write(data)
                hashes[name].update(data)
            features_hash.update(features.tobytes())
            edge_count += len(links)
            if progress and block % 200 == 0:
                progress('ORIGINAL_NATIVE_COMPLETE_EDGE_STREAM', nodes=len(nodes), block=block, edges=edge_count)
    finally:
        for stream in streams.values():
            stream.close()
    if edge_count:
        links = np.memmap(cache / names[0], dtype='<i4', mode='r', shape=(edge_count, 2))
        legal = np.memmap(cache / names[1], dtype='u1', mode='r', shape=(edge_count, 3)).view(bool)
        values = np.memmap(cache / names[2], dtype='<f8', mode='r', shape=(edge_count, 4))
    else:
        links, legal, values = np.empty((0, 2), np.int32), np.empty((0, 3), bool), np.empty((0, 4))
    resources = compact_resources(nodes, case['graph'], case['input_requests'], folder=cache)
    if progress:
        progress('ORIGINAL_NATIVE_ALL_EDGES_AND_RESOURCES_FROZEN', edges=edge_count,
                 resource_rows=len(resources['offsets']) - 1)
    selected, mapping = select_paths(nodes, links, values, legal, resources, wall_seconds=solve_seconds,
                                     allow_node_selection=True, phase=progress)
    decoded = decode_paths(nodes, selected, case['graph'], screen['invalid'])
    decoded.update(runtime_model='PRO359_EQUIVALENT_SPARSE_UPSTREAM', GT_reads=0)
    unary, metadata = unary_features(decoded, nodes, selected, case)
    evidence = dict(interval_nodes=len(nodes), invalid_interval_nodes=len(screen['invalid']),
        object_rows_scored=screen['objects_scored'], edge_count=edge_count,
        source_frames_and_connection_features_sha256=features_hash.hexdigest(),
        edge_files_sha256={name: h.hexdigest() for name, h in hashes.items()},
        resource_rows=len(resources['offsets']) - 1, resource_members=len(resources['indices']),
        path_selection=selected, gap_execution=resolver.receipt(),
        candidate_parents=len(decoded['parents']), candidate_children=decoded['total_prediction_children'],
        full_conditional_edge_graph_preserved=True, GT_reads=0, STEP_reads=0, optimizer_updates=0,
        incremental_run446_branch_executed=False, full_contract_pass=False,
        cached_native_pipeline_seconds=time.perf_counter() - start)
    del links, legal, values, resources, mapping
    return decoded, unary, metadata, evidence
