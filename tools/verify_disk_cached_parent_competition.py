"""Check every frozen original relation and replay without recomputing it."""
import argparse
import copy
import cProfile
from datetime import datetime, timezone
from io import StringIO
import os
from pathlib import Path
import pstats
import time

import numpy as np
from threadpoolctl import threadpool_limits

from tools.diagnose_full_query_geometry import atomic
from tools.export_full_dataset_split_view import local_name
from tools.prepare_full_predicted_parent_pool import load_gzip
from tools.record_training_experiment import git_head
from weld_pipeline.server_disk_cached_parent_competition import DiskCachedParentCompetition
from weld_pipeline.server_lazy_parent_competition import LazyParentCompetition
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.server_predicted_parent_pool import FrozenRun446Priors
from weld_pipeline.server_reciprocal_cached_relation_prior import ReciprocalCachedRun446RelationPrior
from weld_pipeline.sjg_local_runtime_904 import digest, read, write

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['tools/verify_disk_cached_parent_competition.py',
    'weld_pipeline/server_disk_cached_parent_competition.py',
    'weld_pipeline/server_bounded_lazy_parent_competition.py',
    'weld_pipeline/server_reciprocal_cached_relation_prior.py',
    'weld_pipeline/server_bounded_cached_relation_prior.py',
    'weld_pipeline/server_cached_parent_relation_prior.py',
    'weld_pipeline/server_lazy_parent_competition.py',
    'weld_pipeline/server_predicted_parent_pool.py']


def main(args):
    base = ROOT / 'server_rebuild' / local_name(args.dataset_name)
    dataset = PartitionedFullShardDataset(ROOT, base)
    logical = Path(os.path.abspath(ROOT / args.pool_folder))
    logical.relative_to(ROOT / 'server_rebuild')
    folder = logical.resolve(strict=True)
    if not folder.is_relative_to(ROOT / 'server_rebuild'):
        link = logical.parent
        binding = read(link.parent / 'storage_binding.json')
        temporary = Path(binding['temporary_directory']).resolve(strict=True)
        temporary.relative_to('/tmp')
        if (link.name != 'pools' or not link.is_symlink()
                or Path(binding['project_link']) != link
                or not temporary.name.startswith('weld-predicted-parent-pools-')
                or link.resolve() != temporary or folder != temporary / logical.name):
            raise ValueError('The registered owned temporary pool storage changed')
    receipt = read(folder / 'receipt.json')
    if receipt['geometry_GT_reads'] != 0 or receipt['job_key'] not in dataset.jobs:
        raise ValueError('A genuine registered TRAIN input-only pool is required')
    for name, sha in receipt['files'].items():
        if Path(name).name != name or digest(folder / name) != sha:
            raise ValueError('An immutable original relation fixture changed')
    parents = load_gzip(folder / 'parent_pool.json.gz')
    with np.load(folder / 'relations.npz', allow_pickle=False) as z:
        pairs, expected_x, expected_base = z['pairs'], z['pair_features'], z['base_penalties']
    if not len(pairs):
        raise ValueError('A complete frozen original relation array is required')
    prior = ReciprocalCachedRun446RelationPrior(FrozenRun446Priors(ROOT), max_cached_parents=128)
    if prior.checkpoint_sha256 != receipt['prior_run446_sha256']:
        raise ValueError('The original frozen tree prior changed')
    out = base / local_name(args.experiment_name); out.mkdir(exist_ok=False)
    sources = {n:digest(ROOT / n) for n in SOURCES}
    write(out / 'protocol.json', dict(sources=sources, source_commit=git_head(ROOT),
        input_pool_folder=str(logical.relative_to(ROOT)), resolved_input_pool_folder=str(folder),
        pool_receipt_sha256=digest(folder / 'receipt.json'),
        prior_run446_sha256=prior.checkpoint_sha256, all_frozen_original_pairs_required=True,
        reference_geometry_reads=0, optimizer_updates=0))
    owner = None; tiny = None
    try:
        with threadpool_limits(limits=2):
            owner = DiskCachedParentCompetition(parents, prior, max_active_pairs=max(4_000_000, len(pairs)),
                                                max_cached_pairs=32)
            storage = owner.path
            original = LazyParentCompetition(parents, prior, max_active_pairs=max(4_000_000, len(pairs)))
            np.testing.assert_array_equal(owner.pairs_for(np.arange(len(parents))), pairs)
            for selected in [np.arange(min(256, len(parents))),
                    np.random.default_rng(1031).choice(len(parents), min(512, len(parents)), replace=False)]:
                np.testing.assert_array_equal(owner.pairs_for(selected), original.pairs_for(selected))
            profile = cProfile.Profile(); start = time.perf_counter(); profile.enable()
            actual = owner.arrays_for_pairs(pairs); profile.disable(); cold_seconds = time.perf_counter() - start
            stream = StringIO(); pstats.Stats(profile, stream=stream).strip_dirs().sort_stats('cumtime').print_stats(20)
            (out / 'controlled_cold_relation_profile.txt').write_text(stream.getvalue())
            np.testing.assert_array_equal(actual['pair_features'], expected_x)
            np.testing.assert_array_equal(actual['base_penalties'], expected_base)
            cold_computations = owner.computed_rows
            if cold_computations != len(pairs) or owner.stored_rows != len(pairs):
                raise ValueError('The full numerical fixture did not fit its declared cache resources')
            actual['pair_features'][:] = 0; actual['base_penalties'][:] = 0
            start = time.perf_counter(); replay = owner.arrays_for_pairs(pairs[::-1])
            warm_seconds = time.perf_counter() - start
            np.testing.assert_array_equal(replay['pair_features'], expected_x[::-1])
            np.testing.assert_array_equal(replay['base_penalties'], expected_base[::-1])
            if owner.computed_rows != cold_computations or owner.disk_hits != len(pairs):
                raise ValueError('A complete warm replay recomputed a frozen original relation')
            empty = owner.arrays_for_pairs(np.empty((0, 2), np.int64))
            assert empty['pair_features'].shape == (0, 32)
            for invalid in [np.r_[pairs[:1], pairs[:1]], pairs[:1, ::-1], np.array([[-1, 0]])]:
                try: owner.features(invalid)
                except ValueError: pass
                else: raise AssertionError('An invalid relation was admitted')
            saved = copy.deepcopy(parents[0]); parents[0]['segments'][0]['endpoints_mm'][0][0] += 1.
            try:
                try: owner.features(pairs[:1])
                except ValueError: pass
                else: raise AssertionError('Changed frozen input reused an old relation')
            finally:
                parents[0] = saved
            tiny = DiskCachedParentCompetition(parents, prior, max_disk_cache_bytes=65536,
                                               reserve_disk_bytes=0, max_cached_pairs=32)
            ids = np.arange(min(1024, len(pairs))); limited = tiny.arrays_for_pairs(pairs[ids])
            np.testing.assert_array_equal(limited['pair_features'], expected_x[ids])
            np.testing.assert_array_equal(limited['base_penalties'], expected_base[ids])
            limited = tiny.arrays_for_pairs(pairs[ids])
            np.testing.assert_array_equal(limited['pair_features'], expected_x[ids])
            np.testing.assert_array_equal(limited['base_penalties'], expected_base[ids])
            assert tiny.skipped_store_rows > 0 and tiny.path.stat().st_size <= 65536
            metrics = owner.metrics(); tiny_metrics = tiny.metrics()
            owner.close(); owner = None
            assert not storage.exists()
        if sources != {n:digest(ROOT / n) for n in sources}:
            raise ValueError('The numerical control source changed')
        atomic(out / 'receipt.json', dict(status='PASS_ALL_ORIGINAL_FROZEN_RELATIONS_AND_COMPLETE_DISK_REPLAY',
            recorded_utc=datetime.now(timezone.utc).isoformat(), checked_parents=len(parents),
            checked_relations=len(pairs), all_original_graph_pairs_and_active_subgraphs_identical=True,
            all_32_original_features_and_prior_penalties_bitwise_identical=True,
            all_reversed_replay_rows_identical_without_recomputation=True,
            returned_array_mutation_does_not_change_cached_values=True, changed_frozen_input_rejected=True,
            invalid_pairs_rejected=True, bounded_storage_overflow_retains_every_requested_edge=True,
            private_storage_removed_on_close=True, cold_seconds=cold_seconds, warm_replay_seconds=warm_seconds,
            metrics=metrics, tiny_storage_metrics=tiny_metrics, pool_receipt_sha256=digest(folder / 'receipt.json'),
            original_fixture_reused_no_old_inference_repeated=True, reference_geometry_reads=0,
            optimizer_updates=0, production_backend_changed=False,
            complete_training_speed_or_model_quality_not_established=True))
        print('FULL_DISK_RELATION_NUMERICAL_CONTROL_PASSED', len(pairs), cold_seconds, warm_seconds, flush=True)
    finally:
        if owner is not None: owner.close()
        if tiny is not None: tiny.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', default='full_dataset_v2')
    parser.add_argument('--pool-folder', required=True)
    parser.add_argument('--experiment-name', required=True)
    main(parser.parse_args())
