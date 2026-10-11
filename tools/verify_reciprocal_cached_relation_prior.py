"""Check complete genuine relations and original ordered pair arithmetic."""
import argparse
from pathlib import Path
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_bounded_cached_relation_prior import BoundedCachedRun446RelationPrior
from weld_pipeline.server_reciprocal_cached_relation_prior import (
    ReciprocalCachedRun446RelationPrior, _PairContext, _features, _DISTANCES, _METRICS,
)
from weld_pipeline.sjg_parent_set_features_1160 import ParentRelations
from weld_pipeline.sjg_additive_parent_precision_1254 import pair_features
from weld_pipeline.server_predicted_parent_pool import FrozenRun446Priors
from weld_pipeline.sjg_local_runtime_904 import read, write, digest
from tools.prepare_full_predicted_parent_pool import load_gzip

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'
SOURCES = [
    'tools/verify_reciprocal_cached_relation_prior.py',
    'weld_pipeline/server_reciprocal_cached_relation_prior.py',
    'weld_pipeline/server_bounded_cached_relation_prior.py',
    'weld_pipeline/server_cached_parent_relation_prior.py',
    'weld_pipeline/sjg_parent_set_features_1160.py',
    'weld_pipeline/sjg_additive_parent_precision_1254.py',
    'weld_pipeline/world_set_evaluation_285.py',
    'weld_pipeline/contract_curve_geometry.py',
    'weld_pipeline/server_full_prediction_runtime.py',
]


def main(args):
    for value in [args.experiment_name, args.pool_name, args.job]:
        if not value or value in ('.', '..') or Path(value).name != value:
            raise ValueError('Independent local experiment, pool and job required')
    out = BASE / args.experiment_name
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    sources = {name: digest(ROOT / name) for name in SOURCES}
    write(out / 'protocol.json', dict(sources=sources, max_cached_parents=128,
        original_pair_code_called_in_both_directions=True,
        primitive_distance_cache_scope='ONE_IMMUTABLE_PAIR_CONTEXT',
        reference_geometry_reads=0, optimizer_updates=0))
    try:
        folder = BASE / args.pool_name / 'pools' / args.job
        receipt = read(folder / 'receipt.json')
        if receipt['geometry_GT_reads'] != 0:
            raise ValueError('An authentic input-only complete pool required')
        for name, sha in receipt['files'].items():
            if Path(name).name != name or digest(folder / name) != sha:
                raise ValueError('Frozen complete input changed')
        parents = load_gzip(folder / 'parent_pool.json.gz')
        with np.load(folder / 'relations.npz', allow_pickle=False) as arrays:
            pairs = arrays['pairs']
            expected_features = arrays['pair_features']
            expected_penalties = arrays['base_penalties']
        priors = FrozenRun446Priors(ROOT)
        if priors.checkpoint_sha256 != receipt['prior_run446_sha256']:
            raise ValueError('Original frozen relation prior changed')
        current = ReciprocalCachedRun446RelationPrior(priors, 128)

        def check(chosen, reverse=False):
            selected = pairs[chosen]
            if reverse:
                selected = selected[:, ::-1]
            features, probability = current.duplicate_probabilities(parents, selected)
            np.testing.assert_array_equal(features, expected_features[chosen])
            scale = 2 * np.minimum([parents[a]['child_count'] for a in selected[:, 0]],
                                  [parents[b]['child_count'] for b in selected[:, 1]])
            np.testing.assert_array_equal(probability * scale, expected_penalties[chosen])
            assert len(current.cache) <= 128
            assert _DISTANCES.get() is None and _METRICS.get() is None

        measurements = []
        with threadpool_limits(limits=2):
            for start in range(0, len(pairs), 2048):
                check(np.arange(start, min(start + 2048, len(pairs))))
                if start % 32768 == 0:
                    write(out / 'progress.json', dict(status='CHECKING_ALL_FROZEN_RELATIONS',
                        checked_relations=min(start + 2048, len(pairs)), required_relations=len(pairs)))
                    print('RECIPROCAL_RELATIONS_CHECKED', min(start + 2048, len(pairs)), flush=True)
            assert current.primitive_distance_computations == current.reused_primitive_distances > 0
            computed = current.primitive_distance_computations
            full_seconds = time.perf_counter() - started
            chosen = np.random.default_rng(1116).choice(len(pairs), min(512, len(pairs)), replace=False)
            check(chosen, reverse=True)
            # Compare each original directed 16-vector before symmetry averaging.
            for a, b in pairs[chosen[:32]]:
                original = ParentRelations.__new__(ParentRelations)
                memo = _PairContext.__new__(_PairContext)
                first, second = current.entry(a), current.entry(b)
                for column, name in enumerate(('parents', 'metadata', 'points', 'faces', 'face_pairs')):
                    setattr(original, name, [first[column], second[column]])
                    setattr(memo, name, [first[column], second[column]])
                np.testing.assert_array_equal(original.pair(0, 1), memo.pair(0, 1))
                np.testing.assert_array_equal(original.pair(1, 0), memo.pair(1, 0))
                np.testing.assert_array_equal(pair_features(original, 0, 1), _features(memo, 0, 1))
                # A later call must not retain arrays from the previous context.
                original.points = [v.copy() for v in original.points]
                original.points[0][0, 0] += 0.123
                memo.points = original.points
                np.testing.assert_array_equal(pair_features(original, 0, 1), _features(memo, 0, 1))
            assert _DISTANCES.get() is None and _METRICS.get() is None
            empty_features, empty_probs = current.duplicate_probabilities(parents, [])
            assert empty_features.shape == (0, 32) and empty_probs.shape == (0,)
            for invalid in [[[-1, 0]], [[0, len(parents)]]]:
                try:
                    current.duplicate_probabilities(parents, invalid)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Invalid original relation pair accepted')
                assert _DISTANCES.get() is None and _METRICS.get() is None
            fresh = list(parents)
            current.duplicate_probabilities(fresh, pairs[-1:])
            assert current.geometry_cache_evictions == 0
            assert current.parent_geometry_builds == len(np.unique(pairs[-1:]))
            assert current.primitive_distance_computations == current.reused_primitive_distances > 0
            # Measure the same complete relation subset in opposite run orders.
            sample = pairs[:min(16384, len(pairs))]
            probabilities = {}
            for repeat in range(2):
                order = [('original_bounded', BoundedCachedRun446RelationPrior),
                         ('reciprocal', ReciprocalCachedRun446RelationPrior)]
                if repeat:
                    order.reverse()
                for label, cls in order:
                    instance = cls(priors, 128)
                    mark = time.perf_counter()
                    xs, ps = [], []
                    for start in range(0, len(sample), 2048):
                        x, p = instance.duplicate_probabilities(parents, sample[start:start + 2048])
                        xs.append(x); ps.append(p)
                    elapsed = time.perf_counter() - mark
                    np.testing.assert_array_equal(np.vstack(xs), expected_features[:len(sample)])
                    probability = np.concatenate(ps)
                    if probabilities:
                        np.testing.assert_array_equal(probability, next(iter(probabilities.values())))
                    probabilities[label] = probability
                    measurements.append(dict(repeat=repeat, method=label, relations=len(sample),
                        seconds=elapsed, geometry_builds=instance.parent_geometry_builds))
            assert sources == {name: digest(ROOT / name) for name in sources}
        write(out / 'receipt.json', dict(status='PASS_RECIPROCAL_ORIGINAL_ALL_RELATIONS',
            job_key=args.job, pool_name=args.pool_name, parents=len(parents), checked_relations=len(pairs),
            all_original_frozen_relations_checked=True,
            all_32_features_and_prior_penalties_bitwise_equal=True,
            original_directed_arithmetic_and_later_mutated_context_checked=True,
            reversed_pair_replay_bitwise_equal=True, exception_scope_reset_checked=True,
            max_cached_parents=128, primitive_distance_computations=computed,
            reused_primitive_distances=computed, cache_does_not_outlive_one_pair=True,
            represented_geometry_kinds=sorted({s['kind'] for p in parents for s in p['segments']}),
            measurements=measurements, full_control_seconds=full_seconds,
            pool_receipt_sha256=digest(folder / 'receipt.json'),
            prior_run446_sha256=priors.checkpoint_sha256, sources=sources,
            original_candidates_graph_samples_and_features_unchanged=True,
            reference_geometry_reads=0, optimizer_updates=0,
            production_training_backend_changed=False,
            whole_training_speed_or_model_quality_not_established=True))
        print('RECIPROCAL_ORIGINAL_ALL_RELATIONS_PASS', len(pairs), flush=True)
    except BaseException as exc:
        write(out / 'failure.json', dict(error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--pool-name', required=True)
    parser.add_argument('--job', required=True)
    main(parser.parse_args())
