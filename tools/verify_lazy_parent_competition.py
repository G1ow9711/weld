"""Check lazy relations against a real, fully frozen competition graph.

No reference geometry is opened. Mathematical selections test complete set
scores and exclusion, including negative-unary comparator members. These
controls report numerical parity, not extraction accuracy.
"""
import argparse
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_lazy_parent_competition import LazyParentCompetition
from weld_pipeline.server_predicted_parent_pool import FrozenRun446Priors
from weld_pipeline.server_reference_parent_selection import greedy_nullable_set, set_model_utility
from weld_pipeline.sjg_local_runtime_904 import digest, read, write
from tools.prepare_full_predicted_parent_pool import load_gzip

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'


def main(job, pool_name, output_name):
    start = time.perf_counter()
    output = BASE / output_name
    if output.exists():
        raise ValueError('Choose a new immutable control record')
    folder = BASE / pool_name / 'pools' / job
    receipt = read(folder / 'receipt.json')
    assert receipt['geometry_GT_reads'] == 0
    for name, value in receipt['files'].items():
        assert digest(folder / name) == value
    parents = load_gzip(folder / 'parent_pool.json.gz')
    with np.load(folder / 'relations.npz', allow_pickle=False) as arrays:
        pairs = arrays['pairs']
        feature = arrays['pair_features']
        base = arrays['base_penalties']
    priors = FrozenRun446Priors(ROOT)
    assert receipt['prior_run446_sha256'] == priors.checkpoint_sha256
    graph = LazyParentCompetition(parents, priors)
    count = len(parents)
    rng = np.random.default_rng(1083)
    owners = [p['input_children'] for p in parents]
    rows = []
    # All nodes control the original radius/tie graph, independently of scores.
    np.testing.assert_array_equal(graph.pairs_for(np.arange(count)), pairs)
    for size in (0, 1, min(128, count), min(2048, count)):
        active = rng.choice(count, size, replace=False)
        mask = np.zeros(count, bool)
        mask[active] = True
        expected = pairs[mask[pairs[:, 0]] & mask[pairs[:, 1]]]
        np.testing.assert_array_equal(graph.pairs_for(active), expected)
        rows.append(dict(active_parents=size, exact_internal_edges=len(expected)))
    tested = rng.choice(len(pairs), min(128, len(pairs)), replace=False)
    with threadpool_limits(limits=2):
        actual_feature, actual_base = graph.features(pairs[tested])
    np.testing.assert_array_equal(actual_feature, feature[tested])
    np.testing.assert_array_equal(actual_base, base[tested])
    graph.features(pairs[tested])
    assert len(graph.cache) == len(tested)
    selections = []
    for case in range(6):
        unary = rng.normal(-2., 1., count) if case < 5 else np.zeros(count)
        active = np.flatnonzero(unary > 0)
        mask = np.zeros(count, bool)
        mask[active] = True
        restricted = mask[pairs[:, 0]] & mask[pairs[:, 1]]
        actual_pairs = graph.pairs_for(active)
        np.testing.assert_array_equal(actual_pairs, pairs[restricted])
        dense = greedy_nullable_set(unary, pairs, base, owners)
        lazy = greedy_nullable_set(unary, actual_pairs, base[restricted], owners)
        assert dense == lazy
        # Arbitrary comparator deliberately contains negative-unary nodes.
        teacher = np.unique(pairs[tested[:8]].ravel()).tolist()
        if case == 0:
            assert np.any(unary[teacher] <= 0)
        needed = np.unique(np.concatenate([graph.pairs_for(lazy), graph.pairs_for(teacher)]), axis=0)
        full_index = {tuple(pair): i for i, pair in enumerate(pairs)}
        penalty = np.asarray([base[full_index[tuple(pair)]] for pair in needed])
        full_difference = (set_model_utility(unary, pairs, base, dense)
                           - set_model_utility(unary, pairs, base, teacher))
        reduced_difference = (set_model_utility(unary, needed, penalty, lazy)
                              - set_model_utility(unary, needed, penalty, teacher))
        np.testing.assert_equal(reduced_difference, full_difference)
        selections.append(dict(positive_unary_parents=len(active), selected_parents=len(lazy),
                               runtime_edges=len(actual_pairs), loss_edges=len(needed),
                               comparator_includes_nonpositive_unary=bool(np.any(unary[teacher] <= 0))))
    result = dict(status='PASS_FULL_GRAPH_LAZY_RELATION_AND_SET_SCORE_CONTROL_NOT_MODEL_EVALUATION',
                  job_key=job, pool_name=pool_name, candidate_parents=count, frozen_edges=len(pairs),
                  full_graph_and_ties_exact=True, active_subgraph_checks=rows,
                  genuine_32_column_feature_and_penalty_checks=len(tested),
                  selections=selections, comparator_internal_edges_retained=True,
                  reference_geometry_reads=0, model_parameter_updates=0,
                  prior_run446_sha256=priors.checkpoint_sha256,
                  pool_receipt_sha256=digest(folder / 'receipt.json'),
                  source_sha256=digest(__file__), helper_sha256=digest(ROOT / 'weld_pipeline/server_lazy_parent_competition.py'),
                  seconds=time.perf_counter()-start)
    write(output, result)
    print('LAZY_FULL_PARENT_COMPETITION_PARITY_PASS', count, len(pairs), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--pool-name', required=True)
    parser.add_argument('--output-name', required=True)
    args = parser.parse_args()
    main(args.job, args.pool_name, args.output_name)
