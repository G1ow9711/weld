"""Check cached genuine relation features against frozen eager predictions."""
import argparse
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_cached_parent_relation_prior import CachedRun446RelationPrior
from weld_pipeline.server_predicted_parent_pool import FrozenRun446Priors
from weld_pipeline.sjg_local_runtime_904 import digest, read, write
from tools.prepare_full_predicted_parent_pool import load_gzip

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'


def main(job, pool_name, output_name, all_pairs=False):
    start = time.perf_counter()
    output = BASE / output_name
    if output.exists():
        raise ValueError('Choose a new immutable numerical control name')
    folder = BASE / pool_name / 'pools' / job
    receipt = read(folder / 'receipt.json')
    assert receipt['geometry_GT_reads'] == 0
    for name, value in receipt['files'].items():
        assert digest(folder/name) == value
    parents = load_gzip(folder/'parent_pool.json.gz')
    with np.load(folder/'relations.npz', allow_pickle=False) as arrays:
        pairs, features, base = arrays['pairs'], arrays['pair_features'], arrays['base_penalties']
    prior = CachedRun446RelationPrior(FrozenRun446Priors(ROOT))
    assert receipt['prior_run446_sha256'] == prior.checkpoint_sha256
    rng = np.random.default_rng(1085)
    ids = np.arange(len(pairs)) if all_pairs else rng.choice(len(pairs), min(512, len(pairs)), replace=False)
    checked = 0
    with threadpool_limits(limits=2):
        for start_row in range(0, len(ids), 2048):
            chosen = ids[start_row:start_row+2048]
            x, probability = prior.duplicate_probabilities(parents, pairs[chosen])
            np.testing.assert_array_equal(x, features[chosen])
            scale = 2*np.minimum([parents[a]['child_count'] for a in pairs[chosen, 0]],
                                [parents[b]['child_count'] for b in pairs[chosen, 1]])
            np.testing.assert_array_equal(probability*scale, base[chosen])
            checked += len(chosen)
            if start_row % 32768 == 0:
                print('CACHED_RELATION_ROWS_VERIFIED', checked, '/', len(ids), 'parent_builds', prior.parent_geometry_builds, flush=True)
        before = prior.parent_geometry_builds
        first = ids[:min(16, len(ids))]
        prior.duplicate_probabilities(parents, pairs[first])
        assert prior.parent_geometry_builds == before
        prior.duplicate_probabilities(list(parents), pairs[first])
        assert prior.parent_geometry_builds == len(np.unique(pairs[first]))
    kinds = sorted({s['kind'] for p in parents for s in p['segments']})
    write(output, dict(status='PASS_CACHED_GENUINE_RELATION_CONTROL_NOT_MODEL_EVALUATION',
          job_key=job, pool_name=pool_name, parents=len(parents), frozen_relations=len(pairs),
          checked_relations=checked, all_frozen_relations_checked=bool(all_pairs),
          genuine_32_features_and_prior_penalties_bitwise_equal=True,
          repeat_edges_do_not_rebuild_geometry=True, caches_reset_for_new_workpieces=True,
          represented_geometry_kinds=kinds, reference_geometry_reads=0, parameter_updates=0,
          prior_run446_sha256=prior.checkpoint_sha256,
          pool_receipt_sha256=digest(folder/'receipt.json'), source_sha256=digest(__file__),
          helper_sha256=digest(ROOT/'weld_pipeline/server_cached_parent_relation_prior.py'),
          seconds=time.perf_counter()-start))
    print('CACHED_GENUINE_RELATION_PARITY_PASS', checked, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--pool-name', required=True)
    parser.add_argument('--output-name', required=True)
    parser.add_argument('--all-pairs', action='store_true')
    args = parser.parse_args()
    main(args.job, args.pool_name, args.output_name, args.all_pairs)
