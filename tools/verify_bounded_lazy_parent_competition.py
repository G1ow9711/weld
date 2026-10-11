"""Check complete frozen pairs and penalties after both cache evictions."""
import argparse
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_bounded_cached_relation_prior import BoundedCachedRun446RelationPrior
from weld_pipeline.server_bounded_lazy_parent_competition import BoundedLazyParentCompetition
from weld_pipeline.server_lazy_parent_competition import LazyParentCompetition
from weld_pipeline.server_predicted_parent_pool import FrozenRun446Priors
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from tools.prepare_full_predicted_parent_pool import load_gzip

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'server_rebuild/full_dataset_v1'


def main(args):
    for value in [args.experiment_name,args.pool_name,args.job]:
        if not value or value in ('.','..') or Path(value).name != value:
            raise ValueError('Independent local numerical control names required')
    out=BASE/args.experiment_name;out.mkdir(exist_ok=False)
    started=time.perf_counter();folder=BASE/args.pool_name/'pools'/args.job
    receipt=read(folder/'receipt.json')
    if receipt['geometry_GT_reads']!=0:raise ValueError('An authentic input-only pool is required')
    for name,sha in receipt['files'].items():
        if Path(name).name!=name or digest(folder/name)!=sha:raise ValueError('Frozen input changed')
    parents=load_gzip(folder/'parent_pool.json.gz')
    with np.load(folder/'relations.npz',allow_pickle=False) as arrays:
        pairs,features,base=arrays['pairs'],arrays['pair_features'],arrays['base_penalties']
    prior=BoundedCachedRun446RelationPrior(FrozenRun446Priors(ROOT),128)
    if prior.checkpoint_sha256!=receipt['prior_run446_sha256']:raise ValueError('Original relation prior differs')
    bounded=BoundedLazyParentCompetition(parents,prior,max_active_pairs=max(4_000_000,len(pairs)),max_cached_pairs=32)
    original=LazyParentCompetition(parents,prior,max_active_pairs=max(4_000_000,len(pairs)))
    with threadpool_limits(limits=2):
        np.testing.assert_array_equal(bounded.pairs_for(np.arange(len(parents))),pairs)
        for ids in [np.arange(min(256,len(parents))),
                np.random.default_rng(1085).choice(len(parents),512,replace=False)]:
            np.testing.assert_array_equal(bounded.pairs_for(ids),original.pairs_for(ids))
        values=bounded.arrays_for_pairs(pairs)
        np.testing.assert_array_equal(values['pairs'],pairs)
        np.testing.assert_array_equal(values['pair_features'],features)
        np.testing.assert_array_equal(values['base_penalties'],base)
        counts=np.array([p['child_count'] for p in parents])
        np.testing.assert_array_equal(values['pair_scale'],2*np.minimum(counts[pairs[:,0]],counts[pairs[:,1]]))
        assert bounded.feature_cache_evictions>0 and prior.geometry_cache_evictions>0
        peak=bounded.peak_cached_pairs;edge_evictions=bounded.feature_cache_evictions;geometry_evictions=prior.geometry_cache_evictions
        chosen=np.unique(np.r_[np.arange(min(16,len(pairs))),
            np.random.default_rng(1229).choice(len(pairs),512,replace=False)])[::-1]
        replay=bounded.arrays_for_pairs(pairs[chosen])
        np.testing.assert_array_equal(replay['pair_features'],features[chosen])
        np.testing.assert_array_equal(replay['base_penalties'],base[chosen])
        assert len(bounded.cache)<=32 and len(prior.cache)<=128
        empty=bounded.arrays_for_pairs(np.empty((0,2),int))
        assert empty['pair_features'].shape==(0,32) and empty['base_penalties'].shape==(0,)
        for bad in [np.r_[pairs[:1],pairs[:1]],pairs[:1,::-1],np.array([[-1,0]])]:
            try:bounded.features(bad)
            except ValueError:pass
            else:raise AssertionError('Invalid original relation contract was accepted')
    sources=['tools/verify_bounded_lazy_parent_competition.py',
        'weld_pipeline/server_bounded_lazy_parent_competition.py',
        'weld_pipeline/server_lazy_parent_competition.py',
        'weld_pipeline/server_bounded_cached_relation_prior.py',
        'weld_pipeline/server_cached_parent_relation_prior.py']
    write(out/'receipt.json',dict(status='PASS_COMPLETE_ORIGINAL_GRAPH_AND_BOUNDED_FEATURE_RECOMPUTATION',
        job_key=args.job,pool_name=args.pool_name,parents=len(parents),checked_relations=len(pairs),
        complete_graph_all_pairs_bitwise_equal=True,active_subset_full_input_graph_bitwise_equal=True,
        all_32_features_and_prior_penalties_and_scales_bitwise_equal=True,
        evicted_reversed_order_replay_bitwise_equal=True,invalid_pairs_rejected=True,
        max_cached_pairs=32,peak_cached_pairs=peak,feature_cache_evictions=edge_evictions,
        max_cached_parents=128,geometry_cache_evictions=geometry_evictions,
        all_candidate_and_relation_outputs_retained=True,pool_receipt_sha256=digest(folder/'receipt.json'),
        prior_run446_sha256=prior.checkpoint_sha256,sources={n:digest(ROOT/n) for n in sources},
        reference_geometry_reads=0,parameter_updates=0,seconds=time.perf_counter()-started,
        model_quality_and_entire_process_memory_not_established=True))
    print('COMPLETE_BOUNDED_ORIGINAL_COMPETITION_PASS',len(pairs),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment-name',required=True)
    parser.add_argument('--pool-name',required=True)
    parser.add_argument('--job',required=True)
    main(parser.parse_args())
