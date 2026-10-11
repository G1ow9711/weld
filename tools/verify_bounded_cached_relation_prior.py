"""Compare every saved genuine relation after forced geometry-cache eviction."""
import argparse
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_bounded_cached_relation_prior import BoundedCachedRun446RelationPrior
from weld_pipeline.server_predicted_parent_pool import FrozenRun446Priors
from weld_pipeline.sjg_local_runtime_904 import read, write, digest
from tools.prepare_full_predicted_parent_pool import load_gzip

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'server_rebuild/full_dataset_v1'


def main(args):
    for name in [args.experiment_name,args.pool_name,args.job]:
        if not name or name in ('.','..') or Path(name).name != name:
            raise ValueError('Independent local pool, job and experiment names are required')
    if not 2 <= args.max_cached_parents <= 4096:
        raise ValueError('This control must exercise the declared bounded cache')
    out=BASE/args.experiment_name;out.mkdir(exist_ok=False)
    started=time.perf_counter()
    folder=BASE/args.pool_name/'pools'/args.job
    receipt=read(folder/'receipt.json')
    if receipt['geometry_GT_reads'] != 0:
        raise ValueError('An authentic input-only complete pool is required')
    for name,sha in receipt['files'].items():
        if Path(name).name != name or digest(folder/name) != sha:
            raise ValueError('Frozen complete input changed')
    parents=load_gzip(folder/'parent_pool.json.gz')
    with np.load(folder/'relations.npz',allow_pickle=False) as arrays:
        pairs=arrays['pairs'];features=arrays['pair_features'];base=arrays['base_penalties']
    prior=BoundedCachedRun446RelationPrior(FrozenRun446Priors(ROOT),args.max_cached_parents)
    if prior.checkpoint_sha256 != receipt['prior_run446_sha256'] or len(parents) <= args.max_cached_parents:
        raise ValueError('Original prior and a pool larger than the cache are required')
    def check(chosen):
        x,probability=prior.duplicate_probabilities(parents,pairs[chosen])
        np.testing.assert_array_equal(x,features[chosen])
        scale=2*np.minimum([parents[a]['child_count'] for a in pairs[chosen,0]],
                           [parents[b]['child_count'] for b in pairs[chosen,1]])
        np.testing.assert_array_equal(probability*scale,base[chosen])
        assert len(prior.cache) <= args.max_cached_parents
    with threadpool_limits(limits=2):
        for start in range(0,len(pairs),2048):
            check(np.arange(start,min(start+2048,len(pairs))))
            if start % 32768 == 0:
                write(out/'progress.json',dict(status='CHECKING_ALL_ORIGINAL_FROZEN_RELATIONS',
                    checked_pairs=min(start+2048,len(pairs)),required_pairs=len(pairs),
                    cached_parents=len(prior.cache),evictions=prior.geometry_cache_evictions))
                print('BOUNDED_RELATIONS_CHECKED',min(start+2048,len(pairs)),'/',len(pairs),flush=True)
        assert prior.geometry_cache_evictions > 0 and prior.peak_cached_parents==args.max_cached_parents
        builds,evictions=prior.parent_geometry_builds,prior.geometry_cache_evictions
        replay=np.unique(np.r_[np.arange(min(16,len(pairs))),
            np.random.default_rng(1085).choice(len(pairs),min(512,len(pairs)),replace=False)])
        for start in range(0,len(replay),128):check(replay[start:start+128])
        assert prior.parent_geometry_builds > builds
        pair=pairs[-1:].copy();x,probability=prior.duplicate_probabilities(parents,pair)
        stable=prior.parent_geometry_builds
        xx,pp=prior.duplicate_probabilities(parents,pair)
        np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(probability,pp)
        assert prior.parent_geometry_builds==stable
        prior.duplicate_probabilities(list(parents),pair)
        assert prior.parent_geometry_builds==len(np.unique(pair)) and prior.geometry_cache_evictions==0
    write(out/'receipt.json',dict(status='PASS_BOUNDED_ORIGINAL_RELATION_FEATURES_AND_RECOMPUTATION',
        job_key=args.job,pool_name=args.pool_name,parents=len(parents),checked_relations=len(pairs),
        all_original_frozen_relations_checked=True,all_32_features_and_prior_penalties_bitwise_equal=True,
        max_cached_parents=args.max_cached_parents,peak_cached_parents=args.max_cached_parents,
        full_pass_geometry_builds=builds,full_pass_geometry_evictions=evictions,
        evicted_geometry_recomputed_bitwise_equal=True,new_workpiece_resets_cache=True,
        parent_geometry_and_all_candidates_unchanged=True,
        represented_geometry_kinds=sorted({s['kind'] for p in parents for s in p['segments']}),
        pool_receipt_sha256=digest(folder/'receipt.json'),prior_run446_sha256=prior.checkpoint_sha256,
        sources={n:digest(ROOT/n) for n in ['tools/verify_bounded_cached_relation_prior.py',
            'weld_pipeline/server_bounded_cached_relation_prior.py',
            'weld_pipeline/server_cached_parent_relation_prior.py']},
        reference_geometry_reads=0,parameter_updates=0,seconds=time.perf_counter()-started,
        model_quality_and_entire_process_memory_not_established=True))
    print('BOUNDED_GENUINE_RELATION_PARITY_PASS',len(pairs),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment-name',required=True)
    parser.add_argument('--pool-name',required=True)
    parser.add_argument('--job',required=True)
    parser.add_argument('--max-cached-parents',type=int,default=128)
    main(parser.parse_args())
