"""Verify unchanged full TRAIN inputs and costs through the adapted loader."""
import argparse
import copy
from pathlib import Path
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

from tools.compact_endpoint_training_backend import configure, control_arguments
from tools.train_full_predicted_parent_selector import load_job
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.sjg_local_runtime_904 import digest, read, write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'server_rebuild/full_dataset_v1'


def main(args):
    if Path(args.output_name).name!=args.output_name or Path(args.pool_name).name!=args.pool_name:
        raise ValueError('Independent local experiment names required')
    out=BASE/args.output_name;out.mkdir(exist_ok=False)
    sources={name:digest(ROOT/name) for name in ['tools/verify_compact_reference_training_backend.py',
        'tools/compact_endpoint_training_backend.py','tools/train_full_predicted_parent_selector.py',
        'weld_pipeline/server_compact_endpoint_reference.py']}
    begin=time.perf_counter()
    try:
        loader,guards,metadata=configure(args.reference_control,args.synthetic_control,out)
        # Reject a forged/stale synthetic binding before any loader or fitting call.
        fake=copy.deepcopy(read(Path(args.synthetic_control)))
        fake['helper_sha256']='0'*64
        fixture=out/'rejected_synthetic_binding.json';write(fixture,fake)
        try:
            configure(args.reference_control,fixture,out)
        except ValueError:
            negative=True
        else:
            raise AssertionError('A forged compiler binding was accepted')
        dataset=FullShardDataset(ROOT)
        if args.job not in dataset.jobs:
            raise ValueError('One approved TRAIN job is required')
        pools=BASE/args.pool_name
        with threadpool_limits(limits=2):
            original=load_job(dataset,args.job,pools,reference_chunk_size=1024)
            exact=loader(dataset,args.job,pools,reference_chunk_size=1024)
        names=['flat','offsets','counts','pairs','pair_features','base_penalties','pair_scale']
        for key in names:
            np.testing.assert_array_equal(original[key],exact[key],err_msg=key)
        assert original['pc']==exact['pc'] and original['pool']==exact['pool']
        assert original['pool_receipt_sha256']==exact['pool_receipt_sha256']
        assert [v['key'] for v in original['views']]==[v['key'] for v in exact['views']]
        for a,b in zip(original['views'],exact['views']):
            for key in ['key','G','target','weight','pool_capacity_reference_evaluation']:
                assert a[key]==b[key],key
            for key in ['costs','tp','child_counts','invalid_children']:
                np.testing.assert_array_equal(a['table'][key],b['table'][key],err_msg=key)
        if sources!={name:digest(ROOT/name) for name in sources}:
            raise ValueError('Backend source changed during actual comparison')
        write(out/'receipt.json',dict(status='PASS_ORIGINAL_FULL_JOB_LOADER_TOKENS_RELATIONS_COSTS_IDENTICAL',
            job_key=args.job,pool_name=args.pool_name,parents=len(original['counts']),
            token_rows=len(original['flat']),relation_rows=len(original['pairs']),
            all_same_input_versions=[v['key'] for v in original['views']],sources=sources,
            backend=metadata,guards=guards,forged_binding_rejected=negative,
            original_arrays_all_bitwise_equal=True,all_prediction_parents_retained=True,
            original_complete_set_training_terms_preserved=True,fit_calls=0,
            held_reference_reads=0,seconds=time.perf_counter()-begin,model_quality_not_measured=True))
        print('EXACT_REFERENCE_TRAINING_BACKEND_ACTUAL_PARITY_PASS',len(original['counts']),flush=True)
    except Exception as exc:
        write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pool-name',required=True)
    parser.add_argument('--job',required=True)
    parser.add_argument('--output-name',required=True)
    control_arguments(parser)
    main(parser.parse_args())
