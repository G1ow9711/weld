"""Authenticate complete TRAIN pools or frozen outputs, then compare all costs.

This numerical control changes neither predictions nor weights. A full pool
control retains every parent, while saved-output controls describe only the
declared frozen post-update snapshots, never full-dataset model accuracy.
"""
import argparse
import hashlib
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.server_parent_prediction_runtime import load_gzip
from weld_pipeline.server_predicted_parent_pool import decoded_subset
from weld_pipeline.server_exact_endpoint_reference import compile_exact_endpoint_reference
from weld_pipeline.server_offline_parent_set_reference import compile_compact_reference_table, evaluate_reference_subset
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents
from weld_pipeline.sjg_local_runtime_904 import read, write, digest
from tools.assess_full_weld_output_fields import authenticate
from tools.record_training_experiment import git_head

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'server_rebuild/full_dataset_v1'
SOURCES = ['tools/verify_actual_endpoint_reference.py',
    'weld_pipeline/server_exact_endpoint_reference.py', 'weld_pipeline/server_offline_parent_set_reference.py',
    'weld_pipeline/server_predicted_parent_pool.py', 'weld_pipeline/server_full_shard_dataset.py',
    'weld_pipeline/contract_geometry_d3_409.py', 'weld_pipeline/sjg_native_d3_adapter_1056.py',
    'weld_pipeline/world_set_evaluation_285.py', 'weld_pipeline/legacy_incremental_event_runtime_393.py',
    'tools/assess_full_weld_output_fields.py']


def named(value):
    if not value or Path(value).name != value or value in ('.','..'):
        raise ValueError('A local experiment name is required')
    return BASE/value


def atomic(path, value):
    temporary=path.with_name(path.name+'.incomplete')
    write(temporary,value)
    temporary.replace(path)


def candidates(args, dataset):
    if args.pool_name:
        if args.job not in dataset.jobs:
            raise ValueError('One registered TRAIN job is required')
        run=named(args.pool_name); folder=run/'pools'/args.job
        receipt=read(folder/'receipt.json')
        if (receipt['status']!='ACTUAL_PREDICTION_PARENT_POOL_FROZEN_BEFORE_TRAIN_REFERENCE_READ'
                or receipt['registry_sha256']!=dataset.registry_sha256
                or receipt['job_key']!=args.job or receipt['geometry_GT_reads']!=0
                or receipt['STEP_reads']!=0
                or receipt['query_checkpoint_sha256']!=digest(run/'query_checkpoint.pt')):
            raise ValueError('A complete input-only frozen TRAIN pool is required')
        for name, sha in receipt['files'].items():
            if Path(name).name!=name or digest(folder/name)!=sha:
                raise ValueError('Frozen candidate files changed')
        return [dict(job_key=args.job, path=str(folder/'parent_pool.json.gz'),
            sha256=digest(folder/'parent_pool.json.gz'), parents=receipt['parents'],
            receipt_path=str(folder/'receipt.json'),receipt_sha256=digest(folder/'receipt.json'),
            versions=[v['key'] for v in dataset.jobs[args.job]['versions']], complete_pool=True)]
    assessed=named(args.assessment_run)
    summary,protocol=read(assessed/'summary.json'),read(assessed/'protocol.json')
    if (summary['role']!='TRAIN' or protocol['role']!='TRAIN'
            or summary['source_protocol_sha256']!=digest(assessed/'protocol.json')
            or protocol['registry_sha256']!=dataset.registry_sha256):
        raise ValueError('Authenticated frozen TRAIN output assessment is required')
    run=Path(protocol['prediction_run']).resolve();run.relative_to(ROOT)
    source_protocol_sha=digest(run/'protocol.json')
    if source_protocol_sha!=protocol['prediction_protocol_sha256']:
        raise ValueError('The source prediction protocol changed')
    result=[]
    for item in summary['records']:
        path=ROOT/item['path']
        if digest(path)!=item['sha256']:
            raise ValueError('Saved assessment changed')
        assessed_record=read(path);source=assessed_record['source_receipt']
        receipt_path=ROOT/source['path']
        if digest(receipt_path)!=source['sha256']:
            raise ValueError('Saved TRAIN record changed')
        receipt,freeze=authenticate(run,receipt_path,source_protocol_sha,dataset.registry_sha256,'TRAIN')
        if receipt['job_key'] not in dataset.jobs or freeze['checkpoint_sha256']!=item['checkpoint_sha256']:
            raise ValueError('A frozen TRAIN model binding is required')
        versions=[v['key'] for v in dataset.jobs[receipt['job_key']]['versions']]
        if sorted(item['versions'])!=sorted(versions):
            raise ValueError('Every same-input registered label view must be retained')
        selected=receipt_path.parent/'selected_parents.json.gz'
        result.append(dict(job_key=receipt['job_key'],path=str(selected),sha256=digest(selected),
            receipt_path=str(receipt_path),receipt_sha256=digest(receipt_path),
            versions=versions,complete_pool=False,saved_training_views=receipt['training_views']))
    if not result:
        raise ValueError('No actual frozen inputs to compare')
    return result


def compare(parents,target,out,label):
    begin=time.perf_counter()
    original,original_score=compile_compact_reference_table(parents,target,decoded_subset,1024)
    baseline_seconds=time.perf_counter()-begin
    begin=time.perf_counter()
    def progress(value):
        atomic(out/'progress.json',dict(status='COMPARING_ALL_ACTUAL_COST_ROWS',label=label,**value))
    exact,exact_score=compile_exact_endpoint_reference(parents,target,decoded_subset,1024,progress=progress)
    accelerated_seconds=time.perf_counter()-begin
    hashes={}
    for key in ('costs','tp','child_counts','invalid_children'):
        np.testing.assert_array_equal(original[key],exact[key],err_msg=f'{label}:{key}')
        hashes[key]=hashlib.sha256(exact[key].tobytes()).hexdigest()
    for a,b in zip(linear_sum_assignment(original['costs']),linear_sum_assignment(exact['costs'])):
        np.testing.assert_array_equal(a,b)
    assert original_score['categories']['overall']==exact_score['categories']['overall']
    rng=np.random.default_rng(1163)
    ids=rng.choice(len(parents),min(64,len(parents)),replace=False)
    for selected in (ids,ids[::-1],np.empty(0,np.int64)):
        direct=score_native_parents(decoded_subset(parents,selected),target,[False]*len(target['parents']))
        fast=evaluate_reference_subset(exact,selected)
        assert all(fast[k]==direct['categories']['overall'][k] for k in ('predicted','TP','FP'))
    return dict(label=label,all_costs_and_tp_bitwise_equal=True,global_assignment_indices_equal=True,
        direct_original_subset_and_order_checks=3,array_sha256=hashes,
        cost_rows=len(parents),cost_columns=len(target['parents']),all_candidates_retained=True,
        original_seconds=baseline_seconds,exact_endpoint_seconds=accelerated_seconds,
        original_score=original_score['categories']['overall'],exact_compilation_audit=exact_score)


def main(args):
    out=named(args.output_name);out.mkdir(exist_ok=False)
    sources={name:digest(ROOT/name) for name in SOURCES}
    try:
        dataset=FullShardDataset(ROOT)
        inputs=candidates(args,dataset)  # ALL frozen predictions authenticated before labels.
        write(out/'protocol.json',dict(status='AUTHENTICATED_ALL_ORIGINAL_D3_COST_PARITY_CONTROL',
            source_commit=git_head(ROOT),sources=sources,registry_sha256=dataset.registry_sha256,
            inputs=inputs,fit_calls=0,held_reference_reads=0,model_accuracy=False))
        results=[]
        with threadpool_limits(limits=2):
            for index,item in enumerate(inputs):
                if sources!={name:digest(ROOT/name) for name in sources} or digest(item['path'])!=item['sha256']:
                    raise ValueError('Source or frozen predictions changed during comparison')
                parents=load_gzip(item['path'])
                if item['complete_pool'] and len(parents)!=item['parents']:
                    raise ValueError('Complete frozen pool was truncated')
                views=[]
                for key in item['versions']:
                    target=dataset.supervision(key).target
                    result=compare(parents,target,out,key)
                    if not item['complete_pool']:
                        saved=next(v for v in item['saved_training_views'] if v['version_key']==key)
                        assert all(result['original_score'][k]==saved[k] for k in ('G','predicted','TP','FP'))
                    views.append(result)
                path=out/f'job_{index:04d}.json'
                write(path,dict(input=item,views=views,predictions_changed=False,optimizer_updates=0))
                results.append(dict(path=str(path.relative_to(ROOT)),sha256=digest(path),job_key=item['job_key']))
                print('ACTUAL_ENDPOINT_REFERENCE_PARITY',item['job_key'],len(parents),len(views),flush=True)
        if sources!={name:digest(ROOT/name) for name in sources}:
            raise ValueError('Source changed before control completion')
        write(out/'receipt.json',dict(status='PASS_ALL_ACTUAL_ORIGINAL_COSTS_AND_GLOBAL_MATCHING_NOT_MODEL_ACCURACY',
            records=results,actual_input_records=len(inputs),all_same_input_label_views_retained=True,
            complete_pool_control=bool(args.pool_name),all_costs_and_tp_bitwise_equal=True,
            protocol_sha256=digest(out/'protocol.json'),fit_calls=0,held_reference_reads=0,
            predictions_changed=False,best_model_promoted=False,full_contract_pass=False))
    except Exception as exc:
        write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--pool-name');mode.add_argument('--assessment-run')
    parser.add_argument('--job');parser.add_argument('--output-name',required=True)
    main(parser.parse_args())
