"""Independent full-input repeats with a frozen completed V5 query snapshot.

The parent selector is an explicitly partial TRAIN snapshot. Each process
regenerates all queries and candidates for the same registered raw input,
selects complete parents and exports CSV without reading supervision.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
from types import FunctionType

import numpy as np
import torch

from weld_pipeline.server_legacy_runtime import install
install()
from tools.prepare_full_predicted_parent_pool import prepare
from tools.qualify_full_dataset_labels import ply_points
from tools.verify_original_best_frozen_replay import canonical, audit_inference_io
from tools.export_full_dataset_split_view import local_name
from tools.record_training_experiment import git_head
from weld_pipeline.server_interval_endpoint_checkpoint import load_interval_endpoint_model, verify_complete_interval_endpoint_fit
from weld_pipeline.server_local_query_runtime import LocalCompleteQueryProposals
from weld_pipeline.server_parent_prediction_runtime import load_input_pool, load_selector, load_gzip, predict
from weld_pipeline.server_predicted_parent_pool import FrozenRun446Priors
from weld_pipeline.server_full_weld_csv_export import export_decoded, verify_roundtrip
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.sjg_local_runtime_904 import ROOT, digest, read, write


BASE = ROOT / 'server_rebuild/full_dataset_v2'
QUERY = BASE / 'full_interval_endpoint_geometry_v5_round04/last.pt'
QUERY_SHA = '1f1488106cc00501c39522a251b069fdc3ccf865de0d3e382d7f0f10649cc0aa'
PARENT = BASE / 'full_interval_endpoint_parent_v5_round04'
SOURCES = ['tools/check_interval_complete_output_repeatability.py',
    'tools/prepare_full_predicted_parent_pool.py', 'tools/qualify_full_dataset_labels.py',
    'tools/verify_original_best_frozen_replay.py',
    'weld_pipeline/server_local_query_runtime.py', 'weld_pipeline/server_query_point_neighborhood.py',
    'weld_pipeline/server_parent_prediction_runtime.py', 'weld_pipeline/server_ordered_parent_set_model.py',
    'weld_pipeline/server_predicted_parent_pool.py', 'weld_pipeline/server_lazy_parent_competition.py',
    'weld_pipeline/server_full_prediction_runtime.py', 'weld_pipeline/server_full_weld_csv_export.py',
    'weld_pipeline/server_full_shard_dataset.py', 'weld_pipeline/server_partitioned_full_dataset.py',
    'weld_pipeline/contract_curve_geometry.py', 'weld_pipeline/server_interval_endpoint_checkpoint.py',
    'weld_pipeline/server_interval_endpoint_residual_model.py']


def worker(out, attempt):
    torch.set_num_threads(2); torch.manual_seed(1031)
    torch.use_deterministic_algorithms(True)
    protocol = read(out / 'protocol.json')
    sources = protocol['sources']
    if sources != {name: digest(ROOT / name) for name in sources}:
        raise ValueError('Declared repeat source changed')
    dataset = PartitionedFullShardDataset(ROOT, BASE)
    job_key = protocol['job_key']; job = dataset.jobs[job_key]
    raw = ROOT.parent / 'data/G_Data' / job['workpiece_source']
    if digest(raw) != job['workpiece_sha256']:
        raise ValueError('Registered raw XYZ changed')
    xyz = ply_points(raw)
    if len(xyz) != job['original_points']:
        raise ValueError('A complete raw point cloud is required')
    state = torch.load(out / 'query_checkpoint.pt', map_location='cpu', weights_only=False)
    if digest(out / 'query_checkpoint.pt') != QUERY_SHA:
        raise ValueError('Frozen completed V5 query changed')
    fit = verify_complete_interval_endpoint_fit(ROOT, BASE, state, QUERY_SHA, QUERY.parent / 'receipt.json')
    model, architecture = load_interval_endpoint_model(state, 'cuda' if torch.cuda.is_available() else 'cpu')
    priors = FrozenRun446Priors(ROOT)
    folder = out / f'process_{attempt}'; folder.mkdir(exist_ok=False)
    (folder / 'query_checkpoint.pt').symlink_to(out / 'query_checkpoint.pt')
    storage = Path(tempfile.mkdtemp(prefix='weld-v5-complete-repeat-', dir='/tmp'))
    (folder / 'pools').symlink_to(storage, target_is_directory=True)
    write(folder / 'storage_binding.json', dict(storage=str(storage),
        scope='Own rebuildable complete raw-input proposals; retained for authenticated replay'))
    factory = lambda m, mean, std, **kwargs: LocalCompleteQueryProposals(m, mean, std, xyz, **kwargs)
    original = FunctionType(prepare.__code__, dict(prepare.__globals__, CompleteQueryProposals=factory),
                            prepare.__name__, prepare.__defaults__, prepare.__closure__)
    forbidden = {p.resolve() for p in (BASE / 'supervision_shards').rglob('*') if p.is_file()}
    with audit_inference_io(folder, forbidden) as inputs:
        pool_receipt = original(dataset, job_key, model, state, priors, folder,
            protocol['query_mode'], tuple(protocol['fixed_query_fractions']),
            protocol['competition_mode'], protocol['resource_mode'])
        selector, selector_state, training = load_selector(out / 'selector_checkpoint.pt',
            protocol['selector_sha256'], out / 'selector_protocol.json', QUERY_SHA,
            device='cuda' if torch.cuda.is_available() else 'cpu')
        data = load_input_pool(dataset, job_key, folder, priors)
        parents, decoded, freeze = predict(selector, data)
    if any(digest(path) != sha for path, sha in inputs.items()):
        raise ValueError('An actual repeat input changed')
    csv_directory = folder / 'csv'
    schema = export_decoded(csv_directory, job_key, decoded)
    roundtrip = verify_roundtrip(csv_directory, decoded)
    pool = folder / 'pools' / job_key
    candidate_hash = canonical(load_gzip(pool / 'child_proposals.json.gz'))
    parent_hash = canonical(load_gzip(pool / 'parent_pool.json.gz'))
    if sources != {name: digest(ROOT / name) for name in sources}:
        raise ValueError('Actual repeat source changed during inference')
    result = dict(status='ACTUAL_COMPLETE_INPUT_OUTPUT_REPEAT_FINISHED_PARTIAL_PARENT_CONTROL',
        pid=os.getpid(), query_architecture=architecture, query_complete_fit=fit,
        selector_completed_job_visits=len(selector_state['completed']),
        selector_optimizer_updates=selector_state['optimizer_steps'], job_key=job_key,
        original_raw_xyz_sha256=job['workpiece_sha256'], original_points=len(xyz),
        input_profiles=pool_receipt['original_input_profiles'],
        child_candidates=pool_receipt['children'], parent_candidates=pool_receipt['parents'],
        candidates_canonical_sha256=candidate_hash, parents_canonical_sha256=parent_hash,
        input_query_audit=read(pool / 'input_audit.json'),
        child_tensor_file_sha256=digest(pool / 'child_inputs.npz'),
        freeze_comparison={key: freeze[key] for key in ('selected_parent_indices',
            'unary_sha256', 'pairs_sha256', 'penalty_sha256', 'pool_parents',
            'selected_parents', 'total_prediction_children')},
        selected_complete_parents_sha256=canonical(parents),
        decoded_canonical_sha256=canonical(decoded), csv_rows=schema['published_rows'],
        csv_files={str(p.relative_to(csv_directory)): digest(p)
                   for p in sorted(csv_directory.rglob('*')) if p.is_file()},
        csv_roundtrip=roundtrip, reference_or_STEP_reads=0, fit_calls=0, full_contract_pass=False)
    write(folder / 'receipt.json', result)


def main(args):
    out = BASE / local_name(args.experiment_name)
    if args.worker is not None:
        worker(out, args.worker); return
    out.mkdir(exist_ok=False)
    dataset = PartitionedFullShardDataset(ROOT, BASE)
    job_key = min(dataset.jobs, key=lambda key: (dataset.jobs[key]['original_points'], key))
    snapshot = read(PARENT / 'resume.json')
    source = (PARENT / snapshot['checkpoint']).resolve(strict=True); source.relative_to(PARENT)
    for src, dest, sha in [(QUERY, out / 'query_checkpoint.pt', QUERY_SHA),
                          (source, out / 'selector_checkpoint.pt', snapshot['sha256'])]:
        if digest(src) != sha:
            raise ValueError('The actual immutable snapshot fingerprint changed')
        shutil.copyfile(src, dest)
        if digest(dest) != sha:
            raise ValueError('Immutable repeat snapshot copy differs')
    shutil.copyfile(PARENT / 'protocol.json', out / 'selector_protocol.json')
    training = read(out / 'selector_protocol.json')
    query = torch.load(out / 'query_checkpoint.pt', map_location='cpu', weights_only=False)
    sources = {name: digest(ROOT / name) for name in dict.fromkeys([*SOURCES, *query['sources']])}
    write(out / 'protocol.json', dict(source_commit=git_head(ROOT), sources=sources,
        job_key=job_key, registry_sha256=dataset.registry_sha256, query_sha256=QUERY_SHA,
        selector_sha256=snapshot['sha256'], selector_source_snapshot=str(source.relative_to(ROOT)),
        selector_protocol_sha256=digest(out / 'selector_protocol.json'),
        query_mode=training['query_mode'], fixed_query_fractions=training['fixed_query_fractions'],
        competition_mode=training['competition_mode'], resource_mode=training['resource_mode'],
        independent_processes=2, scope='One complete registered TRAIN raw input, fixed completed query and partial parent snapshot',
        comparison_excludes_timing_and_attempt_directory_prefixes=True,
        fit_calls=0, held_or_test_reads=0, complete_model_quality_certified=False))
    try:
        receipts = []
        for attempt in range(2):
            write(out / 'progress.json', dict(stage='INDEPENDENT_COMPLETE_INPUT_REPEAT', attempt=attempt,
                                             controller_pid=os.getpid(), job_key=job_key))
            env = dict(os.environ, PYTHONHASHSEED='1031', CUBLAS_WORKSPACE_CONFIG=':4096:8')
            subprocess.run([sys.executable, '-m', 'tools.check_interval_complete_output_repeatability',
                '--experiment-name', args.experiment_name, '--worker', str(attempt)], cwd=ROOT, env=env, check=True)
            receipts.append(read(out / f'process_{attempt}/receipt.json'))
        if receipts[0]['pid'] == receipts[1]['pid']:
            raise ValueError('Repeat workers were not independent')
        changed = [key for key in receipts[0] if key != 'pid' and receipts[0][key] != receipts[1][key]]
        if changed:
            raise ValueError('Actual complete raw-input repeat changed: ' + ', '.join(changed))
        write(out / 'receipt.json', dict(status='PASS_V5_FULL_INPUT_COMPLETE_OUTPUT_REPEATABILITY_PARTIAL_PARENT_CONTROL',
            protocol_sha256=digest(out / 'protocol.json'), job_key=job_key,
            processes=[r['pid'] for r in receipts], original_points=receipts[0]['original_points'],
            input_profiles=receipts[0]['input_profiles'], child_candidates=receipts[0]['child_candidates'],
            parent_candidates=receipts[0]['parent_candidates'], csv_rows=receipts[0]['csv_rows'],
            all_semantic_candidate_geometry_direction_parent_order_selection_and_CSV_equal=True,
            worker_receipts=[dict(path=f'process_{i}/receipt.json', sha256=digest(out / f'process_{i}/receipt.json'))
                             for i in range(2)], fit_calls=0, held_or_test_reads=0,
            partial_parent_control_not_final_model=True, raw_XYZ_latency_certified=False,
            complete_model_quality_certified=False, full_contract_pass=False))
        print('PASS_V5_ACTUAL_COMPLETE_INPUT_OUTPUT_REPEATABILITY', flush=True)
    except BaseException as exc:
        write(out / 'failure.json', dict(error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    main(parser.parse_args())
