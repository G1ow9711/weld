"""Reuse every original completed input and prepare all additional TRAIN inputs.

The original live input producer continues its complete registry. This controller
incrementally authenticates its finished shards and computes only inputs absent
from that registry. It never substitutes features from a different raw cloud.
"""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

from tools import reuse_repaired_full_input_shards as reuse
from tools.prepare_full_training_case import STAGES
from tools.prepare_partitioned_full_training_case import configure, SOURCES as WORKER_SOURCES
from tools.reclaim_completed_training_read_cache import memory_snapshot, GIB
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.sjg_local_runtime_904 import digest, read, write

ROOT = reuse.ROOT
SOURCES = list(dict.fromkeys(['tools/complete_partitioned_full_training_inputs.py',
    'tools/reuse_repaired_full_input_shards.py', *WORKER_SOURCES]))


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.incomplete')
    write(temporary, value)
    os.replace(temporary, path)


def workspace(base):
    link = base / 'training_workspace'
    if not link.exists() and not link.is_symlink():
        target = Path(tempfile.mkdtemp(prefix='weld-full-training-', dir='/tmp'))
        link.symlink_to(target, target_is_directory=True)
        write(base / 'temporary_workspace_binding.json', dict(temporary_directory=str(target),
            project_link=str(link), cleanup_scope='Verified completed new input job intermediates only'))
    target = link.resolve(strict=True)
    if not link.is_symlink() or target.parent != Path('/tmp') or not target.name.startswith('weld-full-training-'):
        raise ValueError('Unexpected corrected input workspace owner')
    return target


def source_guard(sources):
    if sources != {name: digest(ROOT / name) for name in sources}:
        raise ValueError('Corrected complete input producer source changed')


def main(args):
    dataset = configure(args.dataset_name)
    old = FullShardDataset(ROOT)
    if not set(old.jobs) <= set(dataset.jobs):
        raise ValueError('The repaired TRAIN curriculum must retain all original input jobs')
    if Path(args.experiment_name).name != args.experiment_name or args.experiment_name in ('', '.', '..'):
        raise ValueError('An independent input completion experiment name is required')
    out = dataset.base / args.experiment_name
    out.mkdir(exist_ok=False)
    preparation = dataset.base / 'preparation'
    preparation.mkdir(exist_ok=True)
    dataset.shards.mkdir(exist_ok=True)
    with (preparation / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        temporary = workspace(dataset.base)
        sources = {name: digest(ROOT / name) for name in SOURCES}
        additional = sorted(set(dataset.jobs) - set(old.jobs))
        protocol = dict(status='COMPLETE_CORRECTED_TRAIN_INPUTS_FROM_ALL_ORIGINAL_AND_ADDITIONAL_XYZ',
            registry_sha256=dataset.registry_sha256, original_registry_sha256=old.registry_sha256,
            sources=sources, registered_clouds=len(dataset.jobs), registered_versions=len(dataset.versions),
            additional_jobs=additional, all_original_jobs_reused_only_after_authenticated_completion=True,
            input_geometry_or_point_weights_modified=False, held_fit_calls=0, optimizer_updates=0,
            all_original_stages=list(STAGES), headroom_before_new_worker_gib=args.headroom_gib)
        atomic_json(out / 'protocol.json', protocol)
        child = None
        current = None
        log = None
        reuse_index = 0
        reuse_records = []
        try:
            while True:
                source_guard(sources)
                if dataset.registry_sha256 != digest(dataset.base / 'training_registry.json'):
                    raise ValueError('Complete corrected TRAIN registration changed')
                if child is not None and child.poll() is not None:
                    code = child.returncode
                    log.close()
                    job_key, stage = current
                    if code:
                        raise RuntimeError(f'Original full input stage failed: {job_key} {stage} exit {code}')
                    marker = preparation / job_key / 'stages' / (stage + '.json')
                    if not marker.is_file() or read(marker)['registry_sha256'] != dataset.registry_sha256:
                        raise ValueError('Additional worker exited without a complete bound stage marker')
                    if stage == 'pack':
                        dataset.input_case(job_key)
                        own = temporary / job_key
                        if own.is_dir():
                            if own.is_symlink() or own.parent != temporary or not job_key.startswith('cloud_'):
                                raise ValueError('Unexpected temporary cleanup scope')
                            shutil.rmtree(own)
                    child = current = log = None
                ready = set(dataset.ready_jobs())
                available = set(old.ready_jobs()) - ready
                if available and child is None:
                    name = args.experiment_name + f'_reuse_{reuse_index:04d}'
                    reuse.main(SimpleNamespace(dataset_name=args.dataset_name, experiment_name=name))
                    reuse_records.append(dict(path=str((dataset.base / name / 'receipt.json').relative_to(ROOT)),
                        sha256=digest(dataset.base / name / 'receipt.json')))
                    reuse_index += 1
                    ready = set(dataset.ready_jobs())
                if len(ready) == len(dataset.jobs):
                    break
                status = 'WAITING_ORIGINAL_COMPLETE_INPUT_SHARDS'
                if child is None:
                    pending = [key for key in additional if key not in ready]
                    headroom = memory_snapshot()['conservative_headroom_bytes']
                    if pending and headroom >= args.headroom_gib * GIB:
                        job_key = pending[0]
                        job = dataset.jobs[job_key]
                        provider = job['prior_input_provider']
                        stages = ('pack',) if provider and provider['observed_features_complete'] else STAGES
                        needed = [stage for stage in stages if not (
                            preparation / job_key / 'stages' / (stage + '.json')).is_file()]
                        if not needed:
                            raise ValueError('All additional stages exist without a complete input shard')
                        stage = needed[0]
                        if shutil.disk_usage(temporary).free < 5 * GIB or shutil.disk_usage(dataset.base).free < 5 * GIB:
                            raise RuntimeError('Complete input preparation lacks original five GiB storage headroom')
                        case_out = preparation / job_key
                        case_out.mkdir(exist_ok=True)
                        log = (case_out / (stage + '.log')).open('a')
                        argv = [sys.executable, '-u', '-m', 'tools.prepare_partitioned_full_training_case',
                            '--dataset-name', args.dataset_name, '--job', job_key, '--stage', stage]
                        child = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                        current = (job_key, stage)
                        print('ADDITIONAL_TRAIN_INPUT_STAGE_STARTED', job_key, stage, child.pid, flush=True)
                    elif pending:
                        status = 'WAITING_ORIGINAL_RESOURCE_HEADROOM_FOR_ADDITIONAL_TRAIN_INPUT'
                if child is not None:
                    status = 'RUNNING_COMPLETE_ADDITIONAL_TRAIN_INPUT_STAGE'
                atomic_json(out / 'progress.json', dict(status=status, controller_pid=os.getpid(),
                    child_pid=child.pid if child else None, job_key=current[0] if current else None,
                    stage=current[1] if current else None, ready_clouds=len(ready),
                    registered_clouds=len(dataset.jobs), registered_versions=len(dataset.versions),
                    optimizer_updates=0, GT_geometry_reads=0, unix_time=time.time()))
                time.sleep(args.poll_seconds)
            receipts = [read(dataset.shards / key / 'receipt.json') for key in dataset.jobs]
            families = {family for receipt in receipts for family in receipt['families']}
            if sum(receipt['training_versions'] for receipt in receipts) != len(dataset.versions):
                raise ValueError('Complete corrected input shards lost a TRAIN label version')
            source_guard(sources)
            completion = dict(status='ALL_QUALIFIED_TRAIN_INPUTS_PREPARED_NOT_YET_MODEL_TRAINED',
                clouds=len(dataset.jobs), training_versions=len(dataset.versions), families=len(families),
                registry_sha256=dataset.registry_sha256, fit_calls=0, validation_or_test_preparation_jobs=0,
                source_sha256=digest(__file__), sources=sources, protocol_sha256=digest(out / 'protocol.json'),
                incremental_reuse_records=reuse_records, input_pipeline_and_original_point_weights_unchanged=True)
            atomic_json(preparation / 'receipt.json', completion)
            atomic_json(out / 'receipt.json', completion)
            print('ALL_CORRECTED_TRAIN_INPUTS_READY', len(dataset.jobs), len(dataset.versions), flush=True)
        except BaseException:
            atomic_json(out / 'failure.json', dict(error=traceback.format_exc(), controller_pid=os.getpid(),
                child_pid=child.pid if child else None, current=current, unix_time=time.time()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--headroom-gib', type=float, default=15)
    parser.add_argument('--poll-seconds', type=float, default=5)
    args = parser.parse_args()
    if args.headroom_gib < 15 or args.poll_seconds < 1:
        raise ValueError('Original conservative resource policy and bounded polling are required')
    main(args)
