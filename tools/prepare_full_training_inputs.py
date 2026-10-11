"""Resume all qualified training clouds; reuse inputs and stream owned work.

The complete registry is authoritative. Completed compressed shards persist;
only this controller's temporary per-cloud intermediates are removed after
packing. Raw data, previous experiments and frozen models are never deleted.
"""
import fcntl
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from tools.prepare_full_training_case import ROOT, BASE, WORKSPACE, SHARDS, STAGES, registered_job
from weld_pipeline.sjg_local_runtime_904 import read, write, digest


def verify_shard(job):
    dest = SHARDS / job['job_key']
    receipt = read(dest / 'receipt.json')
    if receipt['registry_sha256'] != digest(BASE / 'training_registry.json') or receipt['workpiece_sha256'] != job['workpiece_sha256']:
        raise ValueError('Prepared shard registry/input binding changed')
    for name, sha in receipt['files'].items():
        if digest(dest / name) != sha:
            raise ValueError('Prepared shard file changed')
    if digest(dest / 'supervision_versions.json') != receipt['supervision_versions_sha256']:
        raise ValueError('Shard supervision version bindings changed')
    return receipt


def workspace():
    if WORKSPACE.is_symlink():
        target = WORKSPACE.resolve()
        if target.parent != Path('/tmp') or not target.name.startswith('weld-full-training-') or not target.is_dir():
            raise ValueError('Unexpected temporary workspace; refuse to mutate it')
        return target
    if WORKSPACE.exists():
        raise ValueError('Training workspace must be an owned temporary directory symlink')
    target = Path(tempfile.mkdtemp(prefix='weld-full-training-', dir='/tmp'))
    WORKSPACE.symlink_to(target, target_is_directory=True)
    write(BASE / 'temporary_workspace_binding.json', dict(temporary_directory=str(target),
          project_link=str(WORKSPACE), cleanup_scope='Per-cloud intermediates after verified shard completion only'))
    return target


def main():
    out = BASE / 'preparation'
    out.mkdir(exist_ok=True)
    SHARDS.mkdir(exist_ok=True)
    registry = read(BASE / 'training_registry.json')
    with (out / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        temporary = workspace()
        completed = []
        for registered in registry['cases']:
            job = registered_job(registered['job_key'])
            job_key = job['job_key']
            destination = SHARDS / job_key / 'receipt.json'
            if destination.exists():
                completed.append(verify_shard(job))
                continue
            prior = job['prior_input_provider']
            stages = ('pack',) if prior and prior['observed_features_complete'] else STAGES
            case_out = out / job_key
            case_out.mkdir(exist_ok=True)
            for stage in stages:
                marker = case_out / 'stages' / (stage + '.json')
                if marker.exists():
                    prior_marker = read(marker)
                    if prior_marker['registry_sha256'] != digest(BASE / 'training_registry.json'):
                        raise ValueError('Preparation stage registry binding changed')
                    continue
                if shutil.disk_usage(temporary).free < 5 * 1024**3:
                    raise RuntimeError('Temporary workspace has less than 5GiB headroom')
                if shutil.disk_usage(BASE).free < 5 * 1024**3:
                    raise RuntimeError('Persistent training cache has less than 5GiB headroom')
                write(out / 'progress.json', dict(job_key=job_key, stage=stage, status='RUNNING',
                      controller_pid=os.getpid(), started_unix=time.time(), completed_clouds=len(completed),
                      total_clouds=len(registry['cases']), completed_training_versions=sum(r['training_versions'] for r in completed)))
                print('START', job_key, job['runtime_key'], stage, len(completed), '/', len(registry['cases']), flush=True)
                with (case_out / (stage + '.log')).open('a') as log:
                    subprocess.run([sys.executable, '-u', '-m', 'tools.prepare_full_training_case',
                                    '--job', job_key, '--stage', stage], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                if not marker.exists():
                    raise ValueError('Worker finished without a completed stage receipt')
                print('COMPLETE', job_key, stage, flush=True)
            receipt = verify_shard(job)
            completed.append(receipt)
            own_work = temporary / job_key
            if own_work.exists():
                # The path is constructed from the registered cloud job key and
                # an authenticated task-owned /tmp root. rmtree does not follow
                # child symlinks into reused original inputs.
                if own_work.parent != temporary or not job_key.startswith('cloud_'):
                    raise ValueError('Unexpected cleanup scope')
                shutil.rmtree(own_work)
            write(case_out / 'complete.json', dict(status='VERIFIED_SHARD_COMPLETE',
                  shard_receipt_sha256=digest(destination), temporary_intermediates_removed=True,
                  previous_experiments_or_raw_data_deleted=False))
        write(out / 'receipt.json', dict(status='ALL_QUALIFIED_TRAIN_INPUTS_PREPARED_NOT_YET_MODEL_TRAINED',
              clouds=len(completed), training_versions=sum(r['training_versions'] for r in completed),
              families=len({f for r in completed for f in r['families']}),
              registry_sha256=digest(BASE / 'training_registry.json'), fit_calls=0,
              validation_or_test_preparation_jobs=0, source_sha256=digest(__file__)))
        print('ALL_QUALIFIED_TRAIN_INPUTS_READY', len(completed), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        write(BASE / 'preparation/failure.json', dict(error=traceback.format_exc(), time_unix=time.time()))
        raise
