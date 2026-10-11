"""Prepare all held XYZ inputs, then their complete frozen feature shards.

CPU-only XYZ extraction starts immediately. Frozen point-model computation
waits for the TRAIN input producer to finish so it does not compete with that
producer for GPU memory. No held predictions or reference scores are produced.
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

from tools.prepare_full_evaluation_case import BASE, ROOT, SHARDS, STAGES, OUT, WORKSPACE
from tools.prepare_full_training_case import GEOMETRY
from weld_pipeline.server_full_evaluation_dataset import EvaluationShardDataset
from weld_pipeline.sjg_local_runtime_904 import digest, read, write

SOURCE_NAMES = ['tools/prepare_full_evaluation_inputs.py', 'tools/prepare_full_evaluation_case.py',
    'tools/prepare_full_training_case.py', 'weld_pipeline/server_full_evaluation_dataset.py',
    'weld_pipeline/server_full_shard_dataset.py', 'weld_pipeline/server_fixed_fragment_queries.py',
    'tools/rebuild_expansion_input_blocks.py', 'tools/infer_legacy_point_features.py',
    'tools/infer_run26_point_features.py', 'tools/rebuild_legacy_rank2_candidates.py',
    'tools/rebuild_legacy_downstream.py', 'weld_pipeline/sjg_local_runtime_904.py'] + [
        'tools/' + name + '.py' for name, _ in GEOMETRY.values()]


def owned_storage(link, prefix):
    if not link.is_symlink():
        if link.exists():
            raise ValueError('Refuse to replace an existing held input storage directory')
        link.symlink_to(Path(tempfile.mkdtemp(prefix=prefix, dir='/tmp')), target_is_directory=True)
    target = link.resolve(strict=True)
    if target.parent != Path('/tmp') or not target.name.startswith(prefix):
        raise ValueError('Unexpected held input storage owner')
    return target


def training_input_preparation_complete():
    path = BASE / 'preparation/receipt.json'
    if not path.is_file():
        return False
    receipt, registry = read(path), read(BASE / 'training_registry.json')
    if (receipt['status'] != 'ALL_QUALIFIED_TRAIN_INPUTS_PREPARED_NOT_YET_MODEL_TRAINED'
            or receipt['registry_sha256'] != digest(BASE / 'training_registry.json')
            or receipt['clouds'] != len(registry['cases'])
            or receipt['training_versions'] != registry['qualified_training_versions']):
        raise ValueError('TRAIN input completion evidence does not match the complete registry')
    return True


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dataset = EvaluationShardDataset(ROOT)
        work = owned_storage(WORKSPACE, 'weld-full-evaluation-work-')
        shards = owned_storage(SHARDS, 'weld-full-evaluation-shards-')
        sources = {name: digest(ROOT / name) for name in SOURCE_NAMES}
        protocol_file = OUT / 'protocol.json'
        protocol = dict(status='COMPLETE_HELD_INPUT_PREPARATION_NOT_MODEL_FIT_OR_EVALUATION',
            registry_sha256=dataset.registry_sha256, source_bindings=sources,
            registered_cloud_jobs=len(dataset.jobs), registered_versions=len(dataset.versions),
            role_version_counts=dataset.registry['role_version_counts'], fit_allowed=False,
            source_commit_before_start=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            point_features_wait_for_complete_train_inputs=True, optimizer_updates=0,
            model_predictions_scored=0, test_reference_scoring=0,
            temporary_workspace=str(work), temporary_shards=str(shards))
        if protocol_file.exists():
            previous = read(protocol_file)
            for field in ['registry_sha256', 'source_bindings', 'temporary_workspace', 'temporary_shards']:
                if previous[field] != protocol[field]:
                    raise ValueError('Held input controller resume protocol changed')
        else:
            write(protocol_file, protocol)
        write(OUT / 'storage_binding.json', dict(workspace=str(work), shards=str(shards),
            scope='Task-owned rebuildable held input caches; raw inputs and frozen weights are retained'))

        def state(status, job=None, stage=None):
            raw_ready = sum((OUT / key / 'stages/input.json').is_file() for key in dataset.jobs)
            write(OUT / 'progress.json', dict(status=status, controller_pid=os.getpid(),
                job_key=job, stage=stage, raw_xyz_jobs_prepared=raw_ready,
                complete_input_jobs=len(dataset.ready_jobs()), registered_jobs=len(dataset.jobs),
                registered_versions=len(dataset.versions), fit_allowed=False, optimizer_updates=0,
                test_reference_scoring=0, unix_time=time.time()))

        def stage_job(job, stage):
            job_key = job['job_key']
            case_out = OUT / job_key
            case_out.mkdir(exist_ok=True)
            marker = case_out / 'stages' / (stage + '.json')
            if marker.exists():
                previous = read(marker)
                if (previous['registry_sha256'] != dataset.registry_sha256
                        or previous['source_sha256'] != sources['tools/prepare_full_evaluation_case.py']):
                    raise ValueError('Held preparation stage binding changed')
                return
            if shutil.disk_usage(work).free < 5 * 1024**3 or shutil.disk_usage(BASE).free < 5 * 1024**3:
                raise RuntimeError('Held input preparation lacks five GiB disk headroom')
            state('PREPARING_HELD_INPUT_ONLY', job_key, stage)
            print('HELD_INPUT_STAGE_START', job_key, stage, flush=True)
            with (case_out / (stage + '.log')).open('a') as log:
                subprocess.run([sys.executable, '-u', '-m', 'tools.prepare_full_evaluation_case',
                    '--job', job_key, '--stage', stage], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            if not marker.is_file():
                raise ValueError('Held worker exited without a complete stage marker')
            print('HELD_INPUT_STAGE_COMPLETE', job_key, stage, flush=True)

        for job in dataset.jobs.values():
            if job['job_key'] not in dataset.ready_jobs():
                stage_job(job, 'input')
        while not training_input_preparation_complete():
            state('HELD_XYZ_READY_WAITING_FOR_TRAIN_INPUT_PRODUCER_BEFORE_GPU_STAGES')
            time.sleep(5)
        train_completion_sha = digest(BASE / 'preparation/receipt.json')
        for job in dataset.jobs.values():
            key = job['job_key']
            if key not in dataset.ready_jobs():
                provider = job['prior_input_provider']
                stages = ('pack',) if provider and provider['observed_features_complete'] else STAGES[1:]
                for stage in stages:
                    stage_job(job, stage)
            dataset.input_case(key)
            own_work = work / key
            if own_work.exists():
                if own_work.is_symlink() or own_work.parent != work or not key.startswith('eval_cloud_'):
                    raise ValueError('Unexpected held cache cleanup scope')
                shutil.rmtree(own_work)
            state('COMPLETE_HELD_INPUT_JOB_VERIFIED', key)
        write(OUT / 'receipt.json', dict(status='ALL_QUALIFIED_HELD_INPUTS_READY_NOT_MODEL_EVALUATED',
            registry_sha256=dataset.registry_sha256, clouds=len(dataset.jobs), versions=len(dataset.versions),
            role_version_counts=dataset.registry['role_version_counts'], optimizer_updates=0,
            model_predictions_scored=0, test_reference_scoring=0,
            train_input_completion_sha256=train_completion_sha, source_bindings=sources))
        state('ALL_QUALIFIED_HELD_INPUTS_READY_NOT_MODEL_EVALUATED')
        print('ALL_QUALIFIED_HELD_INPUTS_READY', len(dataset.jobs), len(dataset.versions), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        OUT.mkdir(exist_ok=True)
        write(OUT / 'failure.json', dict(error=traceback.format_exc(), unix_time=time.time()))
        raise
