"""Resume the unchanged full input producer after verified resource-guard exits.

Only observed geometry-stage guard failures are recoverable here. All source
artifacts are preserved before same-filesystem renaming an incomplete owned
stage. Other errors stop with evidence. No input is skipped or truncated and
the original resource threshold, data and model weights are unchanged.
"""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from tools.prepare_full_training_case import ROOT, BASE, WORKSPACE, registered_job
from tools.reclaim_completed_training_read_cache import memory_snapshot, GIB
from weld_pipeline.sjg_local_runtime_904 import read, write, digest

GUARD = 'RuntimeError: Linux resource guard: less than 2.5 GiB headroom'
STAGE_FOLDERS = {'finite': 'learned_finite_support', 'surfaces': 'run26_learned_surfaces'}
SOURCES = ['tools/supervise_full_training_inputs.py', 'tools/prepare_full_training_inputs.py',
    'tools/prepare_full_training_case.py', 'tools/reclaim_completed_training_read_cache.py',
    'tools/rebuild_legacy_finite_support.py', 'tools/rebuild_run26_learned_surfaces.py',
    'weld_pipeline/server_legacy_runtime.py']


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.incomplete')
    write(temporary, value)
    os.replace(temporary, path)


def running_preparers():
    """Inspect actual Python module argv, not stale preparation state files."""
    matches = []
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():
            continue
        try:
            argv = folder.joinpath('cmdline').read_bytes().split(b'\0')
            modules = [argv[i+1] for i, value in enumerate(argv[:-1]) if value == b'-m']
            if any(module in (b'tools.prepare_full_training_inputs', b'tools.prepare_full_training_case')
                   for module in modules) and folder.joinpath('cwd').resolve() == ROOT:
                matches.append(int(folder.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return matches


def failure_stage():
    preparation = BASE / 'preparation'
    failed, progress = read(preparation / 'failure.json'), read(preparation / 'progress.json')
    job, stage = registered_job(progress['job_key']), progress['stage']
    log = preparation / job['job_key'] / (stage + '.log')
    text = log.read_text()
    final_line = text.rstrip().splitlines()[-1]
    if (final_line != GUARD or stage not in STAGE_FOLDERS
            or 'CalledProcessError' not in failed['error']
            or (preparation / job['job_key'] / 'stages' / (stage + '.json')).exists()):
        raise ValueError('Only the actual last resource-guard exit of an incomplete supported stage may retry')
    return job, stage, log, failed, progress


def file_manifest(folder):
    rows = []
    for path in sorted(folder.rglob('*')):
        if path.is_symlink():
            raise ValueError('Failed-stage backup must not follow symlinks')
        if path.is_file():
            rows.append(dict(path=str(path.relative_to(folder)), bytes=path.stat().st_size, sha256=digest(path)))
    return rows


def preserve_and_release(out, index):
    if running_preparers():
        raise ValueError('A live input producer or stage worker prevents recovery mutation')
    job, stage, log, failed, progress = failure_stage()
    temporary = WORKSPACE.resolve()
    if (not WORKSPACE.is_symlink() or temporary.parent != Path('/tmp')
            or not temporary.name.startswith('weld-full-training-')):
        raise ValueError('Unexpected full-input workspace ownership')
    source = temporary / job['job_key'] / 'frozen_features' / STAGE_FOLDERS[stage] / ('source_' + job['runtime_key'])
    source.resolve().relative_to(temporary)
    if not source.is_dir() or source.is_symlink() or (source / 'receipt.json').exists():
        raise ValueError('Recovery requires an existing incomplete owned geometry stage')
    archive = out / f'recovery_{index:04d}'
    archive.mkdir(exist_ok=False)
    write(archive / 'original_failure.json', failed)
    write(archive / 'original_progress.json', progress)
    shutil.copyfile(log, archive / 'original_stage.log')
    before = file_manifest(source)
    shutil.copytree(source, archive / 'incomplete_stage')
    if file_manifest(source) != before or file_manifest(archive / 'incomplete_stage') != before:
        raise ValueError('Incomplete-stage archival changed bytes')
    retained = source.with_name(source.name + '_resource_recovery_' + out.name + f'_{index:04d}')
    if retained.exists():
        raise FileExistsError(retained)
    # Both paths share the original temporary filesystem. Persistent copy above
    # has already been verified; no cross-device move or file deletion occurs.
    source.rename(retained)
    evidence = dict(status='INCOMPLETE_STAGE_PRESERVED_FOR_VERIFIED_RESOURCE_RETRY',
        job_key=job['job_key'], stage=stage, source_path=str(source), retained_workspace_path=str(retained),
        source_files=before, failure_sha256=digest(archive / 'original_failure.json'),
        worker_log_sha256=digest(archive / 'original_stage.log'),
        completed_shards_changed=False, input_algorithm_or_guard_changed=False, model_updates_reset=False)
    write(archive / 'receipt.json', evidence)
    return evidence


def main(args):
    if (not args.experiment_name or Path(args.experiment_name).name != args.experiment_name
            or args.experiment_name in ('.', '..') or args.headroom_gib < 2.5 or args.poll_seconds < 5):
        raise ValueError('An independent name and explicit conservative retry policy are required')
    out = BASE / args.experiment_name
    out.mkdir(exist_ok=False)
    with (BASE / 'input_resource_supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = {name: digest(ROOT / name) for name in SOURCES}
        registry_sha = digest(BASE / 'training_registry.json')
        write(out / 'protocol.json', dict(status='UNCHANGED_ALL_REGISTERED_INPUT_PRODUCER_RESOURCE_SUPERVISOR',
            sources=sources, registry_sha256=registry_sha, recovery_stage_folders=STAGE_FOLDERS,
            retry_headroom_gib=args.headroom_gib, poll_seconds=args.poll_seconds,
            original_guard_threshold_unchanged=True, all_registered_clouds_required=True,
            model_fitting_or_point_features_training=False))
        recoveries, attempt = [], 0
        try:
            if running_preparers():
                raise ValueError('Cannot start a duplicate full input producer')
            if args.recover_current_failure:
                recoveries.append(preserve_and_release(out, len(recoveries)))
            while not (BASE / 'preparation/receipt.json').exists():
                while memory_snapshot()['conservative_headroom_bytes'] < args.headroom_gib * GIB:
                    if sources != {name: digest(ROOT / name) for name in SOURCES}:
                        raise ValueError('Supervisor source changed while waiting')
                    atomic_json(out / 'progress.json', dict(status='WAITING_HEADROOM_BEFORE_UNCHANGED_INPUT_RETRY',
                        controller_pid=os.getpid(), attempts=attempt, recoveries=len(recoveries),
                        memory=memory_snapshot(), optimizer_updates=0, unix_time=time.time()))
                    time.sleep(args.poll_seconds)
                if (running_preparers() or registry_sha != digest(BASE / 'training_registry.json')
                        or sources != {name: digest(ROOT / name) for name in SOURCES}):
                    raise ValueError('Duplicate producer or frozen source/registry binding changed')
                attempt += 1
                with (out / f'attempt_{attempt:04d}.log').open('x') as log:
                    child = subprocess.Popen([sys.executable, '-u', '-m', 'tools.prepare_full_training_inputs'],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    atomic_json(out / 'progress.json', dict(status='RUNNING_UNCHANGED_FULL_INPUT_PRODUCER',
                        controller_pid=os.getpid(), child_pid=child.pid, attempt=attempt,
                        recoveries=len(recoveries), optimizer_updates=0, unix_time=time.time()))
                    code = child.wait()
                write(out / f'attempt_{attempt:04d}.json', dict(exit_code=code, attempt=attempt,
                    log_sha256=digest(out / f'attempt_{attempt:04d}.log'), unix_time=time.time()))
                if code == 0:
                    if not (BASE / 'preparation/receipt.json').exists():
                        raise ValueError('Input producer exited successfully without complete receipt')
                    break
                if code != 1:
                    raise RuntimeError('Input producer terminated unexpectedly: ' + str(code))
                recoveries.append(preserve_and_release(out, len(recoveries)))
            receipt = read(BASE / 'preparation/receipt.json')
            registry = read(BASE / 'training_registry.json')
            if (receipt['clouds'] != len(registry['cases'])
                    or receipt['training_versions'] != sum(len(j['versions']) for j in registry['cases'])
                    or receipt['registry_sha256'] != registry_sha):
                raise ValueError('Producer did not actually finish every registered input')
            write(out / 'receipt.json', dict(status='ALL_REGISTERED_TRAIN_INPUTS_PREPARED_WITH_VERIFIED_RESOURCE_RETRIES',
                attempts=attempt, recoveries=recoveries,
                complete_input_receipt_sha256=digest(BASE / 'preparation/receipt.json'), optimizer_updates=0))
        except BaseException:
            write(out / 'failure.json', dict(status='INPUT_SUPERVISOR_STOPPED_WITH_EVIDENCE_PRESERVED',
                attempts=attempt, recoveries=len(recoveries), traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--recover-current-failure', action='store_true')
    parser.add_argument('--headroom-gib', type=float, default=15.)
    parser.add_argument('--poll-seconds', type=int, default=10)
    main(parser.parse_args())
