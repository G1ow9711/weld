"""Advise eviction of clean read cache for completed registered TRAIN inputs.

Linux file advice does not remove files, change their bytes, alter model
parameters or lower resource guards. Only registered completed input artifacts
are eligible; the caller can exclude inputs currently used by another worker.
No global page-cache command or cgroup configuration mutation is performed.
"""
import argparse
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'
GIB = 1024 ** 3


def read(path):
    return json.loads(path.read_text())


def memory_snapshot():
    cg = Path('/sys/fs/cgroup')
    maximum = (cg / 'memory.max').read_text().strip()
    current = int((cg / 'memory.current').read_text())
    stats = dict(line.split() for line in (cg / 'memory.stat').read_text().splitlines())
    host = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))
    limit = None if maximum == 'max' else int(maximum)
    return dict(cgroup_limit_bytes=limit, cgroup_current_bytes=current,
                conservative_headroom_bytes=host if limit is None else min(host, max(0, limit-current)),
                anonymous_bytes=int(stats['anon']), file_cache_bytes=int(stats['file']))


def main(args):
    if not hasattr(os, 'posix_fadvise'):
        raise RuntimeError('Linux POSIX file advice is unavailable')
    if not args.experiment_name or Path(args.experiment_name).name != args.experiment_name:
        raise ValueError('An independent local evidence name is required')
    out = BASE / args.experiment_name
    out.mkdir(exist_ok=False)
    registry = read(BASE / 'training_registry.json')
    excluded = set(args.exclude_job)
    if not excluded <= {job['job_key'] for job in registry['cases']}:
        raise ValueError('Excluded inputs must be registered TRAIN jobs')
    eligible = {}
    completed = []
    for job in registry['cases']:
        if job['job_key'] in excluded:
            continue
        folder = BASE / 'input_shards' / job['job_key']
        if not (folder / 'receipt.json').exists():
            continue
        receipt = read(folder / 'receipt.json')
        if receipt['workpiece_sha256'] != job['workpiece_sha256']:
            raise ValueError('Completed input receipt differs from its registration')
        completed.append(job['job_key'])
        eligible[(ROOT.parent/'data/G_Data'/job['workpiece_source']).resolve()] = 'REGISTERED_COMPLETED_TRAIN_PLY'
        for name in receipt['files']:
            if Path(name).name != name:
                raise ValueError('Unexpected input artifact filename')
            eligible[(folder/name).resolve()] = 'COMPLETE_IMMUTABLE_TRAIN_SHARD'
        provider = job.get('prior_input_provider')
        if provider:
            prior = ROOT/provider['input_directory']
            prior.resolve().relative_to(ROOT)
            eligible[(prior/'xyz_mm.npy').resolve()] = 'REGISTERED_PRIOR_XYZ'
    before = memory_snapshot()
    records = []
    for path, role in eligible.items():
        with path.open('rb') as stream:
            prior_stat = os.fstat(stream.fileno())
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            after_stat = os.fstat(stream.fileno())
            fields = ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
            if any(getattr(prior_stat, field) != getattr(after_stat, field) for field in fields):
                raise RuntimeError('Input artifact metadata changed during file advice')
        records.append(dict(path=str(path), role=role, bytes=prior_stat.st_size,
                            stat_binding_unchanged=True, operation='READONLY_FD_POSIX_FADV_DONTNEED'))
    after = memory_snapshot()
    result = dict(status='COMPLETED_TRAIN_READ_CACHE_ADVICE_RECORDED', before=before, after=after,
        completed_jobs=completed, excluded_active_jobs=sorted(excluded), files=records,
        eligible_file_bytes=sum(item['bytes'] for item in records),
        resource_guard_thresholds_changed=False, global_page_cache_dropped=False,
        files_removed=False, optimizer_updates=0, unix_time=time.time())
    (out/'receipt.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], files=len(records),
        before_headroom_gib=before['conservative_headroom_bytes']/GIB,
        after_headroom_gib=after['conservative_headroom_bytes']/GIB,
        observed_cgroup_current_reduction_gib=(before['cgroup_current_bytes']-after['cgroup_current_bytes'])/GIB)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--exclude-job', action='append', default=[])
    main(parser.parse_args())
