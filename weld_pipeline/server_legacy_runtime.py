"""Explicit Linux resource adapter for the preserved legacy inference modules.

Call install() before importing legacy workers. No model, feature, algorithm,
receipt or historical hash is changed. Linux memory counters are not Windows
commit counters; the mapping is recorded in every resource snapshot.
"""
import gc
import platform
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3


def resources():
    if platform.system() != "Linux":
        raise RuntimeError("This adapter supports Linux only")
    status = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmData:')):
            status[line.split(':')[0]] = int(line.split()[1]) * 1024
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            memory['available'] = int(line.split()[1]) * 1024
    # Fail closed if the current deployment's cgroup-v2 counters are absent.
    cg = Path('/sys/fs/cgroup')
    raw_limit = (cg / 'memory.max').read_text().strip()
    used = int((cg / 'memory.current').read_text())
    limit = None if raw_limit == 'max' else int(raw_limit)
    remaining = memory['available'] if limit is None else max(0, limit - used)
    available = min(memory['available'], remaining)
    return dict(available_physical_bytes=available,
                remaining_commit_bytes=available,
                private_commit_bytes=status['VmData'],
                working_set_bytes=status['VmRSS'],
                host_available_bytes=memory['available'],
                cgroup_limit_bytes=limit, cgroup_current_bytes=used,
                metric_mapping='Linux min(MemAvailable,cgroup headroom); '
                'legacy remaining_commit uses same conservative headroom; '
                'private_commit maps to VmData, not Windows commit equivalence')


def validate(snapshot, disk_free):
    if snapshot['available_physical_bytes'] < 2.5 * GIB:
        raise RuntimeError('Linux resource guard: less than 2.5 GiB headroom')
    if snapshot['private_commit_bytes'] > 12 * GIB:
        raise RuntimeError('Linux resource guard: VmData exceeds 12 GiB')
    if disk_free < 5 * GIB:
        raise RuntimeError('Linux resource guard: less than 5 GiB disk free')
    return snapshot


def check_resources():
    return validate(resources(), shutil.disk_usage(ROOT).free)


def reclaim_own_idle_pages():
    before = resources()
    gc.collect()
    return dict(before=before, after=resources(),
                scope='calling_worker_python_gc_only_no_OS_page_trim')


def install():
    from weld_pipeline import sjg_local_runtime_904 as legacy
    check_resources()
    legacy.resources = resources
    legacy.check_resources = check_resources
    legacy.reclaim_own_idle_pages = reclaim_own_idle_pages
    return dict(adapter='server_legacy_runtime',
                adapter_sha256=legacy.digest(__file__),
                preserved_runtime_sha256=legacy.digest(legacy.__file__),
                snapshot=resources(),
                install_before_legacy_worker_imports_required=True)
