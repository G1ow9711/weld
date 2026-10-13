"""Keep registered TRAIN raw file cache from interrupting live training.

Only POSIX_FADV_DONTNEED on read-only descriptors is permitted. Open/mapped
files are excluded. This changes no file contents, model, optimizer or recipe.
It is a training resource aid, never an inference benchmark or quality check.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from tools.export_full_dataset_split_view import local_name
from weld_pipeline.sjg_local_runtime_904 import digest, read


ROOT = Path(__file__).resolve().parents[1]
MEMORY = Path('/sys/fs/cgroup/memory.current')


def process_identity(pid):
    stat = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
    argv = Path('/proc', str(pid), 'cmdline').read_bytes().split(b'\0')
    return stat[19], stat[0], [word.decode() for word in argv if word]


def busy_files(paths):
    """Conservatively skip the entire hint if a process cannot be inspected."""
    busy, inaccessible = set(), 0
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            for fd in (proc / 'fd').iterdir():
                try:
                    name = Path(os.readlink(fd)).resolve()
                    if name in paths:
                        busy.add(name)
                except FileNotFoundError:
                    pass
            for line in (proc / 'maps').read_text().splitlines():
                parts = line.split(None, 5)
                if len(parts) == 6 and parts[5].startswith('/'):
                    name = Path(parts[5]).resolve()
                    if name in paths:
                        busy.add(name)
        except PermissionError:
            inaccessible += 1
        except (FileNotFoundError, ProcessLookupError):
            pass
    return busy, inaccessible


def release_cold_files(paths):
    busy, inaccessible = busy_files(paths)
    before = int(MEMORY.read_text())
    value = dict(memory_before_GiB=round(before / 2**30, 2),
                 busy_open_or_mapped_raw_files_skipped=len(busy),
                 inaccessible_processes=inaccessible, training_raw_files_hinted=0,
                 logical_bytes_without_content_reads=0, file_contents_written=False)
    if inaccessible:
        return dict(value, status='NO_HINT_UNINSPECTABLE_PROCESS_HANDLES')
    for path in sorted(paths - busy):
        previous = path.stat()
        fd = os.open(path, os.O_RDONLY)
        try:
            opened = os.fstat(fd)
            if (opened.st_dev, opened.st_ino) != (previous.st_dev, previous.st_ino):
                raise ValueError('Registered raw path changed before the read-only cache hint')
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        after = path.stat()
        if ((previous.st_size, previous.st_ino, previous.st_mtime_ns)
                != (after.st_size, after.st_ino, after.st_mtime_ns)):
            raise ValueError('Raw file metadata changed during the read-only cache hint')
        value['training_raw_files_hinted'] += 1
        value['logical_bytes_without_content_reads'] += previous.st_size
    return dict(value, status='READ_ONLY_COLD_TRAIN_RAW_FILE_CACHE_HINT',
                memory_after_GiB=round(int(MEMORY.read_text()) / 2**30, 2),
                raw_file_sizes_inodes_and_mtimes_unchanged=True)


def main(args):
    if (args.threshold_gib <= 0 or args.poll_seconds < 1
            or args.cooldown_seconds < args.poll_seconds):
        raise ValueError('Positive memory threshold and bounded observation intervals are required')
    base = ROOT / 'server_rebuild' / local_name(args.dataset_name)
    registry_path = base / 'training_registry.json'
    registry_sha = digest(registry_path)
    registry = read(registry_path)
    if (registry['role'] != 'TRAIN' or registry['validation_or_test_fit_rows'] != 0
            or not registry['all_qualified_training_versions_registered']):
        raise ValueError('Only an explicit complete TRAIN registration is allowed')
    raw_root = (ROOT.parent / 'data/G_Data').resolve(strict=True)
    paths = set()
    for job in registry['cases']:
        if any(v['split'] != 'train' for v in job['versions']):
            raise ValueError('A held version entered the TRAIN cache hint scope')
        path = (raw_root / job['workpiece_source']).resolve(strict=True)
        path.relative_to(raw_root)
        if not path.is_file():
            raise ValueError('A registered raw input file is required')
        paths.add(path)
    handles = []
    for text in args.handle:
        name, pid, tick = text.rsplit(':', 2)
        name, pid = local_name(name), int(pid)
        actual_tick, state, argv = process_identity(pid)
        if actual_tick != tick or state in ('Z', 'X') or name not in argv:
            raise ValueError('Every protected training handle must be the same actual live process')
        handles.append((name, pid, tick))
    out = base / local_name(args.experiment_name)
    out.mkdir(exist_ok=False)
    source_sha = digest(__file__)
    (out / 'protocol.json').write_text(json.dumps(dict(arguments=vars(args),
        training_registry_sha256=registry_sha, source_sha256=source_sha,
        registered_train_raw_files=len(paths), protected_handles=handles,
        scope='Read-only OS file cache hint; no source, labels, model, optimizer or recipe changes',
        complete_optimization_rounds_added=0), ensure_ascii=False, indent=2) + '\n')
    last_hint = -float('inf')
    events = attempts = 0
    while True:
        if registry_sha != digest(registry_path) or source_sha != digest(__file__):
            raise ValueError('The declared cache hint source or complete TRAIN registration changed')
        live = []
        for name, pid, tick in handles:
            try:
                actual_tick, state, argv = process_identity(pid)
            except (FileNotFoundError, ProcessLookupError):
                continue
            if actual_tick == tick and state not in ('Z', 'X') and name in argv:
                live.append((name, pid, tick))
        memory = int(MEMORY.read_text())
        value = dict(utc=datetime.now(timezone.utc).isoformat(), controller_pid=os.getpid(),
            status='OBSERVING_SAME_LIVE_TRAINING_HANDLES', live_protected_handles=live,
            memory_GiB=round(memory / 2**30, 2), actual_cache_hint_events=events,
            cache_hint_attempts=attempts,
            models_or_raw_file_contents_written_by_watcher=False)
        if not live:
            value['status'] = 'ALL_PROTECTED_PROCESS_HANDLES_EXITED_WATCHER_STOPPED'
        elif (memory >= args.threshold_gib * 2**30
              and time.monotonic() - last_hint >= args.cooldown_seconds):
            event = dict(value, **release_cold_files(paths))
            with (out / 'cache_hint_events.jsonl').open('a') as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            attempts += 1
            if event['training_raw_files_hinted']:
                events += 1
            last_hint = time.monotonic()
            value['actual_cache_hint_events'] = events
            value['cache_hint_attempts'] = attempts
        pending = out / 'progress.json.pending'
        pending.write_text(json.dumps(value, ensure_ascii=False) + '\n')
        os.replace(pending, out / 'progress.json')
        if not live or args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', default='full_dataset_v2')
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--handle', action='append', required=True, help='training_name:pid:start_ticks')
    parser.add_argument('--threshold-gib', type=float, default=67)
    parser.add_argument('--poll-seconds', type=float, default=5)
    parser.add_argument('--cooldown-seconds', type=float, default=30)
    parser.add_argument('--once', action='store_true')
    main(parser.parse_args())
