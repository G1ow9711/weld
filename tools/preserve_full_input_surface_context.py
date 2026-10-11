"""Preserve genuine legacy input context before temporary preparation cleanup.

This separate read-only observer does not alter the running producer or its
compressed shards. Surfaces and input requests are original completed stage
artifacts, never estimates reconstructed from labels or padded model features.
Missing historical context is reported, not fabricated or silently accepted.
"""
import argparse
import fcntl
import gzip
import hashlib
import os
from pathlib import Path
import tempfile
import time

from weld_pipeline.sjg_local_runtime_904 import digest, read, write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'


def bytes_sha(data):
    return hashlib.sha256(data).hexdigest()


def save_original_bytes(path, data):
    with path.open('wb') as output:
        with gzip.GzipFile(filename='', fileobj=output, mode='wb', mtime=0) as stream:
            stream.write(data)
    with gzip.open(path, 'rb') as stream:
        if stream.read() != data:
            raise ValueError('Compressed genuine input metadata changed bytes')


def source_roots(job, evaluation):
    provider = job.get('prior_input_provider')
    if provider:
        source = (ROOT / provider['runtime_directory']).resolve()
        source.relative_to(ROOT)
        yield source, 'RETAINED_ORIGINAL_INPUT_PROVIDER'
    work = BASE / ('evaluation_workspace' if evaluation else 'training_workspace')
    yield work / job['job_key'] / 'frozen_features/legacy_runtime' / ('source_' + job['runtime_key']), 'LIVE_INPUT_PREPARATION'


def preserve(job, sources, destination, registry_sha):
    if (destination / 'receipt.json').exists():
        receipt = read(destination / 'receipt.json')
        if receipt['registry_sha256'] != registry_sha or receipt['workpiece_sha256'] != job['workpiece_sha256']:
            raise ValueError('Preserved input metadata registration changed')
        for name, sha in receipt['files'].items():
            if digest(destination / name) != sha:
                raise ValueError('Preserved genuine input context changed')
        return True
    key = job['runtime_key']
    for source, origin in sources:
        graph_binding = source / 'run285/all_geometry_frozen_before_labels.json'
        request_binding = source / 'run312/all_input_requests_frozen_before_labels.json'
        surfaces = source / 'run185' / ('source_' + key) / 'surfaces.json'
        profiles = surfaces.with_name('profiles.json')
        requests = source / 'run312' / ('source_' + key + '_input_requests.json')
        paths = [graph_binding, request_binding, surfaces, profiles, requests]
        if not all(p.is_file() for p in paths):
            continue
        try:
            graph = next(row for row in read(graph_binding)['cases'] if row['source'] == key)
            request = next(row for row in read(request_binding)['cases'] if row['source'] == key)
            if read(request_binding).get('GT_reads', 0) != 0:
                raise ValueError('Input requests must be frozen without reference geometry')
            surface_data, request_data = surfaces.read_bytes(), requests.read_bytes()
            profile_sha = digest(profiles)
            if (bytes_sha(surface_data) != graph['source_files']['surfaces.json']
                    or profile_sha != graph['source_files']['profiles.json']
                    or bytes_sha(request_data) != request['sha256']):
                raise ValueError('Completed genuine surface/request binding differs')
            # Read each bound artifact twice to reject an in-progress write or
            # simultaneous producer cleanup. No live producer file is mutated.
            if surface_data != surfaces.read_bytes() or request_data != requests.read_bytes():
                continue
        except FileNotFoundError:
            continue
        destination.parent.mkdir(exist_ok=True, parents=True)
        with tempfile.TemporaryDirectory(prefix=job['job_key'] + '-', dir=destination.parent) as tmp:
            stage = Path(tmp) / 'result'
            stage.mkdir()
            save_original_bytes(stage / 'surfaces.json.gz', surface_data)
            save_original_bytes(stage / 'input_requests.json.gz', request_data)
            write(stage / 'source_bindings.json', dict(graph=graph, input_requests=request,
                original_source_root=str(source), origin=origin,
                surface_bytes_sha256=bytes_sha(surface_data), request_bytes_sha256=bytes_sha(request_data)))
            write(stage / 'receipt.json', dict(status='GENUINE_INPUT_SURFACE_AND_REQUEST_CONTEXT_PRESERVED',
                job_key=job['job_key'], runtime_key=key, registry_sha256=registry_sha,
                workpiece_sha256=job['workpiece_sha256'], original_profiles_sha256=profile_sha,
                original_geometry_sha256=graph['geometry_sha256'],
                source_sha256=digest(__file__), files={p.name: digest(p) for p in stage.iterdir()},
                GT_geometry_reads=0, STEP_reads=0, optimizer_updates=0,
                input_metadata_fabricated=False, native_model_output_replay_not_yet_verified=True))
            stage.rename(destination)
        print('GENUINE_INPUT_CONTEXT_PRESERVED', job['job_key'], origin, flush=True)
        return True
    return False


def main(evaluation, once):
    registry_path = BASE / ('evaluation_registry.json' if evaluation else 'training_registry.json')
    registry_sha = digest(registry_path)
    jobs = read(registry_path)['cases']
    role = 'evaluation' if evaluation else 'train'
    out = BASE / ('legacy_surface_context_' + role + '_v1')
    out.mkdir(exist_ok=True)
    with (out / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_sha = digest(__file__)
        while True:
            if digest(registry_path) != registry_sha or digest(__file__) != source_sha:
                raise ValueError('Context observer registration/source changed')
            preserved, verified_shards, missing_complete = [], [], []
            shards = BASE / ('evaluation_input_shards' if evaluation else 'input_shards')
            for job in jobs:
                key, destination = job['job_key'], out / job['job_key']
                has_context = preserve(job, source_roots(job, evaluation), destination, registry_sha)
                if has_context:
                    preserved.append(key)
                if (shards / key / 'receipt.json').is_file():
                    if not has_context:
                        missing_complete.append(key)
                    else:
                        context, shard = read(destination / 'receipt.json'), read(shards / key / 'receipt.json')
                        if (context['original_profiles_sha256'] != shard['files']['profiles.json']
                                or context['original_geometry_sha256'] != shard['input_bindings']['geometry_sha256']):
                            raise ValueError('Preserved context differs from actual compressed input shard')
                        verified_shards.append(key)
            status = dict(status='GENUINE_CONTEXT_OBSERVER_NO_MODEL_FIT', role=role,
                controller_pid=os.getpid(), registry_sha256=registry_sha, preserved_jobs=preserved,
                verified_completed_shards=verified_shards, missing_context_for_completed_inputs=missing_complete,
                registered_jobs=len(jobs), source_sha256=source_sha, optimizer_updates=0,
                missing_context_fabricated=False, full_native_baseline_replay_complete=False, unix_time=time.time())
            temporary = out / 'progress.incomplete'
            write(temporary, status)
            os.replace(temporary, out / 'progress.json')
            producer = BASE / ('evaluation_preparation_v1' if evaluation else 'preparation') / 'receipt.json'
            if once or producer.exists():
                write(out / 'receipt.json', dict(status,
                    status='INPUT_CONTEXT_OBSERVATION_COMPLETE_WITH_GAPS' if missing_complete else 'INPUT_CONTEXT_OBSERVATION_COMPLETE',
                    all_registered_context_preserved=len(preserved) == len(jobs)))
                return
            time.sleep(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    main(args.evaluation, args.once)
