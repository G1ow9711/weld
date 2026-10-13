"""Prepare shared input event banks for every registered complete TRAIN view.

Existing actual assignments choose which profiles are needed for TRAIN. Event
coordinates themselves use only each input profile and original raw XYZ. The
same producer accepts any profile at inference; this cache is TRAIN-only.
"""
import argparse
from collections import defaultdict
import gc
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback

import numpy as np

from tools.export_full_dataset_split_view import local_name
from tools.qualify_full_dataset_labels import ply_points
from tools.record_training_experiment import git_head
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.server_supported_interval_events import FiniteInputIntervalEvents, COLUMNS, SCHEMA
from weld_pipeline.sjg_local_runtime_904 import ROOT, digest, read, write


BASE = ROOT / 'server_rebuild/full_dataset_v2'
ASSIGNMENTS = BASE / 'full_query_dense_endpoint_v5_geometry_comparison_20261012_v1/new_local'
SOURCES = ['tools/prepare_supported_interval_training_events.py',
    'weld_pipeline/server_supported_interval_events.py', 'weld_pipeline/server_finite_carrier_grid.py',
    'weld_pipeline/sjg_curve_queries_907.py', 'weld_pipeline/contract_curve_geometry.py',
    'weld_pipeline/server_partitioned_full_dataset.py', 'weld_pipeline/server_full_shard_dataset.py',
    'tools/qualify_full_dataset_labels.py']


def main(args):
    out = BASE / local_name(args.experiment_name); out.mkdir(exist_ok=False)
    dataset = PartitionedFullShardDataset(ROOT, BASE); dataset.require_complete_inputs()
    sources = {name: digest(ROOT / name) for name in SOURCES}
    prior = read(ASSIGNMENTS / 'receipt.json')
    controls_path = BASE / local_name(args.control_name) / 'receipt.json'
    controls = read(controls_path)
    if controls['status'] != 'PASS_SHARED_FINITE_INPUT_EVENT_AND_INTERVAL_CONTROLS':
        raise ValueError('Actual input event and assembly controls must pass before full preparation')
    control_protocol = read(controls_path.parent / 'protocol.json')
    if (digest(controls_path.parent / 'protocol.json') != controls['protocol_sha256']
            or any(digest(ROOT / name) != sha for name, sha in control_protocol['sources'].items())):
        raise ValueError('Passed input event controls belong to different source')
    if (len(prior['records']) != len(dataset.versions)
            or {r['version_key'] for r in prior['records']} != set(dataset.versions)):
        raise ValueError('All qualified TRAIN versions must remain registered')
    assignments = {}; jobs = defaultdict(set); record_bindings = {}
    for row in prior['records']:
        key = row['version_key']; path = (ROOT / row['path']).resolve(); path.relative_to(ASSIGNMENTS)
        if digest(path) != row['sha256']:
            raise ValueError('Existing actual TRAIN query assignment changed')
        value = read(path); job_key, version = dataset.versions[key]
        if len(value['children']) != version['children']:
            raise ValueError('A complete TRAIN reference denominator changed')
        assignments[key] = [dict(original_child_id=c['original_child_id'], profile_id=int(c['profile_id']),
                                input_query_position_mm=float(c['query_position_mm'])) for c in value['children']]
        jobs[job_key].update(c['profile_id'] for c in assignments[key])
        record_bindings[key] = row['sha256']
    storage = Path(tempfile.mkdtemp(prefix='weld-supported-interval-banks-', dir='/tmp'))
    (out / 'banks').symlink_to(storage, target_is_directory=True)
    write(out / 'storage_binding.json', dict(storage=str(storage),
        scope='Rebuildable input-only event coordinates and evidence; no source data or weights'))
    protocol = dict(source_commit=git_head(ROOT), sources=sources, schema=SCHEMA,
        training_registry_sha256=dataset.registry_sha256, assignment_receipt_sha256=digest(ASSIGNMENTS / 'receipt.json'),
        passed_control_sha256=digest(controls_path), columns=list(COLUMNS), spacing_mm=1., max_events_per_profile=1_000_001,
        registered_jobs=len(dataset.jobs), registered_versions=len(dataset.versions),
        registered_children=sum(v['children'] for _, v in dataset.versions.values()),
        assignment_record_bindings=record_bindings,
        scope='Every complete TRAIN view, caching its assigned input profiles only; not all runtime profiles',
        reference_coordinates_used_to_generate_events=False, GT_event_insertion=False,
        grid_events_are_not_observed_contact=True, held_or_test_reference_reads=0, training_updates=0)
    write(out / 'protocol.json', protocol)
    completed = []; completed_views = 0; completed_children = 0; event_count = 0
    try:
        order = sorted(dataset.jobs, key=lambda k: (dataset.jobs[k]['original_points'], k))
        for index, job_key in enumerate(order):
            if shutil.disk_usage(storage).free < 3 * 1024**3:
                raise RuntimeError('Owned event-bank storage has less than three GiB headroom')
            job = dataset.jobs[job_key]; folder = storage / job_key; folder.mkdir(exist_ok=False)
            raw = ROOT.parent / 'data/G_Data' / job['workpiece_source']
            if digest(raw) != job['workpiece_sha256']:
                raise ValueError('Complete raw XYZ fingerprint changed')
            xyz = ply_points(raw)
            if len(xyz) != job['original_points']:
                raise ValueError('Complete raw XYZ count changed')
            inp = dataset.input_case(job_key)
            input_sha = digest(dataset.shards / job_key / 'receipt.json')
            producer = FiniteInputIntervalEvents(xyz, input_sha)
            profiles = []; bank = None
            for pid in sorted(jobs[job_key]):
                if not 0 <= pid < len(inp.profiles):
                    raise ValueError('A registered training query references an absent input profile')
                begin, end = inp.offsets[pid:pid + 2]
                bank = producer.build(inp.profiles[pid], inp.spans[begin:end, :2])
                name = f'profile_{pid:07d}.npz'
                np.savez_compressed(folder / name, positions_mm=bank['positions_mm'],
                    xyz_mm=bank['xyz_mm'], provenance=bank['provenance'], projection_point_ids=bank['projection_point_ids'])
                profiles.append(dict(profile_id=pid, path=name, sha256=digest(folder / name),
                    curve_representation=bank['curve_representation'], length_mm=bank['length_mm'], audit=bank['audit']))
                event_count += len(bank['positions_mm'])
            versions = {v['key']: dict(children=v['children'], family_equal_version_weight=v['family_equal_version_weight'],
                assignment_record_sha256=record_bindings[v['key']], assignments=assignments[v['key']]) for v in job['versions']}
            if sources != {name: digest(ROOT / name) for name in sources}:
                raise ValueError('Input event producer changed during full preparation')
            receipt = dict(status='ALL_ASSIGNED_TRAIN_INPUT_PROFILE_EVENTS_READY', schema=SCHEMA,
                job_key=job_key, registry_sha256=dataset.registry_sha256, input_receipt_sha256=input_sha,
                raw_xyz_sha256=job['workpiece_sha256'], raw_points=len(xyz), profiles=profiles, versions=versions,
                original_input_profiles=len(inp.profiles), prepared_training_profiles=len(profiles),
                protocol_sha256=digest(out / 'protocol.json'), reference_coordinates_used_to_generate_events=False,
                partial_training_profile_cache_not_full_runtime_output=True, weights_or_raw_data_changed=False)
            write(folder / 'receipt.json', receipt)
            completed.append(dict(job_key=job_key, path=str(folder / 'receipt.json'), sha256=digest(folder / 'receipt.json')))
            completed_views += len(versions); completed_children += sum(v['children'] for v in versions.values())
            write(out / 'progress.json', dict(stage='PREPARING_ALL_REGISTERED_TRAIN_INTERVAL_EVENTS', pid=os.getpid(),
                completed_jobs=index + 1, jobs=len(order), completed_views=completed_views,
                views=len(dataset.versions), children=completed_children, events=event_count, unix_time=time.time()))
            print('COMPLETE_TRAIN_EVENT_JOB', index + 1, len(order), job_key, len(profiles), completed_children, flush=True)
            del bank, inp, producer, xyz
            gc.collect()
        if (completed_views != protocol['registered_versions'] or completed_children != protocol['registered_children']
                or len(completed) != protocol['registered_jobs']):
            raise ValueError('A complete registered TRAIN job/view/child disappeared')
        write(out / 'receipt.json', dict(status='ALL_REGISTERED_TRAIN_SUPPORTED_INTERVAL_INPUT_EVENTS_READY',
            protocol_sha256=digest(out / 'protocol.json'), jobs=len(completed), versions=completed_views,
            children=completed_children, events=event_count, records=completed,
            reference_coordinates_used_to_generate_events=False, raw_data_or_labels_or_weights_modified=False,
            held_or_test_reference_reads=0, training_updates=0, full_contract_pass=False))
    except BaseException as exc:
        write(out / 'failure.json', dict(error=str(exc), traceback=traceback.format_exc(), completed_jobs=len(completed)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--control-name', required=True)
    main(parser.parse_args())
