"""One held XYZ cloud through the same frozen point and fragment computation.

The preserved geometry helpers run in an isolated worker with a distinct
workspace. Some historical receipts use a literal TRAIN routing tag; those
tags do not grant fitting permission. Authoritative registration, raw input
receipts and complete shard receipts always mark these inputs as held.
"""
import argparse
import fcntl
import gc
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from tools import prepare_full_training_case as core
from weld_pipeline.server_full_evaluation_dataset import EvaluationShardDataset
from weld_pipeline.sjg_local_runtime_904 import digest, read, write

ROOT, BASE, DATA = core.ROOT, core.BASE, core.DATA
WORKSPACE = BASE / 'evaluation_workspace'
SHARDS = BASE / 'evaluation_input_shards'
OUT = BASE / 'evaluation_preparation_v1'
STAGES = core.STAGES


def prepare_input(job, registry_sha):
    from tools.qualify_full_dataset_labels import ply_points
    dest = WORKSPACE / job['job_key'] / 'inputs' / job['runtime_key']
    dest.mkdir(parents=True, exist_ok=True)
    (WORKSPACE / job['job_key'] / 'frozen_features').mkdir(exist_ok=True)
    receipt_file = dest / 'receipt.json'
    if receipt_file.exists():
        receipt = read(receipt_file)
        if (receipt['registry_sha256'] != registry_sha or receipt['fit_allowed']
                or digest(dest / 'xyz_mm.npy') != receipt['xyz_sha256']):
            raise ValueError('Existing held XYZ receipt changed')
        return
    source = DATA / job['workpiece_source']
    if digest(source) != job['workpiece_sha256']:
        raise ValueError('Registered held workpiece changed')
    provider = job['prior_input_provider']
    if provider:
        previous = ROOT / provider['input_directory']
        old = read(previous / 'receipt.json')
        if (digest(previous / 'receipt.json') != provider['input_receipt_sha256']
                or digest(previous / 'xyz_mm.npy') != old['xyz_sha256']):
            raise ValueError('Prior held XYZ input changed')
        (dest / 'xyz_mm.npy').symlink_to(previous / 'xyz_mm.npy')
    else:
        xyz = ply_points(source)
        if xyz.shape != (job['original_points'], 3):
            raise ValueError('Held source point count changed')
        if shutil.disk_usage(dest).free < 5 * 1024**3 + xyz.nbytes:
            raise RuntimeError('Insufficient temporary headroom for held XYZ')
        staged = dest / 'xyz_mm.incomplete.npy'
        np.save(staged, xyz)
        with staged.open('rb') as stream:
            if not np.array_equal(np.load(stream, allow_pickle=False), xyz):
                raise ValueError('Saved held XYZ changed')
        os.replace(staged, dest / 'xyz_mm.npy')
    write(receipt_file, dict(role='HELD_EVALUATION_INPUT_ONLY', fit_allowed=False,
        evaluation_roles=sorted({v['role'] for v in job['versions']}),
        original_points=job['original_points'], unit='mm', coordinate_frame='workpiece_local',
        source_file=job['workpiece_source'], source_sha256=job['workpiece_sha256'],
        xyz_sha256=digest(dest / 'xyz_mm.npy'), only_XYZ_used=True,
        coordinate_transform_applied=False, input_extraction_GT_reads=0, STEP_reads=0,
        registry_sha256=registry_sha, source_code_sha256=digest(__file__)))
    print('HELD_XYZ_READY', job['job_key'], job['original_points'], flush=True)


def pack(job, registry_sha):
    from weld_pipeline.server_fixed_fragment_queries import FixedQueries
    provider = job['prior_input_provider']
    source = (ROOT / provider['runtime_directory'] if provider and provider['observed_features_complete']
              else WORKSPACE / job['job_key'] / 'frozen_features/legacy_runtime' / ('source_' + job['runtime_key']))
    destination = SHARDS / job['job_key']
    if destination.exists():
        EvaluationShardDataset(ROOT).input_case(job['job_key'])
        return
    staging = Path(tempfile.mkdtemp(prefix=job['job_key'] + '_incomplete_', dir=SHARDS))
    inputs = FixedQueries(source, job['runtime_key'])
    try:
        arrays = dict(raw_features=inputs.raw, offsets=inputs.offsets, spans=inputs.spans,
                      canonical=inputs.canonical, representative=inputs.representative)
        if inputs.raw.shape != (2 * len(inputs.canonical), 152):
            raise ValueError('Held original fragment feature shape changed')
        np.savez_compressed(staging / 'fragment_inputs.npz', **arrays)
        with np.load(staging / 'fragment_inputs.npz', allow_pickle=False) as saved:
            if any(not np.array_equal(saved[name], array) for name, array in arrays.items()):
                raise ValueError('Held compressed shard changed original values')
        write(staging / 'profiles.json', inputs.profiles)
        freeze = dict(job_key=job['job_key'], runtime_key=job['runtime_key'], fit_allowed=False,
            evaluation_roles=sorted({v['role'] for v in job['versions']}),
            workpiece_source=job['workpiece_source'], workpiece_sha256=job['workpiece_sha256'],
            original_points=job['original_points'], input_bindings=inputs.bindings,
            profiles=len(inputs.profiles), profile_kinds={kind: sum(
                p['representation']['kind'] == kind for p in inputs.profiles) for kind in ('line', 'arc')},
            canonical_fragments=len(inputs.canonical), fixed_input_queries=inputs.n,
            lossless_full_fragment_inputs=True, screened_by_model=False,
            GT_geometry_reads=0, STEP_reads=0, optimizer_updates=0,
            legacy_pipeline_role_tags_do_not_authorize_fit=True,
            source_sha256=digest(__file__), registry_sha256=registry_sha,
            files={p.name: digest(p) for p in staging.iterdir() if p.is_file()})
        write(staging / 'input_freeze.json', freeze)
        write(staging / 'evaluation_versions.json', dict(versions=job['versions'], label_geometry_as_input=False))
        write(staging / 'receipt.json', dict(status='FULL_HELD_INPUT_SHARD_READY_NOT_MODEL_EVALUATED',
            **freeze, evaluation_versions_sha256=digest(staging / 'evaluation_versions.json'),
            evaluation_versions=len(job['versions']), families=sorted({v['family_id'] for v in job['versions']})))
        staging.rename(destination)
        EvaluationShardDataset(ROOT).input_case(job['job_key'])
        print('FULL_HELD_SHARD_READY', job['job_key'], freeze['profiles'], freeze['profile_kinds'], flush=True)
    finally:
        inputs.close()
        gc.collect()


def main(job_key, stage):
    dataset = EvaluationShardDataset(ROOT)
    job = dataset.jobs[job_key]
    if not WORKSPACE.is_symlink() or not SHARDS.is_symlink():
        raise ValueError('Held worker requires explicit task-owned temporary storage')
    for link, prefix in [(WORKSPACE, 'weld-full-evaluation-work-'), (SHARDS, 'weld-full-evaluation-shards-')]:
        target = link.resolve(strict=True)
        if target.parent != Path('/tmp') or not target.name.startswith(prefix):
            raise ValueError('Unexpected held input storage owner')
    protocol = read(OUT / 'protocol.json')
    if protocol['registry_sha256'] != dataset.registry_sha256:
        raise ValueError('Held controller registration changed')
    for name, sha in protocol['source_bindings'].items():
        if digest(ROOT / name) != sha:
            raise ValueError('Held preparation source changed while the controller is live')
    job_out = OUT / job_key
    job_out.mkdir(exist_ok=True)
    with (job_out / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.perf_counter()
        # This worker owns its Python process. The shared pure input helpers
        # receive only a distinct workspace; their TRAIN registry entry points
        # and label-attaching pack function are never called.
        core.WORKSPACE = WORKSPACE
        with threadpool_limits(limits=2):
            if stage == 'input':
                prepare_input(job, dataset.registry_sha256)
            elif stage == 'point_features':
                core.point_features(job)
            elif stage in core.GEOMETRY or stage == 'rank2':
                core.geometry(job, stage)
            elif stage in core.FRAGMENT_STAGES:
                core.fragments(job, stage)
            elif stage == 'pack':
                pack(job, dataset.registry_sha256)
            else:
                raise ValueError('Unknown held input stage')
        markers = job_out / 'stages'
        markers.mkdir(exist_ok=True)
        write(markers / (stage + '.json'), dict(status='COMPLETE_HELD_INPUT_STAGE', job_key=job_key,
            stage=stage, seconds=time.perf_counter() - started, registry_sha256=dataset.registry_sha256,
            source_sha256=digest(__file__), shared_input_helper_sha256=digest(core.__file__),
            GT_geometry_as_input=False, optimizer_updates=0, fit_allowed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--stage', required=True, choices=STAGES)
    args = parser.parse_args()
    main(args.job, args.stage)
