"""One registered TRAIN cloud through the preserved point/finite geometry pipeline.

All input fragments, including arcs and unscreened proposals, are retained in a
losslessly compressed shard. No label geometry enters input preparation. The
original run162/run26 checkpoints initialize the same feature computation used
by the best pipeline; no new model fitting is claimed by this preparation tool.
"""
import argparse
import importlib
import shutil
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from weld_pipeline.sjg_local_runtime_904 import read, write, digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'
DATA = ROOT.parent / 'data/G_Data'
WORKSPACE = BASE / 'training_workspace'
SHARDS = BASE / 'input_shards'
GEOMETRY = {
    'merge162': ('merge_expansion_point_features', 'merge'),
    'merge26': ('merge_expansion_run26_point_features', 'merge'),
    'instances': ('rebuild_legacy_instances', 'prepare'),
    'finite': ('rebuild_legacy_finite_support', 'prepare'),
    'surfaces': ('rebuild_run26_learned_surfaces', 'run'),
    'unified': ('rebuild_legacy_unified_candidates', 'run'),
    'union': ('rebuild_legacy_joint_union', 'run'),
}
FRAGMENT_STAGES = ('graphs', 'object_features', 'relative_features', 'observed_features')
STAGES = ('input', 'point_features', 'merge162', 'merge26', 'instances', 'finite', 'surfaces',
          'rank2', 'unified', 'union') + FRAGMENT_STAGES + ('pack',)


def registered_job(job_key):
    registry = read(BASE / 'training_registry.json')
    if registry['split_manifest_sha256'] != digest(BASE / 'split_manifest.json'):
        raise ValueError('Training registry split changed')
    job = next(c for c in registry['cases'] if c['job_key'] == job_key)
    assigned = {e['key']: e for e in read(BASE / 'split_manifest.json')['examples']}
    for version in job['versions']:
        if assigned[version['key']]['split'] != 'train' or version['family_key'] == 'new_f7b2ad7ae982':
            raise ValueError('Validation/test/protected family cannot enter training preparation')
        path = ROOT / version['label_directory'] / 'receipt.json'
        if digest(path) != version['label_receipt_sha256']:
            raise ValueError('Qualified label binding changed')
    return job


def roots(job):
    work = WORKSPACE / job['job_key']
    return work, work / 'frozen_features', work / 'inputs'


def prepare_input(job):
    from tools.qualify_full_dataset_labels import ply_points
    work, frozen, inputs = roots(job)
    work.mkdir(parents=True, exist_ok=True)
    dest = inputs / job['runtime_key']
    dest.mkdir(parents=True, exist_ok=False)
    source = DATA / job['workpiece_source']
    if digest(source) != job['workpiece_sha256']:
        raise ValueError('Registered workpiece changed')
    provider = job['prior_input_provider']
    if provider:
        prior = ROOT / provider['input_directory']
        if digest(prior / 'receipt.json') != provider['input_receipt_sha256']:
            raise ValueError('Prior XYZ receipt changed')
        receipt = read(prior / 'receipt.json')
        if digest(prior / 'xyz_mm.npy') != receipt['xyz_sha256']:
            raise ValueError('Prior XYZ changed')
        (dest / 'xyz_mm.npy').symlink_to(prior / 'xyz_mm.npy')
    else:
        xyz = ply_points(source)
        if len(xyz) != job['original_points']:
            raise ValueError('Registered workpiece point count changed')
        if shutil.disk_usage(dest).free < 5 * 1024**3 + xyz.nbytes:
            raise RuntimeError('Insufficient temporary disk headroom for XYZ')
        np.save(dest / 'xyz_mm.npy', xyz)
    write(dest / 'receipt.json', dict(role='TRAIN', original_points=job['original_points'], unit='mm',
          coordinate_frame='workpiece_local', source_file=job['workpiece_source'], source_sha256=job['workpiece_sha256'],
          xyz_sha256=digest(dest / 'xyz_mm.npy'), only_XYZ_used=True, coordinate_transform_applied=False,
          input_extraction_GT_reads=0, STEP_reads=0, registry_sha256=digest(BASE / 'training_registry.json')))
    frozen.mkdir(exist_ok=True)


def point_features(job):
    from weld_pipeline.server_completed_point_features import verify_complete_component
    work, frozen, inputs = roots(job)
    key = job['runtime_key']
    blocks = importlib.import_module('tools.rebuild_expansion_input_blocks')
    blocks.DEST = frozen / 'input_blocks'
    blocks.DEST.mkdir(exist_ok=True)
    blocks.prepare(key, input_root=inputs)
    for name in ('infer_legacy_point_features', 'infer_run26_point_features'):
        module = importlib.import_module('tools.' + name)
        module.RESTORE, module.BLOCKS = frozen, blocks.DEST
        component = 'point_features' if name == 'infer_legacy_point_features' else 'run26_point_features'
        if (frozen / component / ('source_' + key) / 'feature_manifest.json').exists():
            model_source = ('weld_pipeline/sjg_material_model_859.py' if name == 'infer_legacy_point_features'
                            else 'weld_pipeline/sjg_ptv3_runtime_807.py')
            binding = verify_complete_component(module, key, model_source)
            write(frozen / (component + '_reuse_binding.json'), binding)
            print(key, 'COMPLETE_POINT_COMPONENT_REUSED', component, binding['checked_blocks'], flush=True)
        else:
            module.run(key, 0)


def geometry(job, stage):
    work, frozen, inputs = roots(job)
    key = job['runtime_key']
    if stage in GEOMETRY:
        name, function = GEOMETRY[stage]
        module = importlib.import_module('tools.' + name)
        module.BASE = frozen
        if stage.startswith('merge'):
            getattr(module, function)(key, input_root=inputs)
        else:
            getattr(module, function)(key)
    elif stage == 'rank2':
        module = importlib.import_module('tools.rebuild_legacy_rank2_candidates')
        module.BASE = frozen
        case = read(frozen / 'run26_merged_point_features' / ('source_' + key) / 'receipt.json')
        case['role'] = 'TRAIN'
        folder = frozen / 'rank2_candidates' / ('source_' + key)
        folder.mkdir(parents=True, exist_ok=False)
        write(folder / 'controls.json', module.controls())
        def progress(*event):
            module.check_resources()
            write(folder / 'progress.json', dict(event=event))
            print(key, *event, flush=True)
        profiles, row = module.generate(case, folder, progress)
        write(folder / 'receipt.json', dict(status='ORIGINAL_RANK2_CANDIDATES_RECONSTRUCTED',
              historical_parity_proven=False, weld_accuracy_measured=False, **row))
    else:
        raise ValueError(stage)


def fragments(job, stage):
    work, frozen, inputs = roots(job)
    module = importlib.import_module('tools.rebuild_legacy_downstream')
    module.BASE = frozen
    module.CASE = frozen / 'legacy_runtime' / ('source_' + job['runtime_key'])
    module.run(job['runtime_key'], stage)


def pack(job):
    from weld_pipeline.server_fixed_fragment_queries import FixedQueries
    provider = job['prior_input_provider']
    if provider and provider['observed_features_complete']:
        source = ROOT / provider['runtime_directory']
    else:
        source = roots(job)[1] / 'legacy_runtime' / ('source_' + job['runtime_key'])
    key = job['runtime_key']
    dest = SHARDS / job['job_key']
    dest.mkdir(parents=True, exist_ok=False)
    inputs = FixedQueries(source, key)
    try:
        # The complete original 152-column feature table is frozen; no learned
        # count gate or positive-reference filtering restricts this input.
        if inputs.raw.shape != (2 * len(inputs.canonical), 152):
            raise ValueError('Original fragment feature table shape mismatch')
        np.savez_compressed(dest / 'fragment_inputs.npz', raw_features=inputs.raw,
                            offsets=inputs.offsets, spans=inputs.spans, canonical=inputs.canonical,
                            representative=inputs.representative)
        with np.load(dest / 'fragment_inputs.npz') as saved:
            for name, array in [('raw_features', inputs.raw), ('offsets', inputs.offsets), ('spans', inputs.spans),
                                ('canonical', inputs.canonical), ('representative', inputs.representative)]:
                if not np.array_equal(saved[name], array):
                    raise ValueError('Compressed shard changed original input values')
        write(dest / 'profiles.json', inputs.profiles)
        kinds = {kind: sum(p['representation']['kind'] == kind for p in inputs.profiles) for kind in ('line', 'arc')}
        # Native predictions may seed later parent training only when they are
        # already available; their absence never removes an input cloud.
        native = source / 'run379' / ('test_' + key) / 'decoded.json'
        if native.exists():
            shutil.copyfile(native, dest / 'prior_native_parents.json')
        freeze = dict(job_key=job['job_key'], runtime_key=key, workpiece_source=job['workpiece_source'],
                      workpiece_sha256=job['workpiece_sha256'], original_points=job['original_points'],
                      input_bindings=inputs.bindings, profiles=len(inputs.profiles), profile_kinds=kinds,
                      canonical_fragments=len(inputs.canonical), fixed_input_queries=inputs.n,
                      lossless_full_fragment_inputs=True, screened_by_model=False,
                      GT_geometry_reads=0, STEP_reads=0, optimizer_updates=0,
                      source_sha256=digest(__file__), registry_sha256=digest(BASE / 'training_registry.json'),
                      files={p.name: digest(p) for p in dest.iterdir() if p.is_file()})
        write(dest / 'input_freeze.json', freeze)
        # Only label references are attached after input freeze. All versions,
        # ordered parents/children, process masks and frames remain available.
        write(dest / 'supervision_versions.json', dict(versions=job['versions'], label_geometry_as_input=False))
        write(dest / 'receipt.json', dict(status='FULL_UNSCREENED_INPUT_SHARD_READY_NOT_TRAINED',
              **freeze, supervision_versions_sha256=digest(dest / 'supervision_versions.json'),
              training_versions=len(job['versions']), families=sorted({v['family_id'] for v in job['versions']})))
        print('FULL_INPUT_SHARD_READY', job['job_key'], 'profiles', freeze['profiles'], kinds,
              'versions', len(job['versions']), flush=True)
    finally:
        inputs.close()


def main(job_key, stage):
    job = registered_job(job_key)
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        if stage == 'input':
            prepare_input(job)
        elif stage == 'point_features':
            point_features(job)
        elif stage in GEOMETRY or stage == 'rank2':
            geometry(job, stage)
        elif stage in FRAGMENT_STAGES:
            fragments(job, stage)
        elif stage == 'pack':
            pack(job)
        else:
            raise ValueError(stage)
    markers = BASE / 'preparation' / job_key / 'stages'
    markers.mkdir(parents=True, exist_ok=True)
    write(markers / (stage + '.json'), dict(status='COMPLETE', job_key=job_key, stage=stage,
          seconds=time.perf_counter() - start, registry_sha256=digest(BASE / 'training_registry.json'),
          source_sha256=digest(__file__), GT_geometry_as_input=False, optimizer_updates=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--stage', required=True, choices=STAGES)
    args = parser.parse_args()
    main(args.job, args.stage)
