"""Register all qualified training versions, deduplicating input computation only.

Every version retains its own label binding and family-balanced training weight.
Byte-identical workpiece PLY files share input computation, not label identity.
Validation/test/64 never enter this registry or model fitting.
"""
from pathlib import Path
import collections
from weld_pipeline.sjg_local_runtime_904 import read, write, digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'
DATA = ROOT.parent / 'data/G_Data'


def existing_sources():
    sources = []
    for inputs, frozen in [
        (ROOT / 'server_rebuild/expansion_v1/inputs', ROOT / 'server_rebuild/expansion_v1/frozen_features'),
        (ROOT / 'server_rebuild/inputs_v1', ROOT / 'server_rebuild/best_baseline_restore')]:
        for path in sorted(inputs.glob('*/receipt.json')):
            receipt = read(path)
            source = receipt.get('source_file')
            if not source or not (path.parent / 'xyz_mm.npy').exists():
                continue
            key = path.parent.name
            runtime = frozen / 'legacy_runtime' / ('source_' + key)
            observed = runtime / ('run344' if key in ('25', '36') else 'run329') / ('source_' + key) / 'receipt.json'
            sources.append(dict(key=key, workpiece_sha256=digest(DATA / source), input_directory=str(path.parent.relative_to(ROOT)),
                                input_receipt_sha256=digest(path), frozen_directory=str(frozen.relative_to(ROOT)),
                                runtime_directory=str(runtime.relative_to(ROOT)), observed_features_complete=observed.exists()))
    return sources


def main():
    manifest = read(BASE / 'split_manifest.json')
    summary = read(BASE / 'qualification_summary.json')
    if summary['split_manifest_sha256'] != digest(BASE / 'split_manifest.json'):
        raise ValueError('Qualification/split binding changed')
    examples = {r['key']: r for r in manifest['examples']}
    train = [r for r in summary['qualified'] if r['split'] == 'train']
    families = collections.Counter(r['family_id'] for r in train)
    old = existing_sources()
    jobs = {}
    for receipt in train:
        example = examples[receipt['key']]
        if example['split'] != 'train' or example['family_key'] == 'new_f7b2ad7ae982':
            raise ValueError('Held or protected family entered training')
        label = BASE / 'labels' / receipt['key']
        if digest(label / 'targets.json') != receipt['artifacts']['targets.json']:
            raise ValueError('Qualified geometry changed')
        cloud_sha = receipt['source_files'][receipt['workpiece_source']]
        if cloud_sha not in jobs:
            candidates = sorted([r for r in old if r['workpiece_sha256'] == cloud_sha],
                                key=lambda r: (not r['observed_features_complete'], r['key']))
            provider = candidates[0] if candidates else None
            runtime_key = provider['key'] if provider else receipt['key']
            jobs[cloud_sha] = dict(job_key='cloud_' + cloud_sha[:16], runtime_key=runtime_key,
                                  workpiece_source=receipt['workpiece_source'], workpiece_sha256=cloud_sha,
                                  original_points=receipt['workpiece_points'], prior_input_provider=provider, versions=[])
        job = jobs[cloud_sha]
        if job['original_points'] != receipt['workpiece_points']:
            raise ValueError('Byte-identical clouds have inconsistent counts')
        job['versions'].append(dict(key=receipt['key'], family_id=receipt['family_id'], family_key=receipt['family_key'],
                                   directory=receipt['directory'], split='train',
                                   label_directory=str(label.relative_to(ROOT)), label_receipt_sha256=digest(label / 'receipt.json'),
                                   targets_sha256=receipt['artifacts']['targets.json'], children=receipt['children'],
                                   family_equal_version_weight=1 / (len(families) * families[receipt['family_id']])))
    if not abs(sum(v['family_equal_version_weight'] for j in jobs.values() for v in j['versions']) - 1) < 1e-12:
        raise ValueError('Family/version weights do not sum to one')
    # Existing inputs can be packed first; all remaining clouds follow by size.
    rows = sorted(jobs.values(), key=lambda j: (not bool(j['prior_input_provider'] and j['prior_input_provider']['observed_features_complete']),
                                               j['original_points'], j['job_key']))
    registry = dict(schema='ALL_QUALIFIED_TRAINING_VERSIONS_V1', role='TRAIN', cases=rows,
                    qualified_training_versions=len(train), qualified_training_families=len(families), input_cloud_jobs=len(rows),
                    all_qualified_training_versions_registered=True, input_computation_only_deduplicated=True,
                    all_label_versions_retained=True, protected_64_excluded=True, validation_or_test_fit_rows=0,
                    split_manifest_sha256=digest(BASE / 'split_manifest.json'),
                    qualification_summary_sha256=digest(BASE / 'qualification_summary.json'), source_sha256=digest(__file__))
    path = BASE / 'training_registry.json'
    if path.exists():
        raise ValueError('Registry already exists; inspect its bindings before replacement')
    write(path, registry)
    print('REGISTERED_TRAINING', len(train), 'versions', len(families), 'families', len(rows), 'cloud computations',
          sum(bool(j['prior_input_provider'] and j['prior_input_provider']['observed_features_complete']) for j in rows), 'prior complete inputs', flush=True)


if __name__ == '__main__':
    main()
