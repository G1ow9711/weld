"""Register every qualified held version without changing training or splits.

Only input computation is deduplicated. Selection, protected diagnostics and
final testing retain separate roles. This registry does not certify that the
historical holdouts have never influenced development.
"""
from collections import Counter
from pathlib import Path

from tools.prepare_full_training_registry import existing_sources
from weld_pipeline.sjg_local_runtime_904 import digest, read, write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'
ROLES = ('MODEL_SELECTION_ONLY', 'PROTECTED_DIAGNOSTIC_ONLY', 'FINAL_TEST_ONLY')


def role_of(example):
    if example['fit_allowed']:
        raise ValueError('Training examples cannot enter an evaluation registry')
    flags = [example['selection_allowed'], example['diagnostic_only'], example['final_evaluation_allowed']]
    if sum(flags) != 1:
        raise ValueError('Each qualified held example needs exactly one evaluation role')
    return ROLES[flags.index(True)]


def build(manifest, qualification, policy, train, providers, root=ROOT, dataset_base=None):
    dataset_base = Path(dataset_base) if dataset_base is not None else root / 'server_rebuild/full_dataset_v1'
    dataset_base.resolve().relative_to(Path(root).resolve())
    assigned = {row['key']: row for row in manifest['examples']}
    qualified = {row['key']: row for row in qualification['qualified']}
    expected = [row for row in policy['examples'] if row['qualified'] and not row['fit_allowed']]
    family_sizes = Counter((role_of(row), row['family_id']) for row in expected)
    family_counts = Counter(role for role, _ in family_sizes)
    train_clouds = {row['workpiece_sha256'] for row in train['cases']}
    jobs = {}
    versions = set()
    for example in expected:
        key, role = example['key'], role_of(example)
        source, assignment = qualified[key], assigned[key]
        if (assignment['split'] != example['assigned_split'] or source['split'] != assignment['split']
                or source['family_id'] != example['family_id'] or assignment['split'] == 'train'):
            raise ValueError('Held source, policy and assignment disagree')
        label = dataset_base / 'labels' / key
        receipt = read(label / 'receipt.json')
        if receipt != source:
            raise ValueError('Qualified held receipt differs from the qualification summary')
        for name, sha in source['artifacts'].items():
            if Path(name).name != name or digest(label / name) != sha:
                raise ValueError('Qualified held artifact changed')
        cloud_sha = source['source_files'][source['workpiece_source']]
        if cloud_sha in train_clouds:
            raise ValueError('A workpiece byte identity crosses training and evaluation')
        if cloud_sha not in jobs:
            hits = sorted([p for p in providers if p['workpiece_sha256'] == cloud_sha],
                          key=lambda p: (not p['observed_features_complete'], p['key']))
            provider = hits[0] if hits else None
            jobs[cloud_sha] = dict(job_key='eval_cloud_' + cloud_sha[:16],
                runtime_key=provider['key'] if provider else key,
                workpiece_source=source['workpiece_source'], workpiece_sha256=cloud_sha,
                original_points=source['workpiece_points'], prior_input_provider=provider,
                fit_allowed=False, versions=[])
        job = jobs[cloud_sha]
        if job['original_points'] != source['workpiece_points'] or key in versions:
            raise ValueError('Duplicate held version or inconsistent input count')
        versions.add(key)
        job['versions'].append(dict(key=key, family_id=source['family_id'], family_key=source['family_key'],
            directory=source['directory'], split=source['split'], role=role, fit_allowed=False,
            label_directory=str(label.relative_to(root)), label_receipt_sha256=digest(label / 'receipt.json'),
            targets_sha256=source['artifacts']['targets.json'], children=source['children'],
            family_equal_version_evaluation_weight=1 / (family_counts[role] * family_sizes[role, source['family_id']])))
    expected_keys = {row['key'] for row in qualification['qualified'] if row['split'] != 'train'}
    if versions != expected_keys:
        raise ValueError('A qualified held version is missing or unexpectedly registered')
    for role in ROLES:
        rows = [v for job in jobs.values() for v in job['versions'] if v['role'] == role]
        if rows and abs(sum(v['family_equal_version_evaluation_weight'] for v in rows) - 1) > 1e-12:
            raise ValueError('Role-specific family/version evaluation mass changed')
    return dict(schema='ALL_QUALIFIED_HELD_INPUTS_AND_SEPARATE_EVALUATION_ROLES_V1',
        role='INPUT_PREPARATION_AND_EVALUATION_ONLY', fit_allowed=False,
        cases=sorted(jobs.values(), key=lambda job: (
            min(ROLES.index(v['role']) for v in job['versions']), job['original_points'], job['job_key'])),
        qualified_evaluation_versions=len(versions), input_cloud_jobs=len(jobs),
        role_version_counts=dict(Counter(role_of(row) for row in expected)), role_family_counts=dict(family_counts),
        all_qualified_held_versions_registered=True, all_label_versions_retained=True,
        input_computation_only_deduplicated=True, cross_train_workpiece_byte_duplicates=[],
        test_for_final_evaluation_only=True, diagnostic_for_selection=False,
        independent_generalization_certified=False,
        nonbyte_identity_and_complete_history_exposure_audits_still_required=True)


def main():
    destination = BASE / 'evaluation_registry.json'
    if destination.exists():
        raise ValueError('Evaluation registry already exists; do not overwrite a frozen registration')
    paths = {name: BASE / name for name in ('split_manifest.json', 'qualification_summary.json',
                                            'model_use_policy.json', 'training_registry.json')}
    values = {name: read(path) for name, path in paths.items()}
    split_sha = digest(paths['split_manifest.json'])
    for name in ('qualification_summary.json', 'model_use_policy.json', 'training_registry.json'):
        if values[name]['split_manifest_sha256'] != split_sha:
            raise ValueError('Evaluation input sources have a different frozen split')
    value = build(values['split_manifest.json'], values['qualification_summary.json'],
                  values['model_use_policy.json'], values['training_registry.json'], existing_sources(), dataset_base=BASE)
    value.update(source_bindings={name: digest(path) for name, path in paths.items()}, source_sha256=digest(__file__))
    write(destination, value)
    print('ALL_QUALIFIED_HELD_INPUTS_REGISTERED', value['qualified_evaluation_versions'],
          value['input_cloud_jobs'], value['role_version_counts'], 'fit_allowed', False, flush=True)


if __name__ == '__main__':
    main()
