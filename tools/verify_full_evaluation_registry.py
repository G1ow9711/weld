"""Verify actual all-version held registration and separation from fitting.

No held predictions are scored. Negative checks use real held and training
identities so admission cannot silently follow a historical TRAIN routing tag.
"""
from collections import Counter
from pathlib import Path

from weld_pipeline.server_full_evaluation_dataset import EvaluationShardDataset, ROLES
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.sjg_local_runtime_904 import digest, read, write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'


def main():
    held, train = EvaluationShardDataset(ROOT), FullShardDataset(ROOT)
    q = read(BASE / 'qualification_summary.json')
    expected = {r['key'] for r in q['qualified'] if r['split'] != 'train'}
    assert set(held.versions) == expected
    assert not set(held.versions) & set(train.versions)
    train_families = set(train.families)
    held_families = {v['family_id'] for _, v in held.versions.values()}
    assert not train_families & held_families
    assert not hasattr(held, 'epoch_plan') and not hasattr(held, 'supervision')
    role_counts = Counter(v['role'] for _, v in held.versions.values())
    for role in ROLES:
        assert abs(sum(v['family_equal_version_evaluation_weight'] for _, v in held.versions.values()
                       if v['role'] == role) - 1) < 1e-12
    protected = [v for _, v in held.versions.values() if v['family_key'] == 'new_f7b2ad7ae982']
    assert protected and all(v['role'] == 'PROTECTED_DIAGNOSTIC_ONLY' for v in protected)
    rejected_test_reads = 0
    for key, (_, version) in held.versions.items():
        if version['role'] != 'FINAL_TEST_ONLY':
            continue
        try:
            held.reference(key, BASE / 'nonexistent_unfrozen_prediction')
        except ValueError as exc:
            assert 'fixed by validation' in str(exc)
            rejected_test_reads += 1
        else:
            raise AssertionError('An unfixed model opened a final-test reference')
    # Attempt to route an actual ready TRAIN shard into the held reader. The
    # registry check must reject it before the complete feature arrays load.
    ready_train = train.ready_jobs()
    assert ready_train, 'An actual ready TRAIN input is required for this control'
    key = ready_train[0]
    held.jobs[key] = train.jobs[key]
    held.shards = train.shards
    try:
        held.input_case(key)
    except ValueError as exc:
        assert 'registration binding' in str(exc)
    else:
        raise AssertionError('A TRAIN receipt was admitted as held input')
    destination = BASE / 'full_evaluation_registration_validation_v1.json'
    if destination.exists():
        raise ValueError('An immutable held registration validation already exists')
    result = dict(status='PASS_COMPLETE_HELD_REGISTRATION_AND_ROLE_ADMISSION_NOT_MODEL_ACCURACY',
        held_versions=len(expected), held_cloud_jobs=held.registry['input_cloud_jobs'],
        held_families=len(held_families), role_version_counts=dict(role_counts),
        training_versions=len(train.versions), qualified_versions=len(q['qualified']),
        all_qualified_versions_accounted=len(train.versions) + len(expected) == len(q['qualified']),
        protected_diagnostic_versions=len(protected), rejected_unfixed_final_test_reference_reads=rejected_test_reads,
        actual_train_shard_rejected_by_held_reader=True, no_held_training_api=True,
        complete_features_or_runtime_parity_not_yet_checked=True, fit_calls=0, predictions_scored=0,
        independence_certified=False, evaluation_registry_sha256=digest(BASE / 'evaluation_registry.json'),
        training_registry_sha256=digest(BASE / 'training_registry.json'),
        source_bindings={name: digest(ROOT / name) for name in [
            'tools/verify_full_evaluation_registry.py', 'tools/prepare_full_evaluation_registry.py',
            'weld_pipeline/server_full_evaluation_dataset.py']})
    write(destination, result)
    print(result['status'], len(expected), held.registry['input_cloud_jobs'], dict(role_counts), flush=True)


if __name__ == '__main__':
    main()
