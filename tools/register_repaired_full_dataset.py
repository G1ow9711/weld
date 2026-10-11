"""Register every requalified version in the corrected family partition."""
import argparse
from pathlib import Path
import time
import traceback

from tools import prepare_full_training_registry as train
from tools import prepare_full_model_use_policy as policy
from tools import audit_full_training_version_compatibility as compatibility
from tools import prepare_full_evaluation_registry as held
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset, PartitionedEvaluationShardDataset
from weld_pipeline.sjg_local_runtime_904 import digest, read, write

ROOT = train.ROOT


def main(args):
    if Path(args.dataset_name).name != args.dataset_name or args.dataset_name in ('', '.', '..', 'full_dataset_v1'):
        raise ValueError('A repaired independent dataset directory is required')
    base = ROOT / 'server_rebuild' / args.dataset_name
    proof = read(base/'qualification_repair/receipt.json')
    if (proof['status'] != 'ALL_REPAIRED_VERSIONS_REQUALIFIED_WITH_GEOMETRY_ATTRIBUTES_AND_FRAMES_PRESERVED'
            or proof['repaired_qualification_sha256'] != digest(base/'qualification_summary.json')
            or proof['sources'] != {name:digest(ROOT/name) for name in proof['sources']}):
        raise ValueError('Complete original-label requalification is required')
    out = base/'registration_repair';out.mkdir(exist_ok=False)
    started = time.perf_counter()
    sources = {name:digest(ROOT/name) for name in ['tools/register_repaired_full_dataset.py',
        'tools/prepare_full_training_registry.py', 'tools/prepare_full_model_use_policy.py',
        'tools/audit_full_training_version_compatibility.py', 'tools/prepare_full_evaluation_registry.py',
        'weld_pipeline/server_partitioned_full_dataset.py']}
    write(out/'protocol.json', dict(sources=sources, dataset_directory=str(base.relative_to(ROOT)),
        qualification_repair_receipt_sha256=digest(base/'qualification_repair/receipt.json'),
        model_fit_calls=0, model_reference_scores=0, old_dataset_registration_changed=False))
    try:
        for module in [train, policy, compatibility, held]:
            module.BASE = base
            module.main()
        training, evaluation = PartitionedFullShardDataset(ROOT, base), PartitionedEvaluationShardDataset(ROOT, base)
        qualified = {row['key'] for row in read(base/'qualification_summary.json')['qualified']}
        if set(training.versions) & set(evaluation.versions) or set(training.versions) | set(evaluation.versions) != qualified:
            raise ValueError('Qualified versions were omitted or their model-use roles overlap')
        if set(training.families) & {v['family_id'] for _,v in evaluation.versions.values()}:
            raise ValueError('A corrected family still crosses training and held roles')
        rejected = 0
        for key, (_,version) in evaluation.versions.items():
            if version['role'] == 'FINAL_TEST_ONLY':
                try:
                    evaluation.reference(key, base/'nonexistent_unfrozen_prediction')
                except ValueError as exc:
                    if 'fixed by validation' not in str(exc): raise
                    rejected += 1
                else:
                    raise ValueError('An unfixed model read a final-test reference')
        if sources != {name:digest(ROOT/name) for name in sources}:
            raise ValueError('Registration implementation changed')
        write(out/'receipt.json', dict(status='ALL_REPAIRED_TRAIN_AND_HELD_VERSIONS_REGISTERED',
            sources=sources, protocol_sha256=digest(out/'protocol.json'),
            qualified_versions=len(qualified), training_versions=len(training.versions),
            training_families=len(training.families), training_input_jobs=len(training.jobs),
            held_versions=len(evaluation.versions), held_input_jobs=len(evaluation.jobs),
            held_role_counts=evaluation.registry['role_version_counts'],
            all_qualified_versions_accounted=True, corrected_families_do_not_cross_train_and_held=True,
            rejected_unfixed_test_reference_reads=rejected, actual_test_reference_reads=0,
            bindings={name:digest(base/name) for name in ['split_manifest.json','qualification_summary.json',
                'model_use_policy.json','training_registry.json','training_version_compatibility.json','evaluation_registry.json']},
            model_fit_calls=0, original_data_labels_models_or_cached_inputs_changed=False,
            independent_generalization_certified=False, seconds=time.perf_counter()-started))
        print('ALL_CORRECTED_MODEL_USE_REGISTRIES_READY',len(training.versions),len(evaluation.versions),
              evaluation.registry['role_version_counts'],flush=True)
    except Exception:
        write(out/'failure.json',dict(traceback=traceback.format_exc(),unix_time=time.time()))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset-name',required=True)
    main(parser.parse_args())
