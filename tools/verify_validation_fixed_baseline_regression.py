"""Check complete baseline ownership and reject partial or substituted inputs."""
import argparse
import copy
from pathlib import Path
import shutil

import torch

from tools.continue_validation_fixed_baseline_regression import collection, SOURCES
from tools.run_partitioned_full_dataset_stage import configure
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents
from weld_pipeline.sjg_local_runtime_904 import read, write, digest

ROOT = Path(__file__).resolve().parents[1]


def main(args):
    base, train, held, modules = configure(args.dataset_name)
    manifest = (ROOT / args.baseline_manifest).resolve(strict=True)
    manifest.relative_to(ROOT)
    declared = collection(ROOT, train, manifest)
    out = base / args.experiment_name
    if Path(args.experiment_name).name != args.experiment_name or args.experiment_name in ('', '.', '..'):
        raise ValueError('A separate control experiment is required')
    out.mkdir(exist_ok=False)
    rejects = {}

    def reject(label, change):
        value = copy.deepcopy(declared)
        change(value)
        path = out / (label + '.json')
        write(path, value)
        try:
            collection(ROOT, train, path)
        except (ValueError, KeyError):
            rejects[label] = True
            return
        raise AssertionError('Malformed baseline collection accepted: ' + label)

    reject('omitted_case', lambda v: v['cases'].pop())
    reject('duplicate_case', lambda v: v['cases'].append(v['cases'][0]))
    reject('held_input', lambda v: v['cases'][0].update(job_key=next(iter(held.jobs))))
    reject('different_raw_input', lambda v: v['cases'][0].update(workpiece_sha256='0'*64))
    reject('different_reference', lambda v: v['cases'][0].update(original_targets_sha256='0'*64))
    reject('different_prediction', lambda v: v['cases'][0].update(baseline_prediction_sha256='0'*64))
    actual = []
    for row in declared['cases']:
        target = read(ROOT / row['original_targets'])
        score = score_native_parents(read(ROOT / row['baseline_prediction']), target, [False] * len(target['parents']))
        assert score['categories'] == read(ROOT / row['baseline_evaluation'])['categories']
        actual.append(dict(original_key=row['original_key'], categories=score['categories']))
    partial = base / 'full_weld_reference_recall_v3/last.pt'
    frozen_partial = out / 'actual_partial_query.pt'
    shutil.copyfile(partial, frozen_partial)
    state = torch.load(frozen_partial, map_location='cpu', weights_only=False)
    try:
        modules['tools.evaluate_full_weld_predictions'].verify_complete_query_fit(
            ROOT, state, digest(frozen_partial), base / 'full_weld_reference_recall_v3/receipt.json')
    except (ValueError, FileNotFoundError):
        rejects['actual_partial_training_not_a_final_model'] = True
    else:
        raise AssertionError('Current incomplete query accepted as a complete final model')
    write(out / 'receipt.json', dict(status='PASS_COMPLETE_ORIGINAL_BASELINE_OWNERSHIP_AND_REAL_PARTIAL_MODEL_REJECTION',
        sources={name:digest(ROOT / name) for name in SOURCES}, manifest_sha256=digest(manifest),
        all_original_cases=len(actual), original_nonempty_geometry_parity=actual, rejection_controls=rejects,
        all_original_inputs_same_sha_as_registered_train=True, optimizer_updates=0,
        final_test_geometry_reads=0, original_train_geometry_reads=len(actual),
        actual_partial_checkpoint_sha256=digest(frozen_partial),
        actual_partial_optimizer_updates=state['optimizer_steps'],
        complete_final_model_regression_not_yet_executed=True,
        model_quality_round_completed=False, independent_generalization_certified=False))
    print('COMPLETE_BASELINE_BINDING_AND_REAL_PARTIAL_MODEL_CONTROLS_PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--baseline-manifest', required=True)
    parser.add_argument('--experiment-name', required=True)
    main(parser.parse_args())
