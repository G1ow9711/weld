"""Run the original complete native branch from actual complete shard inputs.

Complete predictions are frozen before actual TRAIN reference scoring. Exact
retained-input parity is a migration control, not an independent model result
or execution of the additional run446 incremental branch.
"""
import argparse
from pathlib import Path
import tempfile
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.server_native_shard_input import load_native_input, load_frozen_native_models
from weld_pipeline.server_native_parent_runtime import FrozenNativeSelector, complete_native_parents
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents
from weld_pipeline.sjg_local_runtime_904 import digest, read, write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'
SOURCE_NAMES = ['tools/verify_complete_native_parent_runtime.py',
    'weld_pipeline/server_native_parent_runtime.py', 'weld_pipeline/server_native_shard_input.py',
    'weld_pipeline/sjg_interval_forest_runtime_1129.py', 'weld_pipeline/sjg_relative_forest_runtime_1139.py',
    'weld_pipeline/sjg_bounded_gap_runtime_1164.py', 'weld_pipeline/sjg_compact_resources_1169.py',
    'weld_pipeline/sjg_sparse_path_solver_1168.py', 'weld_pipeline/sjg_joint_forest_infer_1111.py',
    'weld_pipeline/sjg_parent_set_features_1160.py', 'weld_pipeline/sjg_precision_residual_1225.py',
    'weld_pipeline/sjg_incompatibility_head_1216.py', 'weld_pipeline/sjg_duplicate_head_1209.py']


def main(job_key, experiment_name):
    if not experiment_name or Path(experiment_name).name != experiment_name or experiment_name in ('.', '..'):
        raise ValueError('An explicit new local control name is required')
    out = BASE / experiment_name
    out.mkdir(exist_ok=False)
    start = time.perf_counter()
    try:
        dataset = FullShardDataset(ROOT)
        job = dataset.jobs[job_key]
        provider = job['prior_input_provider']
        if not provider or not provider['observed_features_complete']:
            raise ValueError('Complete migration parity requires a retained actual old input')
        source = ROOT / provider['runtime_directory'] / 'run379' / ('test_' + job['runtime_key'])
        case = load_native_input(dataset, job_key, BASE / 'legacy_surface_context_train_v1')
        models, upstream_binding = load_frozen_native_models(ROOT)
        selector = FrozenNativeSelector(ROOT)
        protocol = dict(scope='ACTUAL_ORIGINAL_NATIVE_COMPLETE_BRANCH_MIGRATION_CONTROL', job_key=job_key,
            registry_sha256=dataset.registry_sha256, complete_input_receipt_sha256=digest(dataset.shards / job_key / 'receipt.json'),
            genuine_context_receipt_sha256=case['preserved_context_receipt_sha256'],
            native_upstream_checkpoint=upstream_binding, native_selector_bindings=selector.bindings,
            native_selector_training_receipt_sha256=selector.training_receipt_sha256,
            sources={name:digest(ROOT / name) for name in SOURCE_NAMES}, optimizer_updates=0,
            GT_runtime_reads=0, STEP_reads=0, incremental_run446_branch_executed=False,
            independent_model_evaluations=0, full_contract_pass=False)
        write(out / 'protocol.json', protocol)
        def progress(name, **event):
            write(out / 'progress.json', dict(stage=name, **event))
            print('ACTUAL_ORIGINAL_NATIVE_RUNTIME', name, event, flush=True)
        with tempfile.TemporaryDirectory(prefix='weld-native-parent-control-', dir='/tmp') as tmp:
            with threadpool_limits(limits=2):
                candidates, unary, metadata, graph_evidence = complete_native_parents(case, models,
                    Path(tmp) / 'graph', progress=progress)
                final, selection = selector.select(candidates, unary, metadata)
        write(out / 'candidate_decoded.json', candidates)
        write(out / 'parent_metadata.json', metadata)
        np.save(out / 'unary.npy', unary)
        write(out / 'native_graph_evidence.json', graph_evidence)
        write(out / 'final_decoded.json', final)
        np.savez(out / 'precision_scores.npz', probability=selection.pop('probability'),
                 deduction=selection.pop('deduction'))
        write(out / 'prediction_freeze.json', dict(selection, protocol_sha256=digest(out / 'protocol.json'),
            candidate_geometry_sha256=digest(out / 'candidate_decoded.json'),
            selected_geometry_sha256=digest(out / 'final_decoded.json'),
            native_graph_evidence_sha256=digest(out / 'native_graph_evidence.json'),
            reference_geometry_reads_before_freeze=0, scope='Complete original native branch only'))
        actual, old_candidates = read(out / 'candidate_decoded.json'), read(source / 'decoded.json')
        for key in ('parents', 'invalid_predictions', 'total_prediction_children'):
            if actual[key] != old_candidates[key]:
                raise AssertionError('Complete freshly decoded native candidate geometry changed')
        if not np.array_equal(unary, np.load(source / 'unary.npy', allow_pickle=False)):
            raise AssertionError('Genuine original native 160-column parent features changed')
        if read(out / 'parent_metadata.json') != read(source / 'parent_metadata.json'):
            raise AssertionError('Original parent relation metadata changed')
        old_final = read(source / 'final_decoded.json')
        actual_final = read(out / 'final_decoded.json')
        for key in ('parents', 'invalid_predictions', 'total_prediction_children'):
            if actual_final[key] != old_final[key]:
                raise AssertionError('Complete original native selection output changed')
        if selection['selected_candidate_indices'] != read(source / 'final_prediction.json')['selected_candidate_indices']:
            raise AssertionError('Original native selected parent indices changed')
        with np.load(source / 'precision_scores.npz', allow_pickle=False) as old:
            with np.load(out / 'precision_scores.npz', allow_pickle=False) as new:
                for name in old.files:
                    if not np.array_equal(old[name], new[name]):
                        raise AssertionError('Original native precision forest prediction changed')
        # The actual control output is already frozen and authenticated. Only
        # now are the approved TRAIN reference versions opened for diagnostics.
        evaluations = []
        for version in job['versions']:
            supervision = dataset.supervision(version['key'])
            score = score_native_parents(actual_final, supervision.target)
            evaluations.append(dict(version_key=version['key'], score=score,
                label_receipt_sha256=version['label_receipt_sha256'], role='TRAIN_DIAGNOSTIC_NOT_MODEL_SELECTION'))
        result = dict(status='PASS_ACTUAL_COMPLETE_NATIVE_BRANCH_PARITY_NOT_COMPLETE_RUN446',
            job_key=job_key, complete_candidate_geometry_equal=True, all_parent_unary_rows_bitwise_equal=True,
            all_relation_metadata_equal=True, complete_final_native_output_equal=True,
            original_precision_probability_and_deduction_bitwise_equal=True,
            selected_candidate_indices=selection['selected_candidate_indices'],
            actual_output_parents=len(final['parents']), actual_output_children=final['total_prediction_children'],
            complete_reference_versions_evaluated=len(evaluations), evaluations=evaluations,
            actual_train_output_not_independent_accuracy=True, optimizer_updates=0,
            incremental_run446_branch_executed=False, original_526_column_validity_head_executed=False,
            protocol_sha256=digest(out / 'protocol.json'), prediction_freeze_sha256=digest(out / 'prediction_freeze.json'),
            source_reconstructed_native_prediction_sha256=digest(source / 'final_decoded.json'),
            independent_model_evaluations=0, full_contract_pass=False, seconds=time.perf_counter() - start)
        write(out / 'receipt.json', result)
        print('ACTUAL_COMPLETE_NATIVE_BRANCH_PARITY_PASS', result['actual_output_children'],
              [v['score']['categories']['overall'] for v in evaluations], flush=True)
    except Exception:
        write(out / 'failure.json', dict(status='NATIVE_COMPLETE_CONTROL_INCOMPLETE',
            traceback=traceback.format_exc(), seconds=time.perf_counter() - start))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-key', default='cloud_c39d9a8e68d4afe4')
    parser.add_argument('--experiment-name', default='full_native_parent_runtime_control_v1')
    args = parser.parse_args()
    main(args.job_key, args.experiment_name)
