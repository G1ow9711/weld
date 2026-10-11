"""Compare final regression outputs with the exact original best reference view.

Consumes the same already frozen final-model predictions. Original historical
references are an additional TRAIN regression view, not replacement labels or
a model-selection criterion. No prediction or fitting is performed here.
"""
import argparse
from pathlib import Path
import time
import traceback
import os

from tools.continue_validation_fixed_baseline_regression import (
    process_identity, name, atomic, SOURCES as REGRESSION_SOURCES)
from tools.verify_original_best_frozen_replay import SOURCES as REPLAY_SOURCES
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents
from weld_pipeline.sjg_local_runtime_904 import read, digest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = list(dict.fromkeys(['tools/continue_original_best_reference_comparison.py',
                            *REGRESSION_SOURCES, *REPLAY_SOURCES]))


def main(args):
    base = ROOT / 'server_rebuild' / name(args.dataset_name)
    regression = base / name(args.regression_name)
    replay = (ROOT / args.original_replay).resolve(strict=True)
    replay.relative_to(ROOT)
    complete = read(replay / 'receipt.json')
    if (complete['status'] != 'ALL_ORIGINAL_FROZEN_OUTPUT_AND_FROZEN_INPUT_REPLAYS_ASSESSED_NOT_RAW_REINFERENCE'
            or not complete['all_fresh_outputs_complete_canonical_json_identical']
            or complete['optimizer_updates'] != 0 or complete['raw_xyz_reinference_executed']):
        raise ValueError('Complete unchanged original frozen-input inference evidence is required')
    identity = process_identity(args.regression_pid)
    if ('tools.continue_validation_fixed_baseline_regression' not in identity['argv']
            or args.regression_name not in identity['argv']):
        raise ValueError('The existing actual full-model regression process must own this run')
    if args.poll_seconds < 5:
        raise ValueError('A conservative explicit wait interval is required')
    out = base / name(args.experiment_name)
    out.mkdir(exist_ok=False)
    protocol = dict(status='WAIT_SAME_FINAL_MODEL_OUTPUTS_THEN_ORIGINAL_BEST_REFERENCE_COMPARISON',
        sources={item:digest(ROOT / item) for item in SOURCES},
        regression_process=identity, regression_protocol_sha256=digest(regression / 'protocol.json'),
        original_replay_receipt_sha256=digest(replay / 'receipt.json'),
        original_replay_protocol_sha256=digest(replay / 'protocol.json'),
        required_original_cases=complete['required_cases'],
        original_best_baseline_categories=complete['totals']['uploaded_on_original'],
        current_parsed_labels_unchanged=True, historical_references_used_for_selection=False,
        model_fit_calls=0, model_predictions_executed=0, full_contract_pass=False)
    atomic(out / 'protocol.json', protocol)

    def verify():
        prior = read(regression / 'protocol.json')
        if (protocol['sources'] != {item:digest(ROOT / item) for item in SOURCES}
                or prior['sources'] != {item:digest(ROOT / item) for item in prior['sources']}
                or digest(regression / 'protocol.json') != protocol['regression_protocol_sha256']
                or digest(replay / 'receipt.json') != protocol['original_replay_receipt_sha256']
                or digest(replay / 'protocol.json') != protocol['original_replay_protocol_sha256']):
            raise ValueError('Original-best comparison or final regression bindings changed')

    try:
        while not (regression / 'receipt.json').exists():
            verify()
            if (regression / 'failure.json').exists():
                raise RuntimeError('The existing complete final-model regression failed')
            if process_identity(args.regression_pid) != identity:
                raise RuntimeError('The actual final regression process changed before completion')
            atomic(out / 'progress.json', dict(status='WAITING_SAME_COMPLETE_FINAL_MODEL_REGRESSION',
                controller_pid=os.getpid(), regression_pid=args.regression_pid,
                model_fit_calls=0, model_predictions_executed=0, original_reference_geometry_reads=0,
                unix_time=time.time()))
            time.sleep(args.poll_seconds)
        verify()
        summary = read(regression / 'receipt.json')
        fixed = read(regression / 'fixed_model_binding.json')
        if (summary['status'] != 'SAME_VALIDATION_FIXED_MODEL_COMPLETE_ORIGINAL_TRAIN_BASELINE_REGRESSION'
                or summary['required_original_cases'] != protocol['required_original_cases']
                or summary['completed_original_cases'] != len(protocol['required_original_cases'])
                or summary['fixed_model_binding_sha256'] != digest(regression / 'fixed_model_binding.json')
                or summary['baseline_used_for_model_selection'] or summary['fit_calls'] != 0):
            raise ValueError('All original inputs must share the verified complete final model pair')
        rows = {}
        for item in summary['job_records']:
            path = (ROOT / item['path']).resolve(strict=True)
            path.relative_to(regression.resolve())
            if digest(path) != item['sha256']:
                raise ValueError('Final-model regression record changed')
            row = read(path)
            if row['original_key'] in rows:
                raise ValueError('Duplicate final-model original input')
            for artifact, sha in row['artifacts'].items():
                if Path(artifact).name != artifact or digest(path.parent / artifact) != sha:
                    raise ValueError('The already frozen final prediction changed')
            if (row['query_checkpoint_sha256'] != fixed['query_checkpoint_sha256']
                    or row['selector_checkpoint_sha256'] != fixed['selector_checkpoint_sha256']
                    or not row['predictions_frozen_before_original_geometry_reference_reads']):
                raise ValueError('All comparisons must consume the same frozen final model predictions')
            rows[row['original_key']] = path.parent
        if set(rows) != set(protocol['required_original_cases']):
            raise ValueError('Final original-regression input collection is incomplete')
        from tools.prepare_full_predicted_parent_pool import load_gzip
        from weld_pipeline.server_predicted_parent_pool import decoded_subset
        from tools.verify_original_best_frozen_replay import accumulate
        totals, assessments = {}, []
        for original in complete['records']:
            p = (ROOT / original['path']).resolve(strict=True)
            p.relative_to(replay)
            if digest(p) != original['sha256']:
                raise ValueError('Original frozen-input replay record changed')
            proof = read(p)
            target_path = (ROOT / proof['original_target_file']).resolve(strict=True)
            target_path.relative_to(ROOT)
            if digest(target_path) != proof['original_target_sha256']:
                raise ValueError('Exact original historical reference changed')
            key = original['case']
            parents = load_gzip(rows[key] / 'selected_parents.json.gz')
            decoded = decoded_subset(parents, range(len(parents)))
            target = read(target_path)
            score = score_native_parents(decoded, target, [False] * len(target['parents']))
            accumulate(totals, score)
            atomic(out / (key + '_assessment.json'), dict(case=key,
                exact_original_reference_sha256=proof['original_target_sha256'],
                same_final_prediction_sha256=digest(rows[key] / 'selected_parents.json.gz'),
                score=score, comparison_is_train_regression_not_independent_test=True,
                model_fit_calls=0, model_predictions_executed=0, full_contract_pass=False))
            assessments.append(dict(case=key, path=str((out / (key + '_assessment.json')).relative_to(ROOT)),
                sha256=digest(out / (key + '_assessment.json'))))
        verify()
        if totals['overall']['G'] != protocol['original_best_baseline_categories']['overall']['G']:
            raise ValueError('Exact original full reference denominator changed')
        atomic(out / 'receipt.json', dict(status='SAME_FINAL_PREDICTIONS_COMPARED_WITH_EXACT_ORIGINAL_BEST_REFERENCES',
            protocol_sha256=digest(out / 'protocol.json'), same_final_model_binding=fixed,
            original_best_baseline_categories=protocol['original_best_baseline_categories'],
            final_model_categories_on_original_references=totals, assessments=assessments,
            current_parsed_reference_results_remain_separate=True,
            historical_references_used_for_selection=False, current_labels_or_models_changed=False,
            model_fit_calls=0, model_predictions_executed=0, full_contract_pass=False))
    except BaseException:
        atomic(out / 'failure.json', dict(traceback=traceback.format_exc(), model_fit_calls=0))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--regression-name', required=True)
    parser.add_argument('--regression-pid', type=int, required=True)
    parser.add_argument('--original-replay', required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--poll-seconds', type=float, default=10.)
    main(parser.parse_args())
