"""Compare one final validation-fixed model on every original baseline input.

The collection is declared externally and bound to identical TRAIN input files.
No baseline score selects a model. Complete TRAIN and final TEST must already
have finished; predictions are frozen before original reference geometry reads.
This development regression does not certify independent generalization.
"""
import argparse
import gc
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback

import torch

from tools.run_partitioned_full_dataset_stage import configure
from tools.record_training_experiment import record, git_head
from tools.evaluate_full_weld_predictions import SOURCE_NAMES as PREDICTION_SOURCES
from weld_pipeline.server_parent_prediction_runtime import load_selector, load_input_pool, predict
from weld_pipeline.server_predicted_parent_pool import FrozenRun446Priors
from weld_pipeline.server_cached_parent_relation_prior import CachedRun446RelationPrior
from weld_pipeline.server_query_checkpoint_model import load_query_model
from weld_pipeline.server_runtime_aligned_training_samples import FIXED_QUERY_FRACTIONS
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents
from weld_pipeline.sjg_local_runtime_904 import digest, read, write

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['tools/continue_validation_fixed_baseline_regression.py',
    'tools/verify_validation_fixed_baseline_regression.py',
    'tools/run_partitioned_full_dataset_stage.py', 'tools/prepare_full_predicted_parent_pool.py',
    'tools/evaluate_full_weld_predictions.py', 'tools/assess_full_weld_output_fields.py',
    'weld_pipeline/server_parent_prediction_runtime.py',
    'weld_pipeline/server_predicted_parent_pool.py',
    'weld_pipeline/server_cached_parent_relation_prior.py',
    'weld_pipeline/server_partitioned_query_fit.py',
    'weld_pipeline/server_partitioned_full_dataset.py',
    'weld_pipeline/sjg_native_d3_adapter_1056.py']
SOURCES = list(dict.fromkeys([*SOURCES, *PREDICTION_SOURCES]))


def local(root, name):
    path = (root / name).resolve(strict=True)
    path.relative_to(root.resolve())
    if not path.is_file():
        raise ValueError('A bound persistent regression file is required')
    return path


def name(value):
    if not value or value in ('.', '..') or Path(value).name != value:
        raise ValueError('A plain independent experiment name is required')
    return value


def atomic(path, value):
    pending = path.with_name(path.name + '.incomplete')
    write(pending, value)
    os.replace(pending, path)


def collection(root, train, manifest):
    """Require every original case, the original truth and identical TRAIN input."""
    manifest = local(root, manifest)
    value = read(manifest)
    if value['schema'] != 'COMPLETE_ORIGINAL_BASELINE_REGRESSION_COLLECTION_V1':
        raise ValueError('A complete declared original baseline collection is required')
    summary_path = local(root, value['baseline_summary'])
    if digest(summary_path) != value['baseline_summary_sha256']:
        raise ValueError('Original baseline summary changed')
    summary = read(summary_path)
    required = [case['key'] for case in summary['cases']]
    rows = value['cases']
    if (summary['status'] != 'COMPLETE_FIVE_CASE_GEOMETRY_SCORE' or summary['missing']
            or len(set(required)) != len(required)
            or sorted(row['original_key'] for row in rows) != sorted(required)
            or len({row['job_key'] for row in rows}) != len(rows)):
        raise ValueError('Original baseline cases are omitted, duplicated or incomplete')
    for row in rows:
        if row['job_key'] not in train.jobs:
            raise ValueError('A regression input must remain TRAIN, never a held role')
        job = train.jobs[row['job_key']]
        for field in ('original_input_receipt', 'original_label_receipt', 'original_targets',
                      'baseline_prediction', 'baseline_prediction_receipt', 'baseline_evaluation'):
            path = local(root, row[field])
            if digest(path) != row[field + '_sha256']:
                raise ValueError('An original bound regression artifact changed: ' + field)
        inp = read(local(root, row['original_input_receipt']))
        label = read(local(root, row['original_label_receipt']))
        freeze = read(local(root, row['baseline_prediction_receipt']))
        prior = read(local(root, row['baseline_evaluation']))
        if (inp['source_sha256'] != job['workpiece_sha256']
                or row['workpiece_sha256'] != job['workpiece_sha256']
                or inp['original_points'] != job['original_points']
                or label['role'] != 'TRAIN' or label['source_key'] != row['original_key']
                or label['files'][inp['source_file']] != inp['source_sha256']
                or label['targets_sha256'] != row['original_targets_sha256']
                or freeze['source'] != row['original_key'] or freeze['GT_reads'] != 0
                or freeze['STEP_reads'] != 0
                or freeze['decoded_sha256'] != row['baseline_prediction_sha256']
                or prior['source'] != row['original_key'] or prior['role'] != 'TRAIN'
                or prior['target_sha256'] != row['original_targets_sha256']
                or prior['prediction_sha256'] != row['baseline_prediction_sha256']):
            raise ValueError('Original regression ownership, raw input or reference binding differs')
    return value


def process_identity(pid):
    proc = Path('/proc') / str(pid)
    return dict(pid=pid, start_ticks=proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19],
                argv=proc.joinpath('cmdline').read_bytes().decode().strip('\x00').split('\x00'))


def main(args):
    if args.poll_seconds < 5:
        raise ValueError('A conservative explicit polling interval is required')
    base, train, held, modules = configure(args.dataset_name)
    evaluation = modules['tools.evaluate_full_weld_predictions']
    assessment = modules['tools.assess_full_weld_output_fields']
    prepare = modules['tools.prepare_full_predicted_parent_pool']
    manifest = local(ROOT, args.baseline_manifest)
    declared = collection(ROOT, train, manifest)
    control_path = local(ROOT, args.admission_control)
    control = read(control_path)
    if (control['status'] != 'PASS_COMPLETE_ORIGINAL_BASELINE_OWNERSHIP_AND_REAL_PARTIAL_MODEL_REJECTION'
            or control['sources'] != {item: digest(ROOT / item) for item in SOURCES}
            or control['manifest_sha256'] != digest(manifest)
            or not all(control['rejection_controls'].values())
            or control['all_original_cases'] != len(declared['cases'])
            or control['optimizer_updates'] != 0 or control['final_test_geometry_reads'] != 0):
        raise ValueError('Current complete original baseline ownership controls are required')
    final = base / name(args.final_test_name)
    final_protocol = read(final / 'protocol.json')
    identity = process_identity(args.final_test_pid)
    if ('tools.continue_validation_fixed_test' not in identity['argv']
            or args.final_test_name not in identity['argv']):
        raise ValueError('The declared real final TEST process must own this experiment')
    sources = {item: digest(ROOT / item) for item in SOURCES}
    out = base / name(args.experiment_name)
    out.mkdir(exist_ok=False)
    protocol = dict(status='WAIT_COMPLETE_FINAL_TEST_THEN_SAME_FIXED_MODEL_BASELINE_REGRESSION',
        source_commit=git_head(ROOT), sources=sources, manifest=str(manifest.relative_to(ROOT)),
        manifest_sha256=digest(manifest), training_registry_sha256=train.registry_sha256,
        admission_control_sha256=digest(control_path),
        evaluation_registry_sha256=held.registry_sha256,
        final_test_protocol_sha256=digest(final / 'protocol.json'), final_process=identity,
        required_original_cases=[row['original_key'] for row in declared['cases']],
        original_reference_denominator=read(local(ROOT, declared['baseline_summary']))['totals']['overall']['G'],
        baseline_not_used_for_model_selection=True, optimizer_updates=0, held_fit_calls=0,
        independent_generalization_certified=False, full_contract_pass=False)
    atomic(out / 'protocol.json', protocol)

    def verify():
        if (sources != {item: digest(ROOT / item) for item in sources}
                or digest(manifest) != protocol['manifest_sha256']
                or digest(control_path) != protocol['admission_control_sha256']
                or digest(final / 'protocol.json') != protocol['final_test_protocol_sha256']
                or train.registry_sha256 != digest(base / 'training_registry.json')
                or held.registry_sha256 != digest(base / 'evaluation_registry.json')
                or final_protocol['sources'] != {item: digest(ROOT / item) for item in final_protocol['sources']}):
            raise ValueError('Frozen regression, final model workflow or complete registration changed')

    try:
        while not (final / 'receipt.json').exists():
            verify()
            if (final / 'failure.json').exists():
                raise RuntimeError('The existing final TEST process failed')
            if process_identity(args.final_test_pid) != identity:
                raise RuntimeError('The original final TEST process is absent or changed before completion')
            atomic(out / 'progress.json', dict(status='WAITING_EXISTING_COMPLETE_FINAL_TEST',
                controller_pid=os.getpid(), final_test_pid=args.final_test_pid,
                final_test_start_ticks=identity['start_ticks'], fit_calls=0,
                reference_geometry_reads=0, unix_time=time.time()))
            time.sleep(args.poll_seconds)
        verify()
        complete = read(final / 'receipt.json')
        if (complete['status'] != 'ALL_REGISTERED_FINAL_TEST_GEOMETRY_AND_FIELDS_EVALUATED_NOT_PROMOTED'
                or complete['protocol_sha256'] != protocol['final_test_protocol_sha256']
                or complete['fit_calls'] != 0 or complete['test_evaluations_used_for_selection'] != 0):
            raise ValueError('Real complete validation-fixed TEST evidence is required')
        for field in ('geometry_summary', 'field_summary'):
            if digest(local(ROOT, complete[field + '_path'])) != complete[field + '_sha256']:
                raise ValueError('Completed final TEST artifacts changed')
        selection = final / 'final_selection.json'
        fixed = read(selection)
        query_sha, selector_sha = fixed['query_checkpoint_sha256'], fixed['selector_checkpoint_sha256']
        if evaluation.verify_final_selection(selection, query_sha, selector_sha, held.registry_sha256) != complete['final_selection_sha256']:
            raise ValueError('The regression must use the prior validation-fixed final model')
        continuation = base / name(fixed['selected_continuation'])
        binding = read(continuation / 'model_pair_binding.json')
        if binding['query_checkpoint_sha256'] != query_sha or binding['selector_checkpoint_sha256'] != selector_sha:
            raise ValueError('Final regression query and selector are not the same chosen pair')
        cp = read(continuation / 'protocol.json')
        parent = base / name(cp['parent_training_name'])
        query_dir = base / name(cp['query_training_name'])
        for src, dest, sha in [(continuation / 'query_checkpoint.pt', out / 'query_checkpoint.pt', query_sha),
                (local(ROOT, binding['selector_checkpoint']), out / 'selector_checkpoint.pt', selector_sha),
                (parent / 'protocol.json', out / 'selector_training_protocol.json', digest(parent / 'protocol.json'))]:
            evaluation.immutable_copy(src, dest, sha)
        state = torch.load(out / 'query_checkpoint.pt', map_location='cpu', weights_only=False)
        model, architecture = load_query_model(state)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_num_threads(2)
        selector, selector_state, selector_protocol = load_selector(out / 'selector_checkpoint.pt',
            selector_sha, out / 'selector_training_protocol.json', query_sha, device)
        fit = evaluation.complete_train_fit(state, selector_state, query_sha, selector_sha,
            query_dir / 'receipt.json', parent / 'receipt.json', selector_protocol)
        assessment.select_records(local(ROOT, complete['geometry_summary_path']).parent,
                                  held, 'FINAL_TEST_ONLY', selection)
        atomic(out / 'fixed_model_binding.json', dict(query_checkpoint_sha256=query_sha,
            selector_checkpoint_sha256=selector_sha, query_architecture=architecture,
            final_selection_sha256=digest(selection), complete_train_fit=fit,
            final_test_receipt_sha256=digest(final / 'receipt.json'), baseline_used_for_selection=False))
        storage = Path(tempfile.mkdtemp(prefix='weld-fixed-baseline-regression-', dir='/tmp'))
        (out / 'pools').symlink_to(storage, target_is_directory=True)
        atomic(out / 'storage_binding.json', dict(temporary_directory=str(storage),
            scope='Own complete rebuildable prediction pools only; outputs and models persistent'))
        priors = FrozenRun446Priors(ROOT)
        cached = CachedRun446RelationPrior(priors)
        results, totals, old_totals = [], {}, {}
        for original in declared['cases']:
            verify()
            collection(ROOT, train, manifest)
            key = original['job_key']
            folder = out / 'job_records' / name(original['original_key'])
            folder.mkdir(parents=True, exist_ok=False)
            pool = prepare.prepare(train, key, model, state, priors, out,
                'fixed_and_grid', FIXED_QUERY_FRACTIONS, 'lazy', 'input_scaled_v1')
            row = load_input_pool(train, key, out, cached)
            parents, decoded, freeze = predict(selector, row)
            evaluation.immutable_copy(out / 'pools' / key / 'receipt.json', folder / 'pool_receipt.json',
                                      row['pool_receipt_sha256'])
            prepare.save_gzip(folder / 'selected_parents.json.gz', parents)
            freeze.update(checkpoint_sha256=selector_sha, query_checkpoint_sha256=query_sha,
                selected_geometry_sha256=digest(folder / 'selected_parents.json.gz'),
                role='ORIGINAL_TRAIN_BASELINE_REGRESSION_ONLY', full_train_fit_verified=True)
            atomic(folder / 'prediction_freeze.json', freeze)
            target = read(local(ROOT, original['original_targets']))
            allowances = [False] * len(target['parents'])
            new_score = score_native_parents(decoded, target, allowances)
            old_score = score_native_parents(read(local(ROOT, original['baseline_prediction'])), target, allowances)
            if old_score['categories'] != read(local(ROOT, original['baseline_evaluation']))['categories']:
                raise ValueError('Original baseline scorer/input/reference parity changed')
            for score, aggregate in [(new_score, totals), (old_score, old_totals)]:
                for category, values in score['categories'].items():
                    total = aggregate.setdefault(category, dict(G=0, predicted=0, TP=0, FP=0))
                    for field in total:
                        total[field] += values[field]
            result = dict(original_key=original['original_key'], job_key=key,
                workpiece_sha256=original['workpiece_sha256'], original_targets_sha256=original['original_targets_sha256'],
                original_baseline_geometry_score=old_score, final_model_geometry_score=new_score,
                predictions_frozen_before_original_geometry_reference_reads=True,
                query_checkpoint_sha256=query_sha, selector_checkpoint_sha256=selector_sha,
                artifacts={item: digest(folder / item) for item in
                    ['pool_receipt.json', 'prediction_freeze.json', 'selected_parents.json.gz']},
                fit_calls=0, independent_generalization_certified=False, full_contract_pass=False)
            atomic(folder / 'receipt.json', result)
            results.append(dict(path=str((folder / 'receipt.json').relative_to(ROOT)), sha256=digest(folder / 'receipt.json')))
            owned = out / 'pools' / key
            if owned.resolve().parent != storage or owned.name != key:
                raise ValueError('Unexpected owned complete regression pool')
            for item, sha in pool['files'].items():
                if digest(owned / item) != sha:
                    raise ValueError('Complete regression pool changed before verified cache eviction')
            shutil.rmtree(owned)
            cached.clear()
            del row, parents, decoded, target
            gc.collect()
            atomic(out / 'progress.json', dict(status='COMPLETE_ORIGINAL_INPUT_REGRESSION_RECORDED',
                controller_pid=os.getpid(), completed_original_cases=len(results),
                registered_original_cases=len(declared['cases']), fit_calls=0, unix_time=time.time()))
        verify()
        baseline = read(local(ROOT, declared['baseline_summary']))
        if old_totals != baseline['totals'] or totals['overall']['G'] != protocol['original_reference_denominator']:
            raise ValueError('Complete original regression reference denominator changed')
        summary = dict(status='SAME_VALIDATION_FIXED_MODEL_COMPLETE_ORIGINAL_TRAIN_BASELINE_REGRESSION',
            protocol_sha256=digest(out / 'protocol.json'), fixed_model_binding_sha256=digest(out / 'fixed_model_binding.json'),
            required_original_cases=protocol['required_original_cases'], completed_original_cases=len(results),
            job_records=results, original_baseline_totals=old_totals, final_model_totals=totals,
            historical_parity_proven=baseline['historical_parity_proven'],
            historical_96_percent_not_substituted_for_actual_current_baseline=True,
            baseline_used_for_model_selection=False, model_promoted=False, fit_calls=0,
            attributes_business_frames_csv_and_timing_not_assessed_by_this_geometry_regression=True,
            independent_generalization_certified=False, full_contract_pass=False)
        atomic(out / 'receipt.json', summary)
        record(out.name, summary['status'], [str((out / item).relative_to(ROOT)) for item in
            ['protocol.json', 'fixed_model_binding.json', 'receipt.json']],
            [str((out / item).relative_to(ROOT)) for item in ['query_checkpoint.pt', 'selector_checkpoint.pt']],
            dict(baseline_not_used_for_selection=True, complete_original_cases=len(results)))
    except BaseException:
        atomic(out / 'failure.json', dict(status='INCOMPLETE_ORIGINAL_BASELINE_REGRESSION_NOT_PROMOTED',
            traceback=traceback.format_exc(), unix_time=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--baseline-manifest', required=True)
    parser.add_argument('--admission-control', required=True)
    parser.add_argument('--final-test-name', required=True)
    parser.add_argument('--final-test-pid', required=True, type=int)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--poll-seconds', type=float, default=10.)
    main(parser.parse_args())
