"""Assess authenticated saved outputs without changing fitting or predictions.

TRAIN records are different post-update checkpoints, so this entry point never
reports their sum as one model's score. Held use requires the whole registered
role's completed geometry evaluation; TEST also requires prior fixed selection.
"""
import argparse
import gc
import hashlib
import json
import re
import time
import traceback
from pathlib import Path

import numpy as np

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_complete_output_assessment import assess_complete_output, process_reference
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.server_full_evaluation_dataset import EvaluationShardDataset, ROLES
from weld_pipeline.server_parent_prediction_runtime import load_gzip
from weld_pipeline.server_predicted_parent_pool import decoded_subset
from weld_pipeline.sjg_local_runtime_904 import read, digest, write
from tools.evaluate_full_weld_predictions import verify_final_selection
from tools.record_training_experiment import git_head

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'
SOURCES = ['tools/assess_full_weld_output_fields.py',
    'weld_pipeline/server_complete_output_assessment.py',
    'weld_pipeline/server_full_shard_dataset.py', 'weld_pipeline/server_full_evaluation_dataset.py',
    'weld_pipeline/sjg_native_d3_adapter_1056.py', 'weld_pipeline/contract_geometry_d3_409.py',
    'weld_pipeline/contract_curve_geometry.py', 'weld_pipeline/world_set_evaluation_285.py',
    'weld_pipeline/world_set_runtime_285.py', 'weld_pipeline/server_predicted_parent_pool.py',
    'weld_pipeline/server_parent_prediction_runtime.py', 'tools/evaluate_full_weld_predictions.py']


def local(path, directory=False):
    path = Path(path).resolve()
    path.relative_to(ROOT)
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError('A persistent project artifact is required')
    return path


def pool_freeze_fingerprint(pool, saved_sha, role):
    """Recover exact original TRAIN receipt bytes before its one audit addition.

    Existing TRAIN producers append semantic_fingerprints to their saved copy.
    The original sjg_local_runtime_904.write encoding has no trailing newline.
    Equality to the frozen original SHA is still mandatory. Held copies require
    byte equality and never use this reconstruction.
    """
    if role != 'TRAIN' or 'semantic_fingerprints' not in pool:
        return saved_sha
    original = {key: value for key, value in pool.items() if key != 'semantic_fingerprints'}
    encoded = json.dumps(original, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def authenticate(run, path, protocol_sha, registry_sha, role):
    """Authenticate frozen outputs before the caller can request any labels."""
    receipt = read(path)
    folder = path.parent
    freeze_path, pool_path = folder / 'prediction_freeze.json', folder / 'pool_receipt.json'
    freeze, pool = read(freeze_path), read(pool_path)
    saved_pool_sha = digest(pool_path)
    if (receipt['protocol_sha256'] != protocol_sha or pool['registry_sha256'] != registry_sha
            or pool['job_key'] != receipt['job_key'] or pool['geometry_GT_reads'] != 0
            or pool['STEP_reads'] != 0 or freeze['runtime_uses_reference_geometry']
            or pool_freeze_fingerprint(pool, saved_pool_sha, role) != freeze['query_pool_receipt_sha256']
            or digest(folder / 'selected_parents.json.gz') != freeze['selected_geometry_sha256']):
        raise ValueError('Prediction is not bound to an authenticated input-only freeze')
    if role == 'TRAIN':
        if (receipt['status'] != 'COMPLETE_REGISTERED_JOB_FIT_RECORDED_NOT_MODEL_PROMOTION'
                or receipt['independent_evaluations'] != 0
                or receipt['pool_receipt_sha256'] != saved_pool_sha
                or receipt['checkpoint_sha256'] != freeze['checkpoint_sha256']):
            raise ValueError('TRAIN record is not a saved post-update diagnostic')
        checkpoint = local(run / receipt['checkpoint'])
        checkpoint.relative_to(run)
        if digest(checkpoint) != freeze['checkpoint_sha256']:
            raise ValueError('TRAIN selector checkpoint changed')
        query_sha = read(run / 'protocol.json')['query_checkpoint_sha256']
    else:
        if (receipt['status'] != 'HELD_COMPLETE_OUTPUT_FROZEN_AND_GEOMETRY_EVALUATED'
                or receipt['role'] != role or freeze['role'] != role
                or not freeze['full_train_fit_verified']
                or receipt['selector_checkpoint_sha256'] != freeze['checkpoint_sha256']):
            raise ValueError('Held record requires its declared role and full TRAIN fit')
        for name, sha in receipt['artifacts'].items():
            if Path(name).name != name or digest(folder / name) != sha:
                raise ValueError('Saved held prediction or geometry assessment changed')
        query_sha = receipt['query_checkpoint_sha256']
    if pool['query_checkpoint_sha256'] != query_sha:
        raise ValueError('Prediction query checkpoint differs from its protocol')
    method = ('EXACT_ORIGINAL_TRAIN_BYTES_RECONSTRUCTED_AND_FROZEN_SHA_MATCHED'
              if role == 'TRAIN' and 'semantic_fingerprints' in pool else 'SAVED_BYTES_SHA_MATCHED')
    return receipt, dict(freeze, pool_receipt_authentication=method)


def select_records(run, dataset, role, final_selection):
    if role == 'TRAIN':
        records = sorted((run / 'job_records').glob('*/*/receipt.json'))
        if not records:
            raise ValueError('No complete TRAIN post-update records exist')
        return records, None
    summary_path = local(run / 'summary.json')
    summary = read(summary_path)
    required = sorted(k for k, (_, v) in dataset.versions.items() if v['role'] == role)
    if (summary['status'] != 'COMPLETE_REGISTERED_HELD_ROLE_GEOMETRY_EVALUATED_NOT_PROMOTED'
            or summary['role'] != role or not summary['full_train_fit_verified']
            or summary['evaluation_registry_sha256'] != dataset.registry_sha256
            or summary['required_versions'] != required or summary['scored_versions'] != required):
        raise ValueError('Complete registered role evaluation is required before field assessment')
    for name, key in [('query_checkpoint.pt', 'query_checkpoint_sha256'),
                      ('selector_checkpoint.pt', 'selector_checkpoint_sha256')]:
        if digest(local(run / name)) != summary[key]:
            raise ValueError('Saved held model pair changed')
    if role == 'FINAL_TEST_ONLY':
        if final_selection is None:
            raise ValueError('TEST needs a model fixed by validation before testing')
        sha = verify_final_selection(final_selection, summary['query_checkpoint_sha256'],
            summary['selector_checkpoint_sha256'], dataset.registry_sha256)
        if sha != summary['final_selection_sha256']:
            raise ValueError('TEST selection differs from prior saved evaluation')
    records = []
    for item in summary['job_records']:
        path = local(ROOT / item['path'])
        path.relative_to(run)
        if digest(path) != item['sha256']:
            raise ValueError('A complete held job record changed')
        receipt = read(path)
        if (receipt['query_checkpoint_sha256'] != summary['query_checkpoint_sha256']
                or receipt['selector_checkpoint_sha256'] != summary['selector_checkpoint_sha256']):
            raise ValueError('Held jobs do not share the declared frozen model pair')
        records.append(path)
    actual = [v['version_key'] for path in records for v in read(path)['versions']]
    if sorted(actual) != required:
        raise ValueError('Held record list duplicates or omits registered versions')
    return records, summary_path


def main(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.experiment_name):
        raise ValueError('Use a plain unique experiment name')
    run = local(ROOT / args.prediction_run, directory=True)
    dataset = FullShardDataset(ROOT) if args.role == 'TRAIN' else EvaluationShardDataset(ROOT)
    paths, prior_summary = select_records(run, dataset, args.role, args.final_selection)
    source_shas = {name: digest(ROOT / name) for name in SOURCES}
    protocol_sha = digest(local(run / 'protocol.json'))
    # Freeze the exact completed-record list. New live TRAIN jobs are another assessment.
    bindings = [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in paths]
    out = BASE / args.experiment_name
    out.mkdir(exist_ok=False)
    protocol = dict(schema='FROZEN_FULL_OUTPUT_FIELD_ASSESSMENT_V1', role=args.role,
        source_commit_at_initialization=git_head(ROOT),
        prediction_run=str(run.relative_to(ROOT)), prediction_protocol_sha256=protocol_sha,
        registry_sha256=dataset.registry_sha256, records=bindings, sources=source_shas,
        prior_complete_role_summary_sha256=digest(prior_summary) if prior_summary else None,
        aggregate_train_snapshots_as_one_model=False, fit_calls=0, predictions_changed=False,
        complete_contract_or_model_promotion=False)
    write(out / 'protocol.json', protocol)
    results = []
    try:
        for index, (path, binding) in enumerate(zip(paths, bindings)):
            if digest(path) != binding['sha256'] or digest(run / 'protocol.json') != protocol_sha:
                raise ValueError('Frozen source record changed during assessment')
            if source_shas != {name: digest(ROOT / name) for name in SOURCES}:
                raise ValueError('Assessment source changed while running')
            receipt, freeze = authenticate(run, path, protocol_sha, dataset.registry_sha256, args.role)
            key = receipt['job_key']
            if key not in dataset.jobs:
                raise ValueError('Prediction job is not registered in its role')
            versions = receipt['training_views'] if args.role == 'TRAIN' else receipt['versions']
            expected = sorted(v['key'] for v in dataset.jobs[key]['versions']
                              if args.role == 'TRAIN' or v['role'] == args.role)
            if sorted(v['version_key'] for v in versions) != expected:
                raise ValueError('Frozen record lost an authorized label view')
            parents = load_gzip(path.parent / 'selected_parents.json.gz')
            decoded = decoded_subset(parents, range(len(parents)))
            rows = []
            for version in versions:
                version_key = version['version_key']
                if args.role == 'TRAIN':
                    view = dataset.supervision(version_key)
                    target = view.target
                    reference = process_reference(target, view.records, view.attribute_arrays, view.endpoint_evidence)
                    weight, family = view.family_equal_version_weight, view.family_id
                    previous = {k: version[k] for k in ('G', 'TP', 'FP', 'predicted')}
                else:
                    view = dataset.reference(version_key, path.parent, args.final_selection)
                    target, folder = view['targets'], view['label_folder']
                    with np.load(folder / 'attribute_labels.npz', allow_pickle=False) as data:
                        arrays = {name: data[name] for name in data.files}
                    reference = process_reference(target, read(folder / 'attribute_records.json'), arrays,
                                                  read(folder / 'endpoint_evidence.json'))
                    weight, family = view['evaluation_weight'], view['family_id']
                    previous = {k: version['geometry_score']['categories']['overall'][k]
                                for k in ('G', 'TP', 'FP', 'predicted')}
                assessment = assess_complete_output(decoded, target, reference)
                actual = assessment['development_d3_geometry']['overall']
                if any(actual[k] != previous[k] for k in previous):
                    raise ValueError('New assessment differs from original saved direct D3 geometry')
                rows.append(dict(version_key=version_key, family_id=family,
                    reference_weight=weight, assessment=assessment))
            # Scoring must leave frozen inputs, geometry and checkpoint bindings untouched.
            authenticate(run, path, protocol_sha, dataset.registry_sha256, args.role)
            destination = out / f'job_{index:04d}.json'
            result = dict(job_key=key, source_receipt=binding, role=args.role,
                checkpoint_sha256=freeze['checkpoint_sha256'], versions=rows,
                pool_receipt_authentication=freeze['pool_receipt_authentication'],
                predictions_unchanged=True, fit_calls=0, full_contract_pass=False)
            write(destination, result)
            results.append(dict(path=str(destination.relative_to(ROOT)), sha256=digest(destination),
                job_key=key, checkpoint_sha256=freeze['checkpoint_sha256'],
                versions=[r['version_key'] for r in rows]))
            write(out / 'progress.json', dict(status='FROZEN_OUTPUT_FIELDS_ASSESSED',
                completed_records=len(results), required_records=len(paths), role=args.role,
                fit_calls=0, unix_time=time.time()))
            print('FIELD_ASSESSMENT', index+1, '/', len(paths), key,
                  'views', len(rows), 'predictions_unchanged', flush=True)
            del parents, decoded, rows, result, view, target, reference, assessment
            gc.collect()
        summary = dict(status='ALL_FROZEN_RECORDS_FIELDS_ASSESSED_NOT_PROMOTED', role=args.role,
            records=results, source_protocol_sha256=digest(out / 'protocol.json'),
            completed_records=len(results), version_visits=sum(len(r['versions']) for r in results),
            distinct_model_checkpoints=len({r['checkpoint_sha256'] for r in results}),
            single_final_model_accuracy_not_claimed=args.role == 'TRAIN',
            independent_generalization_certified=False, fit_calls=0,
            field_errors_conditioned_on_raw_ordered_geometry=True,
            unknown_labels_not_filled=True, field_mm_tolerance_not_frozen=True,
            business_frame_and_direction_and_side_not_certified=True,
            split_reason_and_csv_and_raw_timing_not_assessed=True, full_contract_pass=False)
        write(out / 'summary.json', summary)
        print('COMPLETE_FIELD_ASSESSMENT', args.role, len(results), flush=True)
    except Exception:
        write(out / 'failure.json', dict(status='INCOMPLETE_FIELD_ASSESSMENT_NOT_PROMOTED',
            role=args.role, completed_records=len(results), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction-run', required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--role', choices=('TRAIN', *ROLES), required=True)
    parser.add_argument('--final-selection')
    main(parser.parse_args())
