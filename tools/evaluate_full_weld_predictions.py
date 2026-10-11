"""Freeze complete held predictions, then score each authorized label view.

Selection, protected diagnostics and final testing are separate executions.
By default both the query model and its bound parent selector must have fitted
all registered TRAIN inputs. Partial controls cannot authorize final testing.
The original D3 geometry score does not certify attributes, frames or CSV.
"""
import argparse
import fcntl
import gc
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback

import torch

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_full_evaluation_dataset import EvaluationShardDataset, ROLES
from weld_pipeline.server_cached_parent_relation_prior import CachedRun446RelationPrior
from weld_pipeline.server_legacy_anchored_full_weld_model import LegacyAnchoredFullWeldModel
from weld_pipeline.server_parent_prediction_runtime import load_input_pool, load_selector, load_gzip, predict
from weld_pipeline.server_predicted_parent_pool import FrozenRun446Priors, decoded_subset
from weld_pipeline.server_runtime_aligned_training_samples import FIXED_QUERY_FRACTIONS
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents
from weld_pipeline.sjg_local_runtime_904 import digest, read, write
from tools.prepare_full_predicted_parent_pool import prepare, save_gzip
from tools.record_training_experiment import record as record_experiment, git_head

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'
SOURCE_NAMES = ['tools/evaluate_full_weld_predictions.py',
    'tools/prepare_full_predicted_parent_pool.py', 'tools/record_training_experiment.py',
    'weld_pipeline/server_parent_prediction_runtime.py', 'weld_pipeline/server_full_evaluation_dataset.py',
    'weld_pipeline/server_full_shard_dataset.py', 'weld_pipeline/server_ordered_parent_set_model.py',
    'weld_pipeline/server_reference_parent_selection.py', 'weld_pipeline/server_lazy_parent_competition.py',
    'weld_pipeline/server_cached_parent_relation_prior.py', 'weld_pipeline/server_predicted_parent_pool.py',
    'weld_pipeline/server_full_prediction_runtime.py', 'weld_pipeline/server_full_weld_model.py',
    'weld_pipeline/server_legacy_anchored_full_weld_model.py',
    'weld_pipeline/server_runtime_aligned_training_samples.py',
    'weld_pipeline/sjg_parent_set_features_1160.py', 'weld_pipeline/sjg_additive_parent_precision_1254.py',
    'weld_pipeline/sjg_child_parent_supervision_1255.py', 'weld_pipeline/sjg_native_d3_adapter_1056.py',
    'weld_pipeline/contract_geometry_d3_409.py', 'weld_pipeline/contract_curve_geometry.py',
    'weld_pipeline/world_set_runtime_285.py', 'weld_pipeline/world_set_evaluation_285.py',
    'weld_pipeline/sjg_local_runtime_904.py']


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.incomplete')
    write(temporary, value)
    os.replace(temporary, path)


def local_file(name):
    path = Path(name).resolve()
    path.relative_to(ROOT)
    if not path.is_file():
        raise ValueError('A persistent project file is required')
    return path


def immutable_copy(source, destination, sha256):
    if digest(source) != sha256:
        raise ValueError('Declared source fingerprint differs')
    if not destination.exists():
        temporary = destination.with_name(destination.name + '.incomplete')
        shutil.copyfile(source, temporary)
        if digest(temporary) != sha256:
            raise ValueError('Snapshot changed while copying')
        os.replace(temporary, destination)
    if digest(destination) != sha256:
        raise ValueError('Existing immutable model snapshot differs')


def complete_train_fit(query, selector, query_sha, selector_sha, query_receipt, selector_receipt, selector_protocol):
    """Check real complete visit sets, not just a receipt's descriptive status."""
    registration = read(BASE / 'training_registry.json')
    jobs = {j['job_key'] for j in registration['cases']}
    versions = {v['key'] for j in registration['cases'] for v in j['versions']}
    qr, sr = read(local_file(query_receipt)), read(local_file(selector_receipt))
    if (qr['status'] != 'ALL_TRAIN_WARMSTART_FITTED_NOT_PROMOTED_NOT_COMPLETE_INFERENCE'
            or qr['checkpoint_sha256'] != query_sha or qr['protocol_sha256'] != query['protocol_sha256']
            or qr['training_versions'] != len(versions) or qr['validation_or_test_label_reads'] != 0
            or qr['epochs'] < 1 or qr['optimizer_steps'] != query['optimizer_steps']
            or query['optimizer_steps'] < 1):
        raise ValueError('Complete TRAIN query receipt is missing or differs from this checkpoint')
    completed = set(query['completed_views'])
    if (completed != {str(e) + ':' + v for e in range(qr['epochs']) for v in versions}
            or len(completed) != qr['completed_views']):
        raise ValueError('Query has not actually visited every TRAIN view each registered epoch')
    if (sr['status'] != 'ALL_REGISTERED_TRAIN_PARENT_FIT_COMPLETE_NOT_PROMOTED'
            or sr['held_or_diagnostic_fit_calls'] != 0
            or selector['protocol_sha256'] != sr['protocol_sha256']
            or selector['completed'] != sr['completed']
            or selector['optimizer_steps'] != sr['optimizer_steps']
            or selector_protocol['registry_sha256'] != digest(BASE / 'training_registry.json')):
        raise ValueError('Complete TRAIN selector receipt differs from this bound checkpoint')
    resume_file = local_file(selector_receipt).parent / 'resume.json'
    resume = read(resume_file)
    final_checkpoint = (resume_file.parent / resume['checkpoint']).resolve()
    final_checkpoint.relative_to(resume_file.parent.resolve())
    if resume['sha256'] != selector_sha or digest(final_checkpoint) != selector_sha:
        raise ValueError('Selector differs from the actual final durable TRAIN checkpoint')
    if (selector['passes'] < 1 or selector['passes'] != selector_protocol['passes']
            or selector['passes_completed'] != selector['passes']
            or set(selector['completed']) != {str(p) + '/' + j for p in range(selector['passes']) for j in jobs}
            or set(selector['processed_versions']) != versions
            or set(selector['job_visits']) != jobs
            or any(selector['job_visits'][j] != selector['passes'] for j in jobs)):
        raise ValueError('Selector has not actually visited all registered TRAIN inputs and label views')
    return dict(query_receipt_sha256=digest(local_file(query_receipt)),
                selector_receipt_sha256=digest(local_file(selector_receipt)),
                selector_final_resume_sha256=digest(resume_file),
                actual_final_selector_checkpoint_sha256=selector_sha,
                query_epochs=qr['epochs'], selector_passes=selector['passes'],
                all_registered_train_views_fitted=True, held_or_diagnostic_fit_calls=0)


def verify_final_selection(path, query_sha, selector_sha, registry_sha):
    """Bind TEST to a prior complete validation record, never diagnostics."""
    path = local_file(path)
    selected = read(path)
    if (selected['status'] != 'FINAL_MODEL_SELECTED_VALIDATION_ONLY_BEFORE_TEST'
            or selected['diagnostic_used_for_selection'] or selected['test_evaluations_for_selection'] != 0
            or selected['query_checkpoint_sha256'] != query_sha
            or selected['selector_checkpoint_sha256'] != selector_sha):
        raise ValueError('Final model selection does not authorize this model pair')
    summary_path = local_file(ROOT / selected['validation_summary_path'])
    if digest(summary_path) != selected['validation_summary_sha256']:
        raise ValueError('Fixed model validation evidence changed')
    summary = read(summary_path)
    registry = read(BASE / 'evaluation_registry.json')
    required = sorted(v['key'] for j in registry['cases'] for v in j['versions']
                      if v['role'] == 'MODEL_SELECTION_ONLY')
    if (summary['status'] != 'COMPLETE_REGISTERED_HELD_ROLE_GEOMETRY_EVALUATED_NOT_PROMOTED'
            or summary['role'] != 'MODEL_SELECTION_ONLY' or not summary['full_train_fit_verified']
            or summary['evaluation_registry_sha256'] != registry_sha
            or summary['query_checkpoint_sha256'] != query_sha
            or summary['selector_checkpoint_sha256'] != selector_sha
            or summary['scored_versions'] != summary['required_versions']
            or summary['scored_versions'] != required):
        raise ValueError('Final model selection needs complete matching validation-only evidence')
    for item in summary['job_records']:
        receipt_path = local_file(ROOT / item['path'])
        if digest(receipt_path) != item['sha256']:
            raise ValueError('Validation prediction/evaluation evidence changed')
        receipt = read(receipt_path)
        if (receipt['role'] != 'MODEL_SELECTION_ONLY'
                or receipt['query_checkpoint_sha256'] != query_sha
                or receipt['selector_checkpoint_sha256'] != selector_sha):
            raise ValueError('Validation record belongs to another role or model pair')
        for name, sha in receipt['artifacts'].items():
            if Path(name).name != name or digest(receipt_path.parent / name) != sha:
                raise ValueError('Fixed validation prediction artifacts changed')
    return digest(path)


def aggregate(results):
    rows = [v for job in results for v in job['versions']]
    categories = {}
    for row in rows:
        for name, value in row['geometry_score']['categories'].items():
            total = categories.setdefault(name, dict(G=0, predicted=0, TP=0, FP=0))
            for k in total:
                total[k] += value[k]
    mass = sum(row['evaluation_weight'] for row in rows)
    utility = sum(row['evaluation_weight'] * (2 * row['geometry_score']['categories']['overall']['TP']
        - row['geometry_score']['categories']['overall']['predicted'])
        / max(1, row['geometry_score']['categories']['overall']['G']) for row in rows)
    return dict(categories=categories,
        family_equal_normalized_reference_utility=utility / mass if mass else None,
        original_d3_geometry_only=True, full_contract_pass=False,
        attributes_and_signed_frames_and_csv_acceptance_not_evaluated=True,
        independent_generalization_certified=False,
        raw_counts_include_distinct_label_views_of_same_cloud=True)


def main(args):
    if (not args.experiment_name or Path(args.experiment_name).name != args.experiment_name
            or args.experiment_name in ('.', '..')):
        raise ValueError('A new explicit local evaluation experiment name is required')
    if args.role == 'FINAL_TEST_ONLY' and (args.allow_partial_training or not args.final_selection):
        raise ValueError('Final TEST needs full TRAIN fitting and a validation-fixed model selection')
    if not args.allow_partial_training and (not args.query_training_receipt or not args.selector_training_receipt):
        raise ValueError('Complete query and selector fitting receipts are required by default')
    torch.set_num_threads(2)
    dataset = EvaluationShardDataset(ROOT)
    out = BASE / args.experiment_name
    out.mkdir(exist_ok=True)
    with (out / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        query_source, selector_source = local_file(args.query_checkpoint), local_file(args.selector_checkpoint)
        immutable_copy(query_source, out / 'query_checkpoint.pt', args.query_sha256)
        immutable_copy(selector_source, out / 'selector_checkpoint.pt', args.selector_sha256)
        selector_protocol_source = local_file(args.selector_protocol)
        immutable_copy(selector_protocol_source, out / 'selector_training_protocol.json', digest(selector_protocol_source))
        query = torch.load(out / 'query_checkpoint.pt', map_location='cpu', weights_only=False)
        anchored = 'legacy.trunk.0.weight' in query['state_dict']
        if not anchored:
            raise ValueError('This evaluation protocol requires the declared anchored full query model')
        model = LegacyAnchoredFullWeldModel()
        model.load_state_dict(query['state_dict'], strict=True)
        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        selector, selector_state, fitted_protocol = load_selector(out / 'selector_checkpoint.pt',
            args.selector_sha256, out / 'selector_training_protocol.json', args.query_sha256, device)
        full_fit = (None if args.allow_partial_training else complete_train_fit(query, selector_state,
            args.query_sha256, args.selector_sha256, args.query_training_receipt,
            args.selector_training_receipt, fitted_protocol))
        selected_sha = (verify_final_selection(args.final_selection, args.query_sha256,
            args.selector_sha256, dataset.registry_sha256) if args.role == 'FINAL_TEST_ONLY' else None)
        jobs = [key for key, job in dataset.jobs.items() if any(v['role'] == args.role for v in job['versions'])]
        required = sorted(v['key'] for key in jobs for v in dataset.jobs[key]['versions'] if v['role'] == args.role)
        if not jobs:
            raise ValueError('No complete registered inputs for this role')
        priors = FrozenRun446Priors(ROOT)
        cached = CachedRun446RelationPrior(priors)
        sources = {name: digest(ROOT / name) for name in SOURCE_NAMES}
        protocol = dict(schema='FULL_HELD_PREDICTION_BEFORE_REFERENCE_V1',
            source_commit_at_initialization=(read(out / 'protocol.json')['source_commit_at_initialization']
                if (out / 'protocol.json').exists() else git_head(ROOT)),
            role=args.role, jobs=jobs, versions=required, evaluation_registry_sha256=dataset.registry_sha256,
            query_checkpoint_sha256=args.query_sha256, selector_checkpoint_sha256=args.selector_sha256,
            selector_training_protocol_sha256=digest(out / 'selector_training_protocol.json'),
            original_run446_prior_sha256=priors.checkpoint_sha256, full_train_fit=full_fit,
            partial_control_not_final_evaluation=args.allow_partial_training,
            final_selection_sha256=selected_sha, sources=sources,
            query_mode='fixed_and_grid', competition_mode='lazy', resource_mode='input_scaled_v1',
            fixed_query_fractions=list(FIXED_QUERY_FRACTIONS), held_or_diagnostic_fit_calls=0,
            runtime_uses_reference_geometry=False, direction_convention_review='UNCONFIRMED_DEVELOPMENT_ALLOWANCE',
            full_contract_pass=False, independent_generalization_certified=False)
        if (out / 'protocol.json').exists():
            if read(out / 'protocol.json') != protocol:
                raise ValueError('Evaluation recipe changed; use a new explicit experiment')
        else:
            atomic_json(out / 'protocol.json', protocol)
        link = out / 'pools'
        if not link.exists():
            storage = Path(tempfile.mkdtemp(prefix='weld-held-parent-pools-', dir='/tmp'))
            link.symlink_to(storage, target_is_directory=True)
            atomic_json(out / 'storage_binding.json', dict(temporary_directory=str(storage),
                scope='Own rebuildable prediction pools; frozen outputs and models remain persistent'))
        storage = link.resolve()
        if storage.parent != Path('/tmp') or not storage.name.startswith('weld-held-parent-pools-'):
            raise ValueError('Unexpected task-owned held prediction cache')
        results = []
        try:
            for key in jobs:
                record = out / 'job_records' / key
                record.mkdir(parents=True, exist_ok=True)
                if (record / 'receipt.json').exists():
                    old = read(record / 'receipt.json')
                    if old['protocol_sha256'] != digest(out / 'protocol.json'):
                        raise ValueError('Completed evaluation is bound to another protocol')
                    for name, sha in old['artifacts'].items():
                        if Path(name).name != name or digest(record / name) != sha:
                            raise ValueError('Completed evaluation evidence changed')
                    results.append(old)
                    continue
                while key not in dataset.ready_jobs(args.role):
                    atomic_json(out / 'evaluation_progress.json', dict(status='WAITING_COMPLETE_HELD_INPUT',
                        controller_pid=os.getpid(), role=args.role, job_key=key, complete_jobs=len(results),
                        required_jobs=len(jobs), fit_calls=0, unix_time=time.time()))
                    if not args.wait_ready:
                        raise ValueError('This held input is not prepared; enable waiting or resume when ready')
                    time.sleep(10)
                if sources != {name: digest(ROOT / name) for name in SOURCE_NAMES}:
                    raise ValueError('Bound prediction/scoring source changed during evaluation')
                pool_receipt = prepare(dataset, key, model, query, priors, out,
                    'fixed_and_grid', FIXED_QUERY_FRACTIONS, 'lazy', 'input_scaled_v1')
                if not (record / 'prediction_freeze.json').exists():
                    row = load_input_pool(dataset, key, out, cached)
                    parents, decoded, freeze = predict(selector, row)
                    # Copy identical pool receipt bytes: the freeze binds this exact SHA.
                    immutable_copy(out / 'pools' / key / 'receipt.json', record / 'pool_receipt.json',
                                   row['pool_receipt_sha256'])
                    save_gzip(record / 'selected_parents.json.gz', parents)
                    freeze.update(checkpoint_sha256=args.selector_sha256,
                        selected_geometry_sha256=digest(record / 'selected_parents.json.gz'),
                        role=args.role, full_train_fit_verified=full_fit is not None)
                    atomic_json(record / 'prediction_freeze.json', freeze)
                    del row, parents
                    gc.collect()
                else:
                    freeze = read(record / 'prediction_freeze.json')
                    if (freeze['checkpoint_sha256'] != args.selector_sha256
                            or freeze['query_pool_receipt_sha256'] != digest(record / 'pool_receipt.json')
                            or freeze['selected_geometry_sha256'] != digest(record / 'selected_parents.json.gz')
                            or read(record / 'pool_receipt.json') != pool_receipt):
                        raise ValueError('Unfinished evaluation prediction freeze changed')
                    parents = load_gzip(record / 'selected_parents.json.gz')
                    decoded = decoded_subset(parents, range(len(parents)))
                    del parents
                evaluations = []
                for version in dataset.jobs[key]['versions']:
                    if version['role'] != args.role:
                        continue
                    reference = dataset.reference(version['key'], record, args.final_selection)
                    score = score_native_parents(decoded, reference['targets'])
                    evaluations.append(dict(version_key=version['key'], family_id=reference['family_id'],
                        evaluation_weight=reference['evaluation_weight'], role=args.role, geometry_score=score,
                        label_receipt_sha256=version['label_receipt_sha256']))
                atomic_json(record / 'evaluation.json', dict(versions=evaluations,
                    predictions_frozen_before_reference_reads=True, held_fit_calls=0, full_contract_pass=False))
                result = dict(status='HELD_COMPLETE_OUTPUT_FROZEN_AND_GEOMETRY_EVALUATED', job_key=key,
                    role=args.role, protocol_sha256=digest(out / 'protocol.json'), versions=evaluations,
                    query_checkpoint_sha256=args.query_sha256, selector_checkpoint_sha256=args.selector_sha256,
                    artifacts={name: digest(record / name) for name in
                        ['pool_receipt.json', 'prediction_freeze.json', 'selected_parents.json.gz', 'evaluation.json']})
                atomic_json(record / 'receipt.json', result)
                results.append(result)
                record_id = 'held_' + digest(out / 'protocol.json')[:12] + '_' + key
                if not (ROOT / 'experiment_records' / (record_id + '.json')).exists():
                    record_experiment(record_id, 'HELD_ROLE_COMPLETE_OUTPUT_GEOMETRY_ONLY_NOT_PROMOTED',
                        [str(p.relative_to(ROOT)) for p in [out / 'protocol.json', record / 'pool_receipt.json',
                            record / 'prediction_freeze.json', record / 'evaluation.json', record / 'receipt.json']],
                        [str((out / name).relative_to(ROOT)) for name in
                            ['query_checkpoint.pt', 'selector_checkpoint.pt']],
                        dict(role=args.role, full_train_fit_verified=full_fit is not None, full_contract_pass=False))
                # Evict only this controller's authenticated, rebuildable candidate pool.
                for name, sha in pool_receipt['files'].items():
                    if digest(out / 'pools' / key / name) != sha:
                        raise ValueError('Candidate pool changed before owned cache eviction')
                shutil.rmtree(storage / key)
                cached.clear()
                del decoded
                gc.collect()
                atomic_json(out / 'evaluation_progress.json', dict(status='HELD_JOB_GEOMETRY_SCORE_DURABLE',
                    controller_pid=os.getpid(), role=args.role, complete_jobs=len(results), required_jobs=len(jobs),
                    fit_calls=0, unix_time=time.time()))
                print('FULL_HELD_JOB_EVALUATED', key, len(evaluations), flush=True)
            scored = sorted(v['version_key'] for r in results for v in r['versions'])
            if scored != required:
                raise ValueError('Complete registered held role lost or duplicated a label view')
            summary = dict(status='COMPLETE_REGISTERED_HELD_ROLE_GEOMETRY_EVALUATED_NOT_PROMOTED',
                role=args.role, required_versions=required, scored_versions=scored,
                evaluation_registry_sha256=dataset.registry_sha256, protocol_sha256=digest(out / 'protocol.json'),
                query_checkpoint_sha256=args.query_sha256, selector_checkpoint_sha256=args.selector_sha256,
                full_train_fit_verified=full_fit is not None, final_selection_sha256=selected_sha,
                job_records=[dict(path=str((out / 'job_records' / r['job_key'] / 'receipt.json').relative_to(ROOT)),
                    sha256=digest(out / 'job_records' / r['job_key'] / 'receipt.json')) for r in results],
                **aggregate(results))
            atomic_json(out / 'summary.json', summary)
            print('FULL_HELD_ROLE_GEOMETRY_COMPLETE', args.role, summary['categories']['overall'], flush=True)
        except Exception:
            atomic_json(out / 'failure.json', dict(status='INCOMPLETE_EVALUATION_NOT_PROMOTED', role=args.role,
                traceback=traceback.format_exc(), complete_jobs=len(results)))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--query-checkpoint', required=True)
    parser.add_argument('--query-sha256', required=True)
    parser.add_argument('--selector-checkpoint', required=True)
    parser.add_argument('--selector-sha256', required=True)
    parser.add_argument('--selector-protocol', required=True)
    parser.add_argument('--query-training-receipt')
    parser.add_argument('--selector-training-receipt')
    parser.add_argument('--role', choices=ROLES, required=True)
    parser.add_argument('--final-selection')
    parser.add_argument('--allow-partial-training', action='store_true')
    parser.add_argument('--wait-ready', action='store_true')
    main(parser.parse_args())
