"""Compare two verified complete TRAIN diagnostics without rerunning inference.

This measures reference-assigned individual child geometry, with either branch
and reversal allowed. It never claims whole-parent extraction or TEST accuracy.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import time
import traceback

import torch

from tools.compare_full_query_geometry_models import process
from tools.diagnose_full_query_geometry import atomic
from tools.export_full_dataset_split_view import local_name
from weld_pipeline.server_interval_endpoint_checkpoint import verify_complete_interval_endpoint_fit
from weld_pipeline.server_interval_endpoint_residual_model import ARCHITECTURE as ENDPOINT_ARCHITECTURE
from weld_pipeline.server_local_query_checkpoint import verify_complete_local_query_fit
from weld_pipeline.server_local_query_weld_model import ARCHITECTURE as LOCAL_ARCHITECTURE
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.sjg_local_runtime_904 import digest, read


ROOT = Path(__file__).resolve().parents[1]
IDENTITY_FIELDS = ('original_child_id', 'category', 'profile_id', 'query_position_mm')
PASS_FIELD = 'geometry_and_type_pass_with_individual_child_reversal'


def verified_arm(base, dataset, comparison_name, arm_name, training_name):
    folder = base / local_name(comparison_name)
    comparison = read(folder / 'receipt.json')
    protocol = read(folder / 'protocol.json')
    receipt_path = folder / local_name(arm_name) / 'receipt.json'
    arm = read(receipt_path)
    if (comparison['status'] not in {
            'ALL_TRAIN_QUERY_MODELS_COMPARED_AT_IDENTICAL_DENSE_INPUT_QUERIES',
            'ALL_TRAIN_NEW_QUERY_COMPARED_WITH_VERIFIED_CACHED_MODELS'}
            or not comparison['query_assignment_fingerprints_identical']
            or protocol['training_registry_sha256'] != dataset.registry_sha256
            or protocol['sources'] != {n:digest(ROOT / n) for n in protocol['sources']}
            or arm['held_reference_reads'] != 0 or arm['optimizer_updates'] != 0
            or not arm['reference_assigned_child_diagnostic_not_deployment_output_accuracy']):
        raise ValueError('A current complete TRAIN diagnostic with unchanged sources is required')
    model_folder = base / local_name(training_name)
    checkpoint = model_folder / 'last.pt'
    model_sha = digest(checkpoint)
    if model_sha != arm['model_sha256']:
        raise ValueError('The saved predictions belong to a different actual model')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    verifier = {LOCAL_ARCHITECTURE:verify_complete_local_query_fit,
                ENDPOINT_ARCHITECTURE:verify_complete_interval_endpoint_fit}.get(state.get('architecture'))
    if verifier is None or arm['architecture'] != state['architecture']:
        raise ValueError('Only an explicitly supported complete model can enter this comparison')
    fit = verifier(ROOT, base, state, model_sha, model_folder / 'receipt.json')
    records = {}
    totals = Counter()
    categories = defaultdict(Counter)
    if set(arm['assignment_fingerprints']) != set(dataset.versions):
        raise ValueError('Every registered TRAIN assignment must be present')
    for item in arm['records']:
        key = item['version_key']
        if key in records or key not in dataset.versions:
            raise ValueError('Duplicate or unregistered diagnostic version')
        path = (ROOT / item['path']).resolve(strict=True)
        path.relative_to(receipt_path.parent.resolve())
        if digest(path) != item['sha256']:
            raise ValueError('An immutable saved prediction record changed')
        record = read(path)
        if (record['version_key'] != key or record['model_sha256'] != model_sha
                or record['assignment_fingerprint'] != arm['assignment_fingerprints'][key]
                or len(record['children']) != dataset.versions[key][1]['children']):
            raise ValueError('A saved view differs from its model, assignment or complete child count')
        child_ids = set()
        for child in record['children']:
            if child['original_child_id'] in child_ids:
                raise ValueError('A registered child was duplicated')
            child_ids.add(child['original_child_id'])
            if not math.isfinite(child['query_position_mm']):
                raise ValueError('The actual input query position must be finite')
            flags = [child[b][PASS_FIELD] for b in ('carrier', 'free')]
            if any(type(flag) is not bool for flag in flags):
                raise ValueError('Saved geometry pass flags must be explicit booleans')
            totals.update(children=1, carrier_diagnostic_pass=int(flags[0]),
                          free_diagnostic_pass=int(flags[1]), either_diagnostic_pass=int(any(flags)))
            categories[child['category']].update(children=1, either_diagnostic_pass=int(any(flags)))
        records[key] = record
    totals['versions'] = len(records)
    if (set(records) != set(dataset.versions)
            or any(arm['totals'][k] != v for k,v in totals.items())
            or dict(categories) != arm['categories']):
        raise ValueError('The complete saved records disagree with their aggregate diagnostic')
    return dict(arm=arm, records=records, fit=fit, diagnostic_receipt_sha256=digest(receipt_path),
                comparison_receipt_sha256=digest(folder / 'receipt.json'),
                protocol_sha256=digest(folder / 'protocol.json'))


def paired_changes(before, after):
    if (set(before['records']) != set(after['records'])
            or before['arm']['assignment_fingerprints'] != after['arm']['assignment_fingerprints']):
        raise ValueError('The actual complete input assignments differ')
    changes = Counter()
    categories = defaultdict(Counter)
    for key, record in before['records'].items():
        previous, current = record['children'], after['records'][key]['children']
        if len(previous) != len(current):
            raise ValueError('The complete child denominators differ')
        for a,b in zip(previous, current):
            if any(a[field] != b[field] for field in IDENTITY_FIELDS):
                raise ValueError('The original child, category, profile or input query responsibility differs')
            was = any(a[branch][PASS_FIELD] for branch in ('carrier', 'free'))
            now = any(b[branch][PASS_FIELD] for branch in ('carrier', 'free'))
            label = ('both_pass' if now else 'regressed_to_fail') if was else (
                     'improved_to_pass' if now else 'both_fail')
            changes[label] += 1
            categories[a['category']][label] += 1
    return dict(changes), {key:dict(value) for key,value in categories.items()}


def main(args):
    torch.set_num_threads(2)
    base = ROOT / 'server_rebuild' / local_name(args.dataset_name)
    dataset = PartitionedFullShardDataset(ROOT, base)
    before = verified_arm(base, dataset, args.before_comparison, args.before_arm, args.before_training)
    after_folder = base / local_name(args.after_comparison)
    identity = None
    if not (after_folder / 'receipt.json').exists():
        if args.after_comparison_pid is None:
            raise ValueError('The exact live diagnostic PID is required while results are pending')
        identity = process(args.after_comparison_pid)
        if identity['state'] in ('Z','X') or after_folder.name not in identity['argv']:
            raise ValueError('The declared complete diagnostic handle is not live or is different')
    out = base / local_name(args.experiment_name)
    out.mkdir(exist_ok=False)
    source_sha = digest(__file__)
    atomic(out / 'protocol.json', dict(arguments=vars(args), source_sha256=source_sha,
        training_registry_sha256=dataset.registry_sha256, after_diagnostic_identity=identity,
        before_diagnostic_receipt_sha256=before['diagnostic_receipt_sha256'],
        held_reference_reads=0, model_fit_calls=0, inference_calls=0))
    try:
        while not (after_folder / 'receipt.json').exists():
            if (after_folder / 'failure.json').exists():
                raise RuntimeError('The registered diagnostic failed before completion')
            current = process(args.after_comparison_pid)
            if current['state'] in ('Z','X') or current['start_ticks'] != identity['start_ticks']:
                raise RuntimeError('The same diagnostic process stopped before its complete receipt')
            if digest(__file__) != source_sha:
                raise ValueError('The bound comparison source changed')
            atomic(out / 'progress.json', dict(status='WAITING_SAME_LIVE_COMPLETE_QUERY_DIAGNOSTIC',
                controller_pid=os.getpid(), after_comparison_pid=args.after_comparison_pid,
                held_reference_reads=0, model_fit_calls=0, inference_calls=0, unix_time=time.time()))
            time.sleep(10)
        # Recheck the old immutable evidence after waiting, then check the new arm.
        before = verified_arm(base, dataset, args.before_comparison, args.before_arm, args.before_training)
        after = verified_arm(base, dataset, args.after_comparison, args.after_arm, args.after_training)
        changes, categories = paired_changes(before, after)
        if digest(__file__) != source_sha:
            raise ValueError('The bound comparison source changed after checking saved records')
        atomic(out / 'receipt.json', dict(status='ALL_TRAIN_SAVED_QUERY_DIAGNOSTICS_PAIRED_NOT_EXTRACTION_ACCURACY',
            utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=digest(out / 'protocol.json'),
            before={k:v for k,v in before.items() if k not in ('arm','records')},
            after={k:v for k,v in after.items() if k not in ('arm','records')},
            before_model_sha256=before['arm']['model_sha256'], after_model_sha256=after['arm']['model_sha256'],
            before_totals=before['arm']['totals'], after_totals=after['arm']['totals'],
            before_categories=before['arm']['categories'], after_categories=after['arm']['categories'],
            paired_changes=changes, paired_changes_by_category=categories,
            all_registered_prediction_record_hashes_and_child_responsibilities_verified=True,
            query_assignment_fingerprints_identical=True, held_reference_reads=0,
            model_fit_calls=0, inference_calls=0, complete_optimization_rounds_added=0,
            reference_assigned_child_reversal_and_best_branch_diagnostic_not_extraction_accuracy=True,
            complete_parent_selection_fields_and_held_quality_not_evaluated=True, full_contract_pass=False))
    except BaseException:
        atomic(out / 'failure.json', dict(traceback=traceback.format_exc(), full_contract_pass=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', default='full_dataset_v2')
    parser.add_argument('--experiment-name', required=True)
    for side in ('before','after'):
        parser.add_argument('--' + side + '-comparison', required=True)
        parser.add_argument('--' + side + '-training', required=True)
        parser.add_argument('--' + side + '-arm', default='new_local')
    parser.add_argument('--after-comparison-pid', type=int)
    main(parser.parse_args())
