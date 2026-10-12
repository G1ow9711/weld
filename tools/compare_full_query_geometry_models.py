"""Compare all TRAIN query geometry on identical deployed dense assignments.

Scores are per-child diagnostics with reference-assigned responsibilities, not
independent full-output accuracy. All registered views and every child are retained.
The original complete model is assessed now; the new arm waits for actual full
training completion and its verified receipt before assessment.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import os
from pathlib import Path
import time
import traceback
from types import FunctionType

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from tools.diagnose_full_query_geometry import atomic, predict_assigned, geometry_check
from tools.export_full_dataset_split_view import local_name
from tools.prepare_full_task_supervision import verify
from tools.qualify_full_dataset_labels import ply_points
from weld_pipeline.server_deployed_query_assignments import DeployedGridTrainingAssignments, eligible_query_positions
from weld_pipeline.server_full_prediction_runtime import CompleteQueryProposals, carrier_segment, free_segment
from weld_pipeline.server_local_query_checkpoint import load_local_query_model, verify_complete_local_query_fit
from weld_pipeline.server_local_query_runtime import LocalCompleteQueryProposals
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.server_partitioned_query_fit import verifier
from weld_pipeline.server_query_checkpoint_model import load_query_model
from weld_pipeline.server_runtime_aligned_training_samples import FIXED_QUERY_FRACTIONS
from weld_pipeline.sjg_local_runtime_904 import digest, read, write
from weld_pipeline.world_set_evaluation_285 import reference_segment


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['tools/compare_full_query_geometry_models.py', 'tools/diagnose_full_query_geometry.py',
    'weld_pipeline/server_deployed_query_assignments.py', 'weld_pipeline/server_local_query_runtime.py',
    'weld_pipeline/server_local_query_checkpoint.py', 'weld_pipeline/server_query_point_neighborhood.py',
    'weld_pipeline/server_full_prediction_runtime.py', 'weld_pipeline/server_full_training_samples.py',
    'weld_pipeline/server_partitioned_full_dataset.py', 'tools/qualify_full_dataset_labels.py']


def process(pid):
    folder = Path('/proc') / str(pid)
    values = folder.joinpath('stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, state=values[0], start_ticks=values[19],
        argv=folder.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0'))


def assess_arm(dataset, base, out, state, localized, seed, model_sha):
    model, architecture = (load_local_query_model(state) if localized else load_query_model(state))
    scorer = FunctionType(predict_assigned.__code__,
        dict(predict_assigned.__globals__, fixed_query_positions=eligible_query_positions), predict_assigned.__name__)
    arm = out / ('new_local' if localized else 'original')
    arm.mkdir(exist_ok=False)
    totals, categories, records, assignment_fingerprints = Counter(), defaultdict(Counter), [], {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for job_key, job in dataset.jobs.items():
        with threadpool_limits(limits=2):
            inp = dataset.input_case(job_key)
            builder = DeployedGridTrainingAssignments(inp)
            if localized:
                raw = ROOT.parent / 'data/G_Data' / job['workpiece_source']
                if digest(raw) != job['workpiece_sha256']:
                    raise ValueError('Original full raw XYZ changed before diagnostic inference')
                xyz = ply_points(raw)
                if len(xyz) != job['original_points']:
                    raise ValueError('Complete raw XYZ points were omitted')
                runtime = LocalCompleteQueryProposals(model, state['mean'], state['std'], xyz,
                    device=device, fixed_query_fractions=FIXED_QUERY_FRACTIONS)
            else:
                runtime = CompleteQueryProposals(model, state['mean'], state['std'],
                    device=device, fixed_query_fractions=FIXED_QUERY_FRACTIONS)
            for version in job['versions']:
                key = version['key']; supervision = dataset.supervision(key)
                folder = base / 'full_task_supervision_v1' / job_key / key
                verify(folder, dataset, version)
                with np.load(folder / 'owner_witnesses.npz', allow_pickle=False) as arrays:
                    witnesses = {name:arrays[name] for name in arrays.files}
                sample = builder.build(supervision, witnesses, seed)
                assignment = hashlib.sha256(sample['profile_ids'].tobytes() +
                                            sample['input_query_positions_mm'].tobytes()).hexdigest()
                assignment_fingerprints[key] = assignment
                predictions = scorer(runtime, inp, sample)
                statistics = Counter(children=len(predictions), strong=sample['report']['strong_carrier_assignments'],
                                     weak=sample['report']['weak_free_geometry_assignments'])
                rows, owner = [], 0
                for parent in supervision.parents:
                    for child in parent['children']:
                        prediction, point, basis = predictions[owner]
                        pid = int(sample['profile_ids'][owner])
                        reference = reference_segment(dict(child, world_representation=child['representation']))
                        category = child['weld_type'] + '/' + child['original_kind']
                        row = dict(original_child_id=child['original_child_id'], category=category,
                            profile_id=pid, query_position_mm=float(sample['input_query_positions_mm'][owner]))
                        for branch in ('carrier', 'free'):
                            try:
                                segment = (carrier_segment(inp.curve(pid), prediction['interval_ratios'],
                                    bool(prediction['direction'].argmax())) if branch == 'carrier'
                                           else free_segment(prediction, point, basis))
                                segment['weld_type'] = 'vertical' if prediction['weld_type'].argmax() else 'horizontal'
                                row[branch] = geometry_check(segment, child, reference,
                                    parent['is_closed'] and len(parent['children']) == 1)
                            except (ValueError, OverflowError, FloatingPointError) as error:
                                row[branch] = dict(valid=False, reason=str(error),
                                    geometry_and_type_pass_with_individual_child_reversal=False)
                            statistics[branch + '_diagnostic_pass'] += int(
                                row[branch]['geometry_and_type_pass_with_individual_child_reversal'])
                        passed = any(row[branch]['geometry_and_type_pass_with_individual_child_reversal']
                                     for branch in ('carrier', 'free'))
                        statistics['either_diagnostic_pass'] += int(passed)
                        categories[category].update(children=1, either_diagnostic_pass=int(passed))
                        rows.append(row); owner += 1
                if owner != supervision.target['original_children']:
                    raise ValueError('A complete original child was omitted')
                totals.update(statistics); totals['versions'] += 1
                path = arm / (key + '.json')
                write(path, dict(version_key=key, model_sha256=model_sha,
                    assignment_fingerprint=assignment, statistics=dict(statistics), children=rows))
                records.append(dict(version_key=key, path=str(path.relative_to(ROOT)), sha256=digest(path)))
                atomic(out / 'progress.json', dict(status='COMPLETE_TRAIN_GEOMETRY_COMPARISON_RUNNING',
                    controller_pid=os.getpid(), arm=arm.name,
                    complete_versions=len(records), registered_versions=len(dataset.versions),
                    complete_children=totals['children'], held_reference_reads=0, unix_time=time.time()))
                print('COMPLETE_TRAIN_GEOMETRY_ARM', arm.name, len(records), '/', len(dataset.versions),
                      dict(statistics), flush=True)
                del sample, predictions, witnesses, supervision, rows
            del inp, builder, runtime
            if localized:
                del xyz
            gc.collect()
    if len(records) != len(dataset.versions):
        raise ValueError('A registered complete TRAIN view was omitted')
    result = dict(architecture=architecture, model_sha256=model_sha, totals=dict(totals),
        categories={key:dict(value) for key,value in categories.items()}, records=records,
        assignment_fingerprints=assignment_fingerprints, held_reference_reads=0, optimizer_updates=0,
        reference_assigned_child_diagnostic_not_deployment_output_accuracy=True)
    write(arm / 'receipt.json', result)
    return result


def main(args):
    torch.set_num_threads(2)
    base = ROOT / 'server_rebuild' / local_name(args.dataset_name)
    dataset = PartitionedFullShardDataset(ROOT, base)
    original = base / 'full_weld_reference_recall_v3/last.pt'
    original_receipt = read(original.parent / 'receipt.json')
    original_sha = digest(original)
    if original_sha != original_receipt['checkpoint_sha256']:
        raise ValueError('The complete original query checkpoint changed')
    state = torch.load(original, map_location='cpu', weights_only=False)
    original_fit = verifier(ROOT, base)(ROOT, state, original_sha, original.parent / 'receipt.json')
    new = base / local_name(args.new_training_name)
    identity = process(args.new_training_pid)
    if identity['state'] in ('Z', 'X') or new.name not in identity['argv']:
        raise ValueError('The declared actual new training process is absent or owns another run')
    out = base / local_name(args.experiment_name); out.mkdir(exist_ok=False)
    sources = {name:digest(ROOT / name) for name in SOURCES}
    write(out / 'protocol.json', dict(sources=sources, original_model_sha256=original_sha,
        original_complete_query_fit=original_fit, new_training_identity=identity,
        training_registry_sha256=dataset.registry_sha256, assignment_seed=args.seed,
        same_actual_deployed_dense_queries_for_both_models=True, all_registered_views_required=True,
        held_reference_reads=0, optimizer_updates=0, full_output_accuracy_not_claimed=True))
    try:
        before = assess_arm(dataset, base, out, state, False, args.seed, original_sha)
        while not (new / 'receipt.json').exists():
            if (new / 'failure.json').exists():
                raise RuntimeError('New full query training actually failed')
            current = process(args.new_training_pid)
            if current['state'] in ('Z', 'X') or current['start_ticks'] != identity['start_ticks']:
                raise RuntimeError('The original new training handle stopped before its completion receipt')
            if sources != {name:digest(ROOT / name) for name in sources}:
                raise ValueError('Complete query comparison source changed')
            atomic(out / 'progress.json', dict(status='ORIGINAL_ALL_TRAIN_DIAGNOSED_WAITING_LIVE_NEW_TRAINING',
                controller_pid=os.getpid(), new_training_pid=args.new_training_pid,
                original_complete_views=before['totals']['versions'], new_optimizer_updates=0,
                held_reference_reads=0, unix_time=time.time()))
            time.sleep(10)
        complete = read(new / 'receipt.json'); sha = digest(new / 'last.pt')
        if sha != complete['checkpoint_sha256']:
            raise ValueError('Actual new completed query checkpoint differs from its receipt')
        newer = torch.load(new / 'last.pt', map_location='cpu', weights_only=False)
        fit = verify_complete_local_query_fit(ROOT, base, newer, sha, new / 'receipt.json')
        after = assess_arm(dataset, base, out, newer, True, args.seed, sha)
        if before['assignment_fingerprints'] != after['assignment_fingerprints']:
            raise ValueError('The two models were compared at different actual input query assignments')
        if sources != {name:digest(ROOT / name) for name in sources}:
            raise ValueError('Complete comparison source changed')
        atomic(out / 'receipt.json', dict(status='ALL_TRAIN_QUERY_MODELS_COMPARED_AT_IDENTICAL_DENSE_INPUT_QUERIES',
            recorded_utc=datetime.now(timezone.utc).isoformat(), original=before['totals'], new=after['totals'],
            original_categories=before['categories'], new_categories=after['categories'],
            new_complete_query_fit=fit, query_assignment_fingerprints_identical=True,
            original_model_sha256=original_sha, new_model_sha256=sha,
            held_reference_reads=0, optimizer_updates=0, completed_optimization_rounds=0,
            complete_parent_and_deployment_output_quality_not_evaluated=True, full_contract_pass=False))
    except BaseException:
        atomic(out / 'failure.json', dict(traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', default='full_dataset_v2')
    parser.add_argument('--new-training-name', default='full_local_query_geometry_grid_v4_round01')
    parser.add_argument('--new-training-pid', type=int, required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--seed', type=int, default=1031)
    main(parser.parse_args())
