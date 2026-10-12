"""Locate geometry failures at deployed queries for every approved TRAIN view.

References assign diagnostic responsibilities only. The inference function
receives the original input case and its existing fixed query coordinates.
This is neither an independent output score nor a model-selection result.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from tools.export_full_dataset_split_view import local_name
from tools.prepare_full_task_supervision import verify
from weld_pipeline.contract_geometry_d3_409 import child_error
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment
from weld_pipeline.server_full_prediction_runtime import (
    CompleteQueryProposals, carrier_segment, free_segment)
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.server_partitioned_query_fit import verifier
from weld_pipeline.server_query_checkpoint_model import load_query_model
from weld_pipeline.server_runtime_aligned_training_samples import (
    RuntimeAlignedTrainingAssignments, FIXED_QUERY_FRACTIONS, fixed_query_positions)
from weld_pipeline.sjg_local_runtime_904 import digest, read, write
from weld_pipeline.world_set_evaluation_285 import reference_segment


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['tools/diagnose_full_query_geometry.py',
    'weld_pipeline/server_full_prediction_runtime.py',
    'weld_pipeline/server_runtime_aligned_training_samples.py',
    'weld_pipeline/server_full_training_samples.py',
    'weld_pipeline/server_reference_recall_weld_model.py',
    'weld_pipeline/server_legacy_anchored_full_weld_model.py',
    'weld_pipeline/server_full_weld_model.py',
    'weld_pipeline/server_partitioned_full_dataset.py',
    'weld_pipeline/server_partitioned_query_fit.py',
    'weld_pipeline/contract_geometry_d3_409.py']


def atomic(path, value):
    temporary = path.with_name(path.name + '.incomplete')
    write(temporary, value)
    os.replace(temporary, path)


def geometry_check(segment, child, reference, closed_parent):
    """Child reversal diagnoses geometry only; parent correspondence is untested."""
    predicted = np.asarray(segment['points'])[[0, -1]]
    target = np.asarray(reference['points'])[[0, -1]]
    endpoint_error = float(min(np.linalg.norm(predicted - target, axis=1).max(),
                               np.linalg.norm(predicted[::-1] - target, axis=1).max()))
    variants = (segment, dict(segment, **reverse_segment(segment)))
    errors = [child_error(candidate, child, reference, closed_parent, False)
              for candidate in variants]
    accepted = [error for error in errors if error is not None]
    return dict(valid=True, kind=segment['kind'], weld_type=segment['weld_type'],
        kind_correct=segment['kind'] == child['original_kind'],
        weld_type_correct=segment['weld_type'] == child['weld_type'],
        unordered_endpoint_maximum_mm=endpoint_error,
        geometry_and_type_pass_with_individual_child_reversal=bool(accepted),
        matched_whole_line_error_mm=min((value['bidirectional_sampled_maximum_mm']
                                        for value in accepted), default=None),
        individual_child_reversal_is_not_complete_parent_acceptance=True)


def predict_assigned(runtime, inp, sample):
    """Use precisely the public input-only scorer used by CompleteQueryProposals."""
    outputs = [None] * len(sample['profile_ids'])
    for pid in np.unique(sample['profile_ids']):
        pid = int(pid)
        owners = np.flatnonzero(sample['profile_ids'] == pid)
        positions = sample['input_query_positions_mm'][owners]
        unique, inverse = np.unique(positions, return_inverse=True)
        allowed = fixed_query_positions(inp, pid)
        if not np.isin(unique, allowed).all():
            raise ValueError('A reference created a query that deployment does not score')
        tokens = torch.tensor(inp.tokens(pid)[0], device=runtime.device, dtype=torch.float32)
        values, frames = runtime.score_queries(inp, pid, unique, tokens)
        for owner, index in zip(owners, inverse):
            point, basis, _ = frames[int(index)]
            if (not np.array_equal(point, sample['input_query_xyz'][owner])
                    or not np.array_equal(basis, sample['input_query_bases'][owner])):
                raise ValueError('Diagnostic frame differs from the deployed input frame')
            outputs[int(owner)] = ({name: value[index] for name, value in values.items()}, point, basis)
    return outputs


def main(args):
    torch.set_num_threads(2)
    base = ROOT / 'server_rebuild' / local_name(args.dataset_name)
    dataset = PartitionedFullShardDataset(ROOT, base)
    checkpoint = (ROOT / args.checkpoint).resolve(strict=True)
    checkpoint.relative_to(base)
    if digest(checkpoint) != args.checkpoint_sha256:
        raise ValueError('The immutable complete query checkpoint changed')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    fit = verifier(ROOT, base)(ROOT, state, args.checkpoint_sha256, checkpoint.parent / 'receipt.json')
    model, architecture = load_query_model(state)
    runtime = CompleteQueryProposals(model, state['mean'], state['std'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        fixed_query_fractions=FIXED_QUERY_FRACTIONS, mode='fixed_and_grid')
    out = base / local_name(args.experiment_name)
    out.mkdir(exist_ok=False)
    sources = {name: digest(ROOT / name) for name in SOURCES}
    protocol = dict(architecture=architecture, checkpoint=str(checkpoint.relative_to(ROOT)),
        checkpoint_sha256=args.checkpoint_sha256, sources=sources,
        training_registry_sha256=dataset.registry_sha256, full_query_training=fit,
        registered_versions=sorted(dataset.versions), registered_jobs=sorted(dataset.jobs),
        diagnostic_assignment_seed=args.seed, fixed_query_fractions=list(FIXED_QUERY_FRACTIONS),
        reference_geometry_used_only_for_training_responsibility_and_diagnostics=True,
        runtime_receives_no_labels_or_source_ids=True, held_reference_reads=0,
        optimizer_updates=0, independent_output_accuracy_not_claimed=True)
    write(out / 'protocol.json', protocol)
    started = time.monotonic()
    totals, by_kind, collisions, records = Counter(), defaultdict(Counter), [], []
    try:
        for job_key in dataset.jobs:
            with threadpool_limits(limits=2):
                inp = dataset.input_case(job_key)
                builder = RuntimeAlignedTrainingAssignments(inp)
                for version in dataset.jobs[job_key]['versions']:
                    key = version['key']
                    supervision = dataset.supervision(key)
                    folder = base / 'full_task_supervision_v1' / job_key / key
                    verify(folder, dataset, version)
                    with np.load(folder / 'owner_witnesses.npz', allow_pickle=False) as arrays:
                        witnesses = {name: arrays[name] for name in arrays.files}
                    sample = builder.build(supervision, witnesses, args.seed)
                    output = predict_assigned(runtime, inp, sample)
                    owners, groups = [], defaultdict(list)
                    statistics = Counter(children=len(output), strong=sample['report']['strong_carrier_assignments'],
                                         weak=sample['report']['weak_free_geometry_assignments'])
                    child_index = 0
                    for parent_index, parent in enumerate(supervision.parents):
                        for index, child in enumerate(parent['children']):
                            prediction, point, basis = output[child_index]
                            pid = int(sample['profile_ids'][child_index])
                            position = float(sample['input_query_positions_mm'][child_index])
                            reference = reference_segment(dict(child, world_representation=child['representation']))
                            kind = child['weld_type'] + '/' + child['original_kind']
                            closed = parent['is_closed'] and len(parent['children']) == 1
                            row = dict(parent_index=parent_index, child_index=index,
                                original_child_id=child['original_child_id'], profile_id=pid,
                                query_position_mm=position, category=kind,
                                interval_supervision_known=bool(sample['targets']['interval_known'][child_index]),
                                predicted_kind='arc' if prediction['kind'].argmax() else 'line',
                                predicted_weld_type='vertical' if prediction['weld_type'].argmax() else 'horizontal',
                                interval_ratios=np.asarray(prediction['interval_ratios']).tolist())
                            # This raw coordinate error is diagnostic even when the free
                            # kind head chooses an arc. No output is corrected using it.
                            raw_ends = point + np.asarray(prediction['line_points']) @ basis.T * 1000
                            true_ends = np.asarray(reference['points'])[[0, -1]]
                            row['raw_free_point_ordered_maximum_mm'] = float(np.linalg.norm(raw_ends - true_ends, axis=1).max())
                            for branch in ('carrier', 'free'):
                                try:
                                    segment = (carrier_segment(inp.curve(pid), prediction['interval_ratios'],
                                                 bool(prediction['direction'].argmax())) if branch == 'carrier'
                                               else free_segment(prediction, point, basis))
                                    segment['weld_type'] = row['predicted_weld_type']
                                    row[branch] = geometry_check(segment, child, reference, closed)
                                except (ValueError, OverflowError, FloatingPointError) as error:
                                    row[branch] = dict(valid=False, reason=str(error),
                                        geometry_and_type_pass_with_individual_child_reversal=False)
                                statistics[branch + '_child_diagnostic_pass'] += int(
                                    row[branch]['geometry_and_type_pass_with_individual_child_reversal'])
                            statistics['either_branch_child_diagnostic_pass'] += int(any(
                                row[branch]['geometry_and_type_pass_with_individual_child_reversal']
                                for branch in ('carrier', 'free')))
                            statistics['free_kind_correct'] += int(row['predicted_kind'] == child['original_kind'])
                            statistics['weld_type_correct'] += int(row['predicted_weld_type'] == child['weld_type'])
                            by_kind[kind].update(children=1,
                                either_branch_child_diagnostic_pass=int(any(
                                    row[branch]['geometry_and_type_pass_with_individual_child_reversal']
                                    for branch in ('carrier', 'free'))))
                            signature = hashlib.sha256(json.dumps(dict(representation=child['representation'],
                                weld_type=child['weld_type']), sort_keys=True).encode()).hexdigest()
                            groups[(pid, position)].append(dict(parent_index=parent_index, child_index=index,
                                reference_signature=signature))
                            owners.append(row)
                            child_index += 1
                    if child_index != supervision.target['original_children']:
                        raise ValueError('A TRAIN reference child was omitted')
                    for (pid, position), assignments in groups.items():
                        if len({row['reference_signature'] for row in assignments}) > 1:
                            collisions.append(dict(version_key=key, profile_id=pid,
                                query_position_mm=position, assignments=assignments))
                            statistics['distinct_targets_at_same_input_query'] += 1
                    totals.update(statistics)
                    totals['versions'] += 1
                    record = dict(version_key=key, job_key=job_key, label_receipt_sha256=version['label_receipt_sha256'],
                        input_receipt_sha256=digest(dataset.shards / job_key / 'receipt.json'),
                        witness_receipt_sha256=digest(folder / 'receipt.json'), statistics=dict(statistics),
                        children=owners, reference_assigned_child_diagnostics_not_deployment_predictions=True)
                    write(out / (key + '.json'), record)
                    records.append(dict(version_key=key, path=str((out / (key + '.json')).relative_to(ROOT)),
                                        sha256=digest(out / (key + '.json'))))
                    atomic(out / 'progress.json', dict(controller_pid=os.getpid(), status='DIAGNOSING_ALL_TRAIN_QUERY_GEOMETRY',
                        complete_versions=len(records), registered_versions=len(dataset.versions),
                        children=totals['children'], last_version=key, optimizer_updates=0,
                        held_reference_reads=0, unix_time=time.time()))
                    print('ALL_TRAIN_QUERY_GEOMETRY', len(records), '/', len(dataset.versions), dict(statistics), flush=True)
                    del sample, output, owners, witnesses, supervision
                del inp, builder
                gc.collect()
        if (len(records) != len(dataset.versions)
                or sources != {name: digest(ROOT / name) for name in sources}):
            raise ValueError('Incomplete TRAIN diagnostics or changed source implementation')
        write(out / 'query_responsibility_collisions.json', collisions)
        atomic(out / 'receipt.json', dict(status='ALL_REGISTERED_TRAIN_QUERY_GEOMETRY_DIAGNOSED_NOT_OUTPUT_ACCURACY',
            recorded_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=digest(out / 'protocol.json'),
            query_checkpoint_sha256=args.checkpoint_sha256, training_registry_sha256=dataset.registry_sha256,
            totals=dict(totals), by_category={key:dict(value) for key, value in by_kind.items()},
            records=records, optimizer_updates=0, held_reference_reads=0,
            references_assign_queries_but_never_enter_input_features=True,
            every_query_is_an_actual_deployed_fixed_fragment_query=True,
            selection_parent_structure_and_independent_accuracy_not_evaluated=True,
            completed_optimization_rounds=0, full_contract_pass=False,
            seconds=time.monotonic()-started))
        print('ALL_TRAIN_QUERY_GEOMETRY_FINISHED', dict(totals), flush=True)
    except BaseException:
        atomic(out / 'failure.json', dict(traceback=traceback.format_exc(), complete_versions=len(records)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', default='full_dataset_v2')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--seed', type=int, default=1031)
    main(parser.parse_args())
