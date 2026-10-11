"""TRAIN-only diagnosis of query coverage and fitted endpoint errors.

This uses existing supervision assignments after actual input predictions have
been frozen. Assigned queries are an offline diagnostic, never an extracted
output or deployment feature. No fitting or reference-coordinate insertion is
performed, and no held-out label is opened.
"""
import argparse
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.contract_curve_geometry import CircularArc
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.server_full_training_samples import FullTrainingAssignments
from weld_pipeline.server_full_weld_model import FullWeldModel
from weld_pipeline.server_full_prediction_runtime import carrier_segment, free_segment
from weld_pipeline.server_joint_reference_query_model import JointReferenceModel
from weld_pipeline.sjg_local_runtime_904 import digest, write, read
from tools.prepare_full_predicted_parent_pool import load_gzip
from tools.train_full_weld_warmstart import TrainingTensors, load_query_initialization

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'


def endpoint_error(segment, reference):
    points = np.asarray(segment['points'])[[0, -1]]
    if reference['original_kind'] == 'line':
        target = np.asarray(reference['representation']['endpoints_mm'])
    else:
        arc = CircularArc.from_parameters(reference['representation']['arc_parameters'])
        target = arc.point_at(np.array([0., arc.length_mm]))
    direct = np.linalg.norm(points - target, axis=1).max()
    reverse = np.linalg.norm(points[::-1] - target, axis=1).max()
    return dict(ordered_max_mm=float(direct), either_direction_max_mm=float(min(direct, reverse)))


def main(job_key, pool_name, seeds, output_name):
    start = time.perf_counter()
    torch.set_num_threads(2)
    dataset = FullShardDataset(ROOT)
    output = BASE / output_name
    if output.exists():
        raise ValueError('Diagnosis evidence is immutable; choose a new output name')
    pool = BASE / pool_name / 'pools' / job_key
    receipt = read(pool / 'receipt.json')
    assert receipt['geometry_GT_reads'] == 0 and receipt['STEP_reads'] == 0
    for name, sha in receipt['files'].items():
        assert digest(pool / name) == sha
    query_checkpoint = BASE / pool_name / 'query_checkpoint.pt'
    assert digest(query_checkpoint) == receipt['query_checkpoint_sha256']
    state = torch.load(query_checkpoint, map_location='cpu', weights_only=False)
    model = FullWeldModel().eval()
    model.load_state_dict(state['state_dict'])
    converter = TrainingTensors(state['mean'], state['std'], 'cpu')
    initial = load_query_initialization()
    old = JointReferenceModel().eval()
    old.load_state_dict(initial['state_dict'])
    old_converter = TrainingTensors(initial['mean'], initial['std'], 'cpu')
    proposals = load_gzip(pool / 'child_proposals.json.gz')
    by_profile = {}
    for child in proposals:
        by_profile.setdefault(child['profile'], set()).add(child['query_position_mm'])
    inp = dataset.input_case(job_key)
    builder = FullTrainingAssignments(inp)
    records = []
    with threadpool_limits(limits=2), torch.inference_mode():
        for version in dataset.jobs[job_key]['versions']:
            key = version['key']
            folder = BASE / 'full_task_supervision_v1' / job_key / key
            supervision = dataset.supervision(key)
            children = [child for parent in supervision.parents for child in parent['children']]
            with np.load(folder / 'owner_witnesses.npz', allow_pickle=False) as data:
                witnesses = {name: data[name] for name in data.files}
            view_evidence = state['view_evidence'].get('0:' + key)
            for step in range(seeds):
                seed = np.random.SeedSequence([1031, 0, list(dataset.versions).index(key), step])
                sample = builder.build(supervision, witnesses, seed)
                batch = converter.batch(inp, sample)
                hidden = model.encode_queries(**batch)
                prediction = {name: values.cpu().numpy() for name, values in model.predict(
                    hidden, converter.tensor(sample['features'][:, 456])).items()}
                old_output = old(old_converter.features(sample['features']),
                                 old_converter.tensor(sample['features'][:, 456]))
                old_bounds = old_output[0].cpu().numpy()
                old_types = old_output[1].cpu().numpy().argmax(1)
                for owner, reference in enumerate(children):
                    pid = int(sample['profile_ids'][owner])
                    query = float(sample['input_query_positions_mm'][owner])
                    actual_queries = np.array(sorted(by_profile.get(pid, set())))
                    nearest = float(abs(actual_queries - query).min()) if len(actual_queries) else None
                    values = {name: array[owner] for name, array in prediction.items()}
                    geometry_errors = {}
                    try:
                        # Legacy query model has no direction head. Both
                        # endpoint orders are reported, not silently aligned.
                        geometry_errors['legacy_carrier_at_assigned_query'] = endpoint_error(
                            carrier_segment(inp.curve(pid), old_bounds[owner], False), reference)
                    except ValueError as exc:
                        geometry_errors['legacy_carrier_at_assigned_query'] = dict(invalid=str(exc))
                    for branch in ('carrier', 'free'):
                        try:
                            segment = (carrier_segment(inp.curve(pid), values['interval_ratios'],
                                       bool(values['direction'].argmax())) if branch == 'carrier'
                                       else free_segment(values, sample['input_query_xyz'][owner],
                                                         sample['input_query_bases'][owner]))
                            geometry_errors[branch + '_at_assigned_query'] = endpoint_error(segment, reference)
                        except (ValueError, OverflowError) as exc:
                            geometry_errors[branch + '_at_assigned_query'] = dict(invalid=str(exc))
                    records.append(dict(version_key=key, seed_step=step, owner=owner,
                        original_kind=reference['original_kind'], weld_type=reference['weld_type'],
                        profile=pid, input_carrier_kind=inp.curve(pid).kind,
                        strong_carrier_assignment=bool(sample['report']['strong_carrier_assignments'] == len(children))
                          if len(children) == 1 else bool(np.any(witnesses['attribute_row'] == owner)),
                        query_position_mm=query, nearest_actual_retained_query_distance_mm=nearest,
                        query_in_actual_retained_pool=nearest is not None and nearest < 1e-8,
                        actual_query_count_on_profile=len(actual_queries),
                        predicted_kind='arc' if values['kind'].argmax() else 'line',
                        predicted_weld_type='vertical' if values['weld_type'].argmax() else 'horizontal',
                        legacy_predicted_weld_type='vertical' if old_types[owner] else 'horizontal',
                        geometry_errors=geometry_errors,
                        fitted_before_query_checkpoint=view_evidence is not None,
                        fit_loss_first=view_evidence['fit_results'][0] if view_evidence else None,
                        fit_loss_last=view_evidence['fit_results'][-1] if view_evidence else None))
    summary = {}
    for branch in ('legacy_carrier', 'carrier', 'free'):
        errors = [row['geometry_errors'][branch + '_at_assigned_query'].get('either_direction_max_mm')
                  for row in records]
        errors = [value for value in errors if value is not None]
        summary[branch] = dict(valid=len(errors), median_endpoint_error_mm=float(np.median(errors)) if errors else None,
                               min_endpoint_error_mm=min(errors) if errors else None,
                               endpoint_distance_at_most_3mm=sum(value <= 3 for value in errors),
                               endpoint_only_test_not_original_D3=True)
    write(output, dict(status='TRAIN_ASSIGNED_QUERY_GEOMETRY_DIAGNOSIS_NOT_OUTPUT_ACCURACY',
        job_key=job_key, source_sha256=digest(__file__), query_checkpoint_sha256=digest(query_checkpoint),
        query_checkpoint_optimizer_steps=state['optimizer_steps'],
        frozen_actual_pool_receipt_sha256=digest(pool / 'receipt.json'), seeds=seeds,
        query_assignment_samples=len(records), runtime_retained_assignment_queries=sum(
            row['query_in_actual_retained_pool'] for row in records), geometry_summary=summary, records=records,
        model_fit_calls=0, held_or_protected_label_reads=0,
        teacher_queries_are_not_runtime_predictions=True, actual_input_pool_not_modified=True,
        reference_coordinates_not_inserted_into_input_features=True, seconds=time.perf_counter()-start))
    print('ASSIGNED_QUERY_GEOMETRY_DIAGNOSIS', len(records), summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--pool-name', default='full_predicted_parent_initial_v1')
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--output-name', required=True)
    args = parser.parse_args()
    if args.seeds < 1:
        raise ValueError('A positive diagnosis sample count is required')
    main(args.job, args.pool_name, args.seeds, args.output_name)
