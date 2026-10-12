"""Compare original and tolerance-scaled geometry learning on a complete TRAIN view.

Every original child, class, field mask, frame and parent/link target is used.
This bounded diagnostic is not a complete dataset optimization/evaluation round.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from tools.diagnose_full_query_geometry import atomic, predict_assigned, geometry_check
from tools.export_full_dataset_split_view import local_name
from tools.prepare_full_task_supervision import verify
from tools.train_full_weld_reference_recall import TrainingTensors, fit_step
from weld_pipeline.server_metric_geometry_losses import (
    metric_child_losses, point_distance_loss, arc_points)
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.server_partitioned_query_fit import verifier
from weld_pipeline.server_query_checkpoint_model import load_query_model
from weld_pipeline.server_reference_recall_weld_model import reference_responsibility_loss
from weld_pipeline.server_runtime_aligned_training_samples import RuntimeAlignedTrainingAssignments, FIXED_QUERY_FRACTIONS
from weld_pipeline.server_full_prediction_runtime import CompleteQueryProposals, carrier_segment, free_segment
from weld_pipeline.sjg_local_runtime_904 import digest, read, write
from weld_pipeline.world_set_evaluation_285 import reference_segment


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['tools/compare_metric_geometry_loss.py',
    'weld_pipeline/server_metric_geometry_losses.py',
    'tools/diagnose_full_query_geometry.py', 'tools/train_full_weld_reference_recall.py',
    'weld_pipeline/server_full_weld_model.py', 'weld_pipeline/server_full_prediction_runtime.py',
    'weld_pipeline/server_runtime_aligned_training_samples.py']


def controls():
    zero = torch.zeros(1, 2, 3)
    displacement = zero.clone(); displacement[:, :, 0] = .003
    value = point_distance_loss(displacement, zero)
    if not torch.allclose(value, torch.full_like(value, .5)):
        raise ValueError('A three-mm displacement was not expressed in the acceptance unit')
    rotated = displacement.clone(); rotated[:, :, 0] = 0; rotated[:, :, 1] = .003
    if not torch.equal(value, point_distance_loss(rotated, zero)):
        raise ValueError('Geometry loss depends on the query coordinate axis')
    shape = torch.tensor([[0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1.]])
    fractions = torch.linspace(0, 1, 65)
    points = arc_points(shape, fractions)
    shifted = shape.clone(); shifted[:, 2] = .003
    if not torch.allclose(point_distance_loss(arc_points(shifted, fractions), points),
                          torch.full((1, 65), .5), atol=1e-5):
        raise ValueError('Analytic arc positions do not use the same three-mm unit')
    zero.requires_grad_(); point_distance_loss(zero, torch.zeros_like(zero)).sum().backward()
    if not torch.isfinite(zero.grad).all() or torch.count_nonzero(zero.grad):
        raise ValueError('Exact geometry did not have a finite zero gradient')
    return dict(three_mm_unit_checked=True, endpoint_axis_invariance_checked=True,
        analytic_arc_translation_checked=True, exact_geometry_zero_gradient_checked=True,
        original_label_and_process_masks_retained=True, acceptance_gate_changed=False)


def metric_step(model, converter, inp, sample, optimizer, loss_mass):
    batch = converter.batch(inp, sample)
    hidden = model.encode_queries(**batch)
    prediction = model.predict(hidden, converter.tensor(sample['features'][:, 456]))
    targets = {name: converter.tensor(value) for name, value in sample['targets'].items()}
    losses = metric_child_losses(prediction, targets, batch['geometry'])
    losses.update(reference_responsibility_loss(prediction))
    losses['parent_closure'] = F.cross_entropy(model.parent_logits(hidden, sample['parent_children']),
                                              converter.tensor(sample['parent_closed']))
    if len(sample['link_left']):
        link = model.link_logits(hidden, converter.tensor(sample['link_left']),
            converter.tensor(sample['link_right']), converter.tensor(sample['link_relative']))
        labels = converter.tensor(sample['link_labels'])
        error = F.cross_entropy(link, labels, reduction='none')
        losses['reference_child_link'] = torch.stack([error[labels == value].mean()
            for value in (0, 1) if (labels == value).any()]).mean()
    else:
        losses['reference_child_link'] = hidden.sum() * 0
    weights = dict(line_points=1., arc_shape=1., interval=1., attribute_state=.5,
        attribute_mm=.5, frame_axes=.5, frame_offsets=2., parent_closure=.5,
        reference_child_link=.5, reference_assigned_positive=1., reference_prior_correction=.001)
    objective = sum(value * weights.get(name, 1.) for name, value in losses.items())
    if not torch.isfinite(objective):
        raise ValueError('Nonfinite complete metric training loss')
    optimizer.zero_grad(set_to_none=True)
    (objective * loss_mass).backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(objective.detach()), gradient_norm=float(grad),
                components={name: float(value.detach()) for name, value in losses.items()})


def assess(model, state, inp, sample, supervision):
    runtime = CompleteQueryProposals(model, state['mean'], state['std'], device=next(model.parameters()).device,
                                    fixed_query_fractions=FIXED_QUERY_FRACTIONS)
    outputs = predict_assigned(runtime, inp, sample)
    rows, errors = [], []
    owner = 0
    totals = dict(children=len(outputs), carrier_child_diagnostic_pass=0,
                  free_child_diagnostic_pass=0, either_child_diagnostic_pass=0)
    for parent in supervision.parents:
        for child in parent['children']:
            prediction, point, basis = outputs[owner]
            pid = int(sample['profile_ids'][owner])
            reference = reference_segment(dict(child, world_representation=child['representation']))
            row = dict(original_child_id=child['original_child_id'])
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
                totals[branch + '_child_diagnostic_pass'] += int(row[branch]['geometry_and_type_pass_with_individual_child_reversal'])
            totals['either_child_diagnostic_pass'] += int(any(row[branch]['geometry_and_type_pass_with_individual_child_reversal']
                                                              for branch in ('carrier', 'free')))
            raw_points = point + np.asarray(prediction['line_points']) @ basis.T * 1000
            target_points = np.asarray(reference['points'])[[0, -1]]
            row['ordered_free_point_error_mm'] = float(np.linalg.norm(raw_points - target_points, axis=1).max())
            errors.append(row['ordered_free_point_error_mm'])
            rows.append(row); owner += 1
    totals['free_point_error_mm_p50_p90_max'] = np.percentile(errors, [50, 90, 100]).tolist()
    return dict(totals=totals, children=rows, complete_parent_and_output_quality_not_assessed=True)


def main(args):
    if not 1 <= args.steps <= 128:
        raise ValueError('A bounded positive complete-view comparison budget is required')
    torch.set_num_threads(2)
    base = ROOT / 'server_rebuild' / local_name(args.dataset_name)
    dataset = PartitionedFullShardDataset(ROOT, base)
    if args.version not in dataset.versions:
        raise ValueError('Only a registered TRAIN view may enter this comparison')
    checkpoint = (ROOT / args.checkpoint).resolve(strict=True); checkpoint.relative_to(base)
    if digest(checkpoint) != args.checkpoint_sha256:
        raise ValueError('Frozen model changed')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    complete_fit = verifier(ROOT, base)(ROOT, state, args.checkpoint_sha256, checkpoint.parent / 'receipt.json')
    out = base / local_name(args.experiment_name); out.mkdir(exist_ok=False)
    sources = {name:digest(ROOT / name) for name in SOURCES}
    control = controls(); write(out / 'arithmetic_controls.json', control)
    job, version = dataset.versions[args.version]
    inp = dataset.input_case(job); supervision = dataset.supervision(args.version)
    folder = base / 'full_task_supervision_v1' / job / args.version
    verify(folder, dataset, version)
    with np.load(folder / 'owner_witnesses.npz', allow_pickle=False) as arrays:
        witnesses = {name:arrays[name] for name in arrays.files}
    sample = RuntimeAlignedTrainingAssignments(inp).build(supervision, witnesses, args.seed)
    sample_sha = {name: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
                  for name, value in sample['targets'].items()}
    protocol = dict(sources=sources, checkpoint_sha256=args.checkpoint_sha256,
        original_complete_query_fit=complete_fit, version=args.version, job_key=job,
        complete_children=supervision.target['original_children'], fixed_sample_seed=args.seed,
        target_array_sha256=sample_sha, steps_per_arm=args.steps, learning_rate=.0001,
        original_optimizer_type_and_gradient_clip_retained=True, new_optimizer_each_arm=True,
        original_mean_std_masks_parent_links_and_labels_retained=True, held_reference_reads=0,
        controlled_complete_single_train_view_not_full_training_round=True)
    write(out / 'protocol.json', protocol)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    converter = TrainingTensors(state['mean'], state['std'], device)
    results = {}
    try:
        for arm in ('original', 'metric'):
            torch.manual_seed(1031)
            model, architecture = load_query_model(state, device=device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.0001)
            before = assess(model, state, inp, sample, supervision)
            model.train()
            trace = []
            for step in range(args.steps):
                with threadpool_limits(limits=2):
                    result = (fit_step if arm == 'original' else metric_step)(model, converter, inp,
                        sample, optimizer, version['family_equal_version_weight'] * len(dataset.versions))
                trace.append(result)
                atomic(out / 'progress.json', dict(controller_pid=os.getpid(), arm=arm,
                    actual_optimizer_updates=step+1, required_updates=args.steps,
                    complete_reference_children=supervision.target['original_children'],
                    held_reference_reads=0, unix_time=time.time()))
            after = assess(model, state, inp, sample, supervision)
            weights = out / (arm + '_control_model.pt')
            torch.save(dict(architecture=architecture, state_dict={name:value.detach().cpu() for name,value in model.state_dict().items()},
                optimizer=optimizer.state_dict(), optimizer_steps=args.steps, mean=state['mean'], std=state['std'],
                scope='PARTIAL_SINGLE_TRAIN_VIEW_CONTROL_NOT_COMPLETE_MODEL_NOT_PROMOTED'), weights)
            results[arm] = dict(before=before, after=after, fit_results=trace,
                checkpoint_path=str(weights.relative_to(ROOT)), checkpoint_sha256=digest(weights))
            write(out / (arm + '_result.json'), results[arm])
            print('COMPLETE_VIEW_GEOMETRY_LOSS_COMPARISON', arm, before['totals'], after['totals'], flush=True)
            del model, optimizer; gc.collect()
        if sources != {name:digest(ROOT / name) for name in sources}:
            raise ValueError('Comparison implementation changed')
        atomic(out / 'receipt.json', dict(status='COMPLETE_SINGLE_TRAIN_VIEW_GEOMETRY_LOSS_COMPARISON_NOT_PROMOTED',
            recorded_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=digest(out / 'protocol.json'),
            complete_reference_children=supervision.target['original_children'], optimizer_updates_per_arm=args.steps,
            original_before=results['original']['before']['totals'], original_after=results['original']['after']['totals'],
            metric_before=results['metric']['before']['totals'], metric_after=results['metric']['after']['totals'],
            held_reference_reads=0, completed_optimization_rounds=0, full_contract_pass=False,
            complete_parent_structure_and_held_output_quality_not_evaluated=True))
    except BaseException:
        atomic(out / 'failure.json', dict(traceback=traceback.format_exc(), completed_arms=list(results)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', default='full_dataset_v2')
    parser.add_argument('--checkpoint', required=True); parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--version', required=True); parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--steps', type=int, default=64); parser.add_argument('--seed', type=int, default=1031)
    main(parser.parse_args())
