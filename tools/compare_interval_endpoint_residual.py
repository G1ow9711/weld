"""Compare endpoint conditioning on one actual complete qualified TRAIN view.

Both arms use the same completed warm model, raw XYZ, deployed input queries,
all supervision and metric objective. This cannot promote a full dataset model.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import os
import time
import traceback
from types import FunctionType

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from tools.compare_metric_geometry_loss import assess, metric_step
from tools.diagnose_full_query_geometry import atomic, predict_assigned
from tools.export_full_dataset_split_view import local_name
from tools.prepare_full_task_supervision import verify
from tools.qualify_full_dataset_labels import ply_points
from tools.record_training_experiment import git_head
from tools.train_full_local_query_geometry_metric import SOURCES as TRAINING_SOURCES
from weld_pipeline.server_cached_local_query_training import CachedLocalQueryInput, CachedLocalTrainingTensors
from weld_pipeline.server_deployed_query_assignments import eligible_query_positions
from weld_pipeline.server_interval_endpoint_residual_model import (
    IntervalEndpointResidualModel, InputEndpointProjection, ARCHITECTURE)
from weld_pipeline.server_local_query_checkpoint import load_local_query_model, verify_complete_local_query_fit
from weld_pipeline.server_local_query_runtime import LocalCompleteQueryProposals
from weld_pipeline.server_metric_geometry_losses import metric_child_losses
from weld_pipeline.server_memoized_training_queries import MemoizedRawQueryNeighborhood, MemoizedTrainingAssignments
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.sjg_local_runtime_904 import digest, read, write


ROOT = Path(__file__).resolve().parents[1]
SOURCES = list(dict.fromkeys(['tools/compare_interval_endpoint_residual.py',
    'weld_pipeline/server_interval_endpoint_residual_model.py', *TRAINING_SOURCES]))


def same_bytes(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                            right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))


def projection_control():
    value = torch.tensor([[-2., 3.], [.9, -.7]], requires_grad=True)
    ratio = torch.tensor([.25, .75])
    output = InputEndpointProjection.apply(value, ratio)
    if not torch.equal(output, torch.tensor([[0., 1.], [.75, .75]])):
        raise ValueError('The projected output left the actual input query bounds')
    output.sum().backward()
    if not torch.equal(value.grad, torch.ones_like(value)):
        raise ValueError('Overshoot did not retain a correction gradient')
    return dict(forward_bounds_checked=True, overshoot_identity_backward_checked=True)


def main(args):
    if not 1 <= args.steps <= 128:
        raise ValueError('A bounded positive complete-view comparison is required')
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    base = ROOT / 'server_rebuild' / local_name(args.dataset_name)
    dataset = PartitionedFullShardDataset(ROOT, base)
    if args.version not in dataset.versions:
        raise ValueError('Only a registered qualified TRAIN view may fit')
    checkpoint = (ROOT / args.checkpoint).resolve(strict=True); checkpoint.relative_to(base)
    if digest(checkpoint) != args.checkpoint_sha256:
        raise ValueError('Completed warm checkpoint changed')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    fit = verify_complete_local_query_fit(ROOT, base, state, args.checkpoint_sha256,
                                          checkpoint.parent / 'receipt.json')
    out = base / local_name(args.experiment_name); out.mkdir(exist_ok=False)
    sources = {name:digest(ROOT / name) for name in SOURCES}
    try:
        control = projection_control()
        job_key, version = dataset.versions[args.version]; job = dataset.jobs[job_key]
        raw = ROOT.parent / 'data/G_Data' / job['workpiece_source']
        if digest(raw) != job['workpiece_sha256']:
            raise ValueError('Original raw input changed')
        xyz = ply_points(raw)
        if len(xyz) != job['original_points']:
            raise ValueError('Complete raw XYZ is required')
        inp = dataset.input_case(job_key); view = dataset.supervision(args.version)
        supervision = base / 'full_task_supervision_v1' / job_key / args.version
        verify(supervision, dataset, version)
        with np.load(supervision / 'owner_witnesses.npz', allow_pickle=False) as arrays:
            witnesses = {name:arrays[name] for name in arrays.files}
        with threadpool_limits(limits=2):
            sample = MemoizedTrainingAssignments(inp).build(view, witnesses, args.seed)
            cached = CachedLocalQueryInput(inp, sample, MemoizedRawQueryNeighborhood(xyz, inp))
        cached.verify(sample)
        if sample['report']['children'] != version['children']:
            raise ValueError('The complete real TRAIN view lost a child')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        converter = CachedLocalTrainingTensors(state['mean'], state['std'], device)
        original, _ = load_local_query_model(state, device)
        residual = IntervalEndpointResidualModel().to(device)
        residual.inherit_complete_local(state['state_dict']); residual.eval()
        with torch.no_grad():
            batch = converter.batch(cached, sample); ratio = converter.tensor(sample['features'][:, 456])
            old_hidden = original.encode_queries(**batch); new_hidden = residual.encode_queries(**batch)
            before = original.predict(old_hidden, ratio); after = residual.predict(new_hidden, ratio)
            if set(before) != set(after) or not all(same_bytes(before[k], after[k]) for k in before):
                raise ValueError('Zero endpoint correction changed an actual named warm output')
            targets = {name:converter.tensor(value) for name,value in sample['targets'].items()}
            old_losses = metric_child_losses(before, targets, batch['geometry'])
            new_losses = metric_child_losses(after, targets, batch['geometry'])
            if not all(same_bytes(old_losses[k], new_losses[k]) for k in old_losses):
                raise ValueError('Initial geometry, known fields or frame losses changed')
        scoring = FunctionType(predict_assigned.__code__, dict(predict_assigned.__globals__,
            fixed_query_positions=eligible_query_positions), predict_assigned.__name__)
        factory = lambda m,mean,std,**kwargs:LocalCompleteQueryProposals(m,mean,std,xyz,**kwargs)
        evaluation = FunctionType(assess.__code__, dict(assess.__globals__,
            CompleteQueryProposals=factory, predict_assigned=scoring), assess.__name__)
        protocol = dict(sources=sources, source_commit=git_head(ROOT), complete_warm_fit=fit,
            warm_checkpoint_sha256=args.checkpoint_sha256, training_registry_sha256=dataset.registry_sha256,
            version=args.version, job_key=job_key, complete_children=version['children'], raw_points=len(xyz),
            raw_sha256=job['workpiece_sha256'], label_receipt_sha256=version['label_receipt_sha256'],
            assignment_seed=args.seed, actual_input_bindings=cached.bindings, steps_per_arm=args.steps,
            learning_rate=.0001, new_architecture=ARCHITECTURE, endpoint_offset_unit='metres',
            original_metric_objective_all_masks_and_parent_link_supervision_retained=True,
            all_initial_named_outputs_and_child_losses_bitwise_equal=True, projection_control=control,
            held_reference_reads=0, full_model_or_complete_optimization_round_not_claimed=True)
        write(out / 'protocol.json', protocol)
        results = {}
        for label,model in [('original',original), ('endpoint_residual',residual)]:
            model.eval(); before = evaluation(model, state, inp, sample, view)
            frozen = {n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                          lr=.0001, weight_decay=.0001)
            trace = []; gradients = []; model.train()
            for step in range(args.steps):
                with threadpool_limits(limits=2):
                    trace.append(metric_step(model, converter, cached, sample, optimizer,
                        version['family_equal_version_weight'] * len(dataset.versions)))
                if label == 'endpoint_residual':
                    value = sum(float(p.grad.detach().abs().sum()) for n,p in model.named_parameters()
                        if n.startswith('interval_endpoint_offset.') and p.grad is not None)
                    if not np.isfinite(value) or (step == 0 and value <= 0):
                        raise ValueError('The new endpoint head lacks actual finite initial gradients')
                    gradients.append(value)
                atomic(out / 'progress.json', dict(controller_pid=os.getpid(), arm=label,
                    actual_updates=step+1, required_updates=args.steps, complete_children=version['children'],
                    held_reference_reads=0, unix_time=time.time()))
            model.eval(); after = evaluation(model, state, inp, sample, view)
            if (not all(torch.equal(frozen[n], p.detach().cpu()) for n,p in model.named_parameters() if n in frozen)
                    or not all(torch.isfinite(v).all() for v in model.state_dict().values())):
                raise ValueError('A frozen warm parameter changed or the model is nonfinite')
            weights = out / (label + '_control_model.pt')
            architecture = ARCHITECTURE if label == 'endpoint_residual' else state['architecture']
            torch.save(dict(architecture=architecture, state_dict={n:v.detach().cpu() for n,v in model.state_dict().items()},
                optimizer=optimizer.state_dict(), optimizer_steps=args.steps, mean=state['mean'], std=state['std'],
                scope='PARTIAL_COMPLETE_SINGLE_TRAIN_VIEW_CONTROL_NOT_COMPLETE_MODEL'), weights)
            results[label] = dict(before=before, after=after, trace=trace,
                endpoint_head_gradient_sums=gradients, all_frozen_original_parameters_bitwise_preserved=True,
                checkpoint_path=str(weights.relative_to(ROOT)), checkpoint_sha256=digest(weights))
            write(out / (label + '_result.json'), results[label])
        partial = torch.load(out / 'endpoint_residual_control_model.pt', map_location='cpu', weights_only=False)
        try:
            load_local_query_model(partial)
        except ValueError:
            pass
        else:
            raise ValueError('An experimental partial architecture entered the old complete model loader')
        if sources != {n:digest(ROOT/n) for n in sources} or digest(checkpoint) != args.checkpoint_sha256:
            raise ValueError('A controlled source or completed warm model changed')
        atomic(out / 'receipt.json', dict(status='ACTUAL_COMPLETE_TRAIN_VIEW_ENDPOINT_RESIDUAL_COMPARISON_NOT_PROMOTED',
            recorded_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=digest(out / 'protocol.json'),
            complete_children=version['children'], actual_raw_points=len(xyz), actual_updates_per_arm=args.steps,
            all_initial_named_outputs_and_child_losses_bitwise_equal=True,
            all_frozen_original_parameters_bitwise_preserved=True, old_model_loader_rejected_partial_architecture=True,
            before={k:v['before']['totals'] for k,v in results.items()},
            after={k:v['after']['totals'] for k,v in results.items()},
            endpoint_head_initial_gradient_sum=results['endpoint_residual']['endpoint_head_gradient_sums'][0],
            held_reference_reads=0, production_optimizer_updates=0, completed_optimization_rounds=0,
            full_contract_pass=False, complete_parent_and_full_dataset_evaluation_remaining=True))
        print('ENDPOINT_RESIDUAL_ACTUAL_COMPARISON_COMPLETE', {k:v['after']['totals'] for k,v in results.items()}, flush=True)
    except BaseException:
        atomic(out / 'failure.json', dict(traceback=traceback.format_exc(), model_not_promoted=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', default='full_dataset_v2')
    parser.add_argument('--checkpoint', required=True); parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--version', required=True); parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--steps', type=int, default=64); parser.add_argument('--seed', type=int, default=1031)
    main(parser.parse_args())
