"""Check changed geometry learning on one actual complete TRAIN workpiece."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from tools.compare_metric_geometry_loss import metric_step
from tools.diagnose_full_query_geometry import atomic
from tools.export_full_dataset_split_view import local_name
from tools.prepare_full_task_supervision import verify
from tools.qualify_full_dataset_labels import ply_points
from tools.record_training_experiment import git_head
from tools.train_full_local_query_geometry_metric import SOURCES as TRAINING_SOURCES
from weld_pipeline.server_cached_local_query_training import CachedLocalQueryInput, CachedLocalTrainingTensors
from weld_pipeline.server_full_weld_model import complete_child_losses
from weld_pipeline.server_local_query_checkpoint import load_local_query_model, verify_complete_local_query_fit
from weld_pipeline.server_memoized_training_queries import MemoizedRawQueryNeighborhood, MemoizedTrainingAssignments
from weld_pipeline.server_metric_geometry_losses import metric_child_losses
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.sjg_local_runtime_904 import digest, write

ROOT = Path(__file__).resolve().parents[1]
SOURCES = list(dict.fromkeys(['tools/check_local_metric_warmstart.py', *TRAINING_SOURCES]))


def main(args):
    torch.set_num_threads(2); torch.manual_seed(1031)
    base = ROOT / 'server_rebuild' / local_name(args.dataset_name)
    dataset = PartitionedFullShardDataset(ROOT, base)
    checkpoint = (ROOT / args.checkpoint).resolve(strict=True); checkpoint.relative_to(base)
    if digest(checkpoint) != args.checkpoint_sha256:
        raise ValueError('The completed warm model changed')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    fit = verify_complete_local_query_fit(ROOT, base, state, args.checkpoint_sha256,
                                          checkpoint.parent / 'receipt.json')
    job_key, version = dataset.versions[args.version]
    view = dataset.supervision(args.version)
    folder = base / 'full_task_supervision_v1' / job_key / args.version
    verify(folder, dataset, version)
    raw = ROOT.parent / 'data/G_Data' / dataset.jobs[job_key]['workpiece_source']
    if digest(raw) != dataset.jobs[job_key]['workpiece_sha256']:
        raise ValueError('Actual raw input changed')
    xyz = ply_points(raw)
    if len(xyz) != dataset.jobs[job_key]['original_points']:
        raise ValueError('Complete raw XYZ is required')
    inp = dataset.input_case(job_key)
    with np.load(folder / 'owner_witnesses.npz', allow_pickle=False) as z:
        witnesses = {k:z[k] for k in z.files}
    out = base / local_name(args.experiment_name); out.mkdir(exist_ok=False)
    sources = {k:digest(ROOT / k) for k in SOURCES}
    with threadpool_limits(limits=2):
        sample = MemoizedTrainingAssignments(inp).build(view, witnesses, 1031)
        cached = CachedLocalQueryInput(inp, sample, MemoizedRawQueryNeighborhood(xyz, inp))
        cached.verify(sample)
        counts = {str(k):int(np.count_nonzero(sample['targets']['kind'] == k)) for k in (0, 1)}
        if not all(counts.values()) or sample['report']['children'] != version['children']:
            raise ValueError('One complete real TRAIN view with both lines and arcs is required')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, architecture = load_local_query_model(state, device)
        converter = CachedLocalTrainingTensors(state['mean'], state['std'], device)
        frozen = {k:p.detach().cpu().clone() for k,p in model.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0001, weight_decay=.0001)
        with torch.no_grad():
            batch = converter.batch(cached, sample)
            prediction = model.predict(model.encode_queries(**batch), converter.tensor(sample['features'][:, 456]))
            targets = {k:converter.tensor(v) for k,v in sample['targets'].items()}
            original = complete_child_losses(prediction, targets)
            metric = metric_child_losses(prediction, targets, batch['geometry'])
            preserved = [k for k in original if k not in ('line_points', 'arc_shape', 'interval')]
            if not all(torch.equal(original[k], metric[k]) for k in preserved):
                raise ValueError('A non-geometry child loss changed')
        model.train(); trace = []; gradients = {}
        for _ in range(2):
            trace.append(metric_step(model, converter, cached, sample, optimizer,
                                     version['family_equal_version_weight'] * len(dataset.versions)))
            for head in ('line_points', 'arc_shape', 'interval_delta'):
                gradients[head] = sum(float(p.grad.detach().abs().sum()) for k,p in model.named_parameters()
                    if k.startswith(head + '.') and p.grad is not None)
            if not all(np.isfinite(v) and v > 0 for v in gradients.values()):
                raise ValueError('A geometry head did not receive finite actual gradients')
        if (not all(torch.equal(frozen[k], p.detach().cpu()) for k,p in model.named_parameters() if k in frozen)
                or not all(torch.isfinite(v).all() for v in model.state_dict().values())):
            raise ValueError('A frozen original parameter changed or a model tensor is nonfinite')
        path = out / 'control_model.pt'
        torch.save(dict(architecture=architecture, state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
            optimizer=optimizer.state_dict(), mean=state['mean'], std=state['std'], optimizer_steps=2,
            scope='SINGLE_COMPLETE_TRAIN_VIEW_CONTROL_NOT_FULL_MODEL_NOT_PROMOTED'), path)
    if sources != {k:digest(ROOT / k) for k in sources} or digest(checkpoint) != args.checkpoint_sha256:
        raise ValueError('A controlled source or frozen model changed')
    write(out / 'protocol.json', dict(sources=sources, source_commit=git_head(ROOT),
        warm_checkpoint_sha256=args.checkpoint_sha256, training_registry_sha256=dataset.registry_sha256,
        actual_version=args.version, actual_job_key=job_key, actual_input_bindings=cached.bindings,
        held_reference_reads=0, model_quality_not_evaluated=True))
    atomic(out / 'receipt.json', dict(status='PASS_ACTUAL_WARM_COMPLETE_VIEW_METRIC_LOSS_AND_GEOMETRY_GRADIENTS',
        recorded_utc=datetime.now(timezone.utc).isoformat(), warm_checkpoint_sha256=args.checkpoint_sha256,
        original_complete_fit=fit, actual_version=args.version, complete_children=version['children'],
        complete_raw_xyz_points=len(xyz), primitive_counts=counts, actual_control_optimizer_updates=2,
        actual_geometry_head_gradient_sums=gradients, all_non_geometry_child_losses_bitwise_preserved=preserved,
        all_frozen_original_parameters_bitwise_preserved=True, fit_results=trace,
        control_checkpoint_path=str(path.relative_to(ROOT)), control_checkpoint_sha256=digest(path),
        held_reference_reads=0, production_optimizer_updates=0, model_quality_not_evaluated=True,
        completed_optimization_rounds=0, full_contract_pass=False))
    print('ACTUAL_COMPLETE_METRIC_WARMSTART_PASSED', args.version, version['children'], gradients, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', default='full_dataset_v2')
    parser.add_argument('--checkpoint', required=True); parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--experiment-name', required=True)
    main(parser.parse_args())
