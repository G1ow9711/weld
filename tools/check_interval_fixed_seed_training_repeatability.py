"""Repeat a short actual complete TRAIN view in two independent CUDA processes.

The unchanged full metric objective includes geometry, process masks, frames
and parent/link supervision. This is a deterministic training control, not a
full-data optimization round or an accuracy evaluation.
"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from tools.compare_metric_geometry_loss import metric_step
from tools.export_full_dataset_split_view import local_name
from tools.prepare_full_task_supervision import verify
from tools.qualify_full_dataset_labels import ply_points
from tools.record_training_experiment import git_head
from tools.train_full_interval_endpoint_geometry import SOURCES as TRAINING_SOURCES
from tools.verify_original_best_frozen_replay import canonical
from weld_pipeline.server_cached_local_query_training import CachedLocalQueryInput, CachedLocalTrainingTensors, array_binding
from weld_pipeline.server_interval_endpoint_checkpoint import load_interval_endpoint_model, verify_complete_interval_endpoint_fit
from weld_pipeline.server_memoized_training_queries import MemoizedRawQueryNeighborhood, MemoizedTrainingAssignments
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.sjg_local_runtime_904 import ROOT, digest, read, write


BASE = ROOT / 'server_rebuild/full_dataset_v2'
CHECKPOINT = BASE / 'full_interval_endpoint_geometry_v5_round04/last.pt'
CHECKPOINT_SHA = '1f1488106cc00501c39522a251b069fdc3ccf865de0d3e382d7f0f10649cc0aa'
SOURCES = list(dict.fromkeys(['tools/check_interval_fixed_seed_training_repeatability.py',
    'tools/compare_metric_geometry_loss.py', 'tools/verify_original_best_frozen_replay.py', *TRAINING_SOURCES]))


def tensor_binding(value):
    value = value.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return dict(dtype=str(value.dtype), shape=list(value.shape), sha256=hashlib.sha256(raw).hexdigest())


def nested_binding(value):
    if torch.is_tensor(value):
        return tensor_binding(value)
    if isinstance(value, dict):
        return {str(key): nested_binding(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [nested_binding(v) for v in value]
    return value


def worker(out, attempt):
    torch.set_num_threads(2); torch.manual_seed(1031); np.random.seed(1031)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise ValueError('The actual training device is required for this CUDA control')
    protocol = read(out / 'protocol.json')
    dataset = PartitionedFullShardDataset(ROOT, BASE)
    key = protocol['version']; job_key, version = dataset.versions[key]
    job = dataset.jobs[job_key]
    raw = ROOT.parent / 'data/G_Data' / job['workpiece_source']
    if digest(raw) != job['workpiece_sha256'] or digest(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('Bound raw input or completed warm model changed')
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    fit = verify_complete_interval_endpoint_fit(ROOT, BASE, state, CHECKPOINT_SHA, CHECKPOINT.parent / 'receipt.json')
    xyz = ply_points(raw); inp = dataset.input_case(job_key); view = dataset.supervision(key)
    if len(xyz) != job['original_points']:
        raise ValueError('Complete raw input was lost')
    supervision = BASE / 'full_task_supervision_v1' / job_key / key
    verify(supervision, dataset, version)
    with np.load(supervision / 'owner_witnesses.npz', allow_pickle=False) as arrays:
        witnesses = {name: arrays[name] for name in arrays.files}
    with threadpool_limits(limits=2):
        sample = MemoizedTrainingAssignments(inp).build(view, witnesses, 1031)
        cached = CachedLocalQueryInput(inp, sample, MemoizedRawQueryNeighborhood(xyz, inp))
    cached.verify(sample)
    if sample['report']['children'] != version['children']:
        raise ValueError('The short repeat did not include every child in its TRAIN view')
    converter = CachedLocalTrainingTensors(state['mean'], state['std'], 'cuda')
    model, _ = load_interval_endpoint_model(state, 'cuda')
    frozen = {n: tensor_binding(p) for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=.0001, weight_decay=.0001, eps=1e-6)
    folder = out / f'process_{attempt}'; folder.mkdir(exist_ok=False)
    trace = []
    for step in range(protocol['updates']):
        model.train()
        result = metric_step(model=model, converter=converter, inp=cached,
                             sample=sample, optimizer=optimizer, loss_mass=1.)
        trace.append(result)
        write(folder / 'progress.json', dict(step=step + 1, updates=protocol['updates'], children=version['children']))
    model.eval()
    with torch.no_grad():
        batch = converter.batch(cached, sample)
        prediction = model.predict(model.encode_queries(**batch), converter.tensor(sample['features'][:, 456]))
    if any(frozen[name] != tensor_binding(p) for name, p in model.named_parameters() if name in frozen):
        raise ValueError('A frozen original parameter changed in the repeat')
    if protocol['sources'] != {name: digest(ROOT / name) for name in protocol['sources']}:
        raise ValueError('Repeat training source changed')
    weights = {name: tensor_binding(value) for name, value in model.state_dict().items()}
    outputs = {name: tensor_binding(value) for name, value in prediction.items()}
    result = dict(pid=os.getpid(), version=key, job_key=job_key, children=version['children'],
        raw_points=len(xyz), actual_input_bindings=cached.bindings,
        targets={name: array_binding(value) for name, value in sample['targets'].items()},
        complete_warm_fit=fit, seed=1031, updates=protocol['updates'], trace=trace,
        all_state_tensors=weights, all_named_predictions=outputs,
        optimizer_state=nested_binding(optimizer.state_dict()), all_original_frozen_parameters_unchanged=True,
        full_known_geometry_process_masks_frames_parent_and_links_retained=True,
        device='cuda', deterministic_algorithms=True, held_or_test_reads=0,
        complete_data_training_or_quality_claimed=False)
    write(folder / 'receipt.json', result)


def main(args):
    out = BASE / local_name(args.experiment_name)
    if args.worker is not None:
        worker(out, args.worker); return
    out.mkdir(exist_ok=False)
    dataset = PartitionedFullShardDataset(ROOT, BASE)
    if args.version not in dataset.versions or not 2 <= args.updates <= 8:
        raise ValueError('A registered TRAIN view and bounded short repeat are required')
    sources = {name: digest(ROOT / name) for name in SOURCES}
    write(out / 'protocol.json', dict(source_commit=git_head(ROOT), sources=sources,
        registry_sha256=dataset.registry_sha256, checkpoint_sha256=CHECKPOINT_SHA,
        version=args.version, updates=args.updates, seed=1031, independent_processes=2,
        single_complete_view_loss_mass=1.,
        actual_training_device='cuda', objective='Unchanged complete metric objective with every known mask and parent/link term',
        scope='One actual complete registered TRAIN view; deterministic repeat only',
        complete_optimization_rounds=0, held_or_test_reads=0))
    try:
        results = []
        for attempt in range(2):
            write(out / 'progress.json', dict(stage='SHORT_ACTUAL_COMPLETE_VIEW_REPEAT', attempt=attempt,
                                             controller_pid=os.getpid(), version=args.version))
            env = dict(os.environ, PYTHONHASHSEED='1031', CUBLAS_WORKSPACE_CONFIG=':4096:8')
            subprocess.run([sys.executable, '-m', 'tools.check_interval_fixed_seed_training_repeatability',
                '--experiment-name', args.experiment_name, '--worker', str(attempt)], cwd=ROOT, env=env, check=True)
            results.append(read(out / f'process_{attempt}/receipt.json'))
        a, b = [dict(x) for x in results]
        a.pop('pid'); b.pop('pid')
        if results[0]['pid'] == results[1]['pid'] or canonical(a) != canonical(b):
            changed = [key for key in a if a[key] != b[key]]
            raise ValueError('Fixed seed CUDA training repeat changed: ' + ', '.join(changed))
        write(out / 'receipt.json', dict(status='PASS_FIXED_SEED_ACTUAL_COMPLETE_TRAIN_VIEW_CUDA_REPEAT',
            protocol_sha256=digest(out / 'protocol.json'), independent_processes=[r['pid'] for r in results],
            children=results[0]['children'], raw_points=results[0]['raw_points'], actual_updates_per_process=args.updates,
            complete_targets_inputs_each_step_losses_model_optimizer_and_named_outputs_identical=True,
            worker_receipts=[dict(path=f'process_{i}/receipt.json', sha256=digest(out / f'process_{i}/receipt.json'))
                             for i in range(2)], full_171_fit_claimed=False, accuracy_claimed=False,
            complete_optimization_rounds=0, held_or_test_reads=0, full_contract_pass=False))
        print('PASS_SHORT_FIXED_SEED_COMPLETE_TRAIN_VIEW_CUDA_REPEAT', flush=True)
    except BaseException as exc:
        write(out / 'failure.json', dict(error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--version', default='new_ceeae49d6777')
    parser.add_argument('--updates', type=int, default=6)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    main(parser.parse_args())
