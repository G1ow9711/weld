"""Train the expanded local XYZ query model with every complete TRAIN version.

Original weights/normalization initialize this independent candidate. All
geometry, attributes, signed frames and parent/link targets remain supervised;
unknown masks are preserved. This does not change the running original model.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import gc
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from tools.diagnose_full_query_geometry import atomic
from tools.export_full_dataset_split_view import local_name
from tools.prepare_full_task_supervision import verify
from tools.qualify_full_dataset_labels import ply_points
from tools.record_training_experiment import git_head
from tools.train_full_weld_reference_recall import TrainingTensors, fit_step
from weld_pipeline.server_deployed_query_assignments import DeployedGridTrainingAssignments
from weld_pipeline.server_local_query_checkpoint import COMPLETE_STATUS, verify_complete_local_query_fit
from weld_pipeline.server_local_query_weld_model import LocalQueryWeldModel, ARCHITECTURE
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.server_query_point_neighborhood import RawQueryNeighborhood
from weld_pipeline.sjg_local_runtime_904 import digest, read, write


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['tools/train_full_local_query_geometry.py',
    'weld_pipeline/server_local_query_checkpoint.py', 'weld_pipeline/server_local_query_weld_model.py',
    'weld_pipeline/server_local_query_runtime.py', 'weld_pipeline/server_query_point_neighborhood.py',
    'weld_pipeline/server_deployed_query_assignments.py', 'weld_pipeline/server_full_training_samples.py',
    'weld_pipeline/server_finite_carrier_grid.py', 'tools/qualify_full_dataset_labels.py',
    'tools/prepare_full_task_supervision.py',
    'weld_pipeline/server_partitioned_full_dataset.py', 'weld_pipeline/server_full_shard_dataset.py',
    'weld_pipeline/server_full_weld_model.py', 'weld_pipeline/server_reference_recall_weld_model.py',
    'weld_pipeline/server_legacy_anchored_full_weld_model.py', 'tools/train_full_weld_reference_recall.py',
    'weld_pipeline/sjg_global_neighborhood_sequence_1239.py']


class LiveLocalTrainingTensors(TrainingTensors):
    def __init__(self, mean, std, device, neighborhoods):
        super().__init__(mean, std, device)
        self.neighborhoods = neighborhoods
        self.last_context_report = None

    def batch(self, inp, sample):
        local = np.empty((len(sample['profile_ids']), 72), np.float32)
        for pid in np.unique(sample['profile_ids']):
            owners = np.flatnonzero(sample['profile_ids'] == pid)
            local[owners] = self.neighborhoods.features(inp, int(pid), sample['input_query_positions_mm'][owners])
        self.last_context_report = dict(input_neighborhood_rows=len(local), local_feature_columns=72,
            queries_belong_to_deployed_input_grid=True, raw_point_tree_size=len(self.neighborhoods.xyz),
            references_or_source_ids_as_features=False)
        return dict(super().batch(inp, sample), query_local_features=self.tensor(local))


def save(path, model, optimizer, state):
    pending = path.with_name(path.name + '.incomplete')
    payload = dict(state, state_dict={name:value.detach().cpu() for name,value in model.state_dict().items()},
        optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state())
    torch.save(payload, pending)
    check = torch.load(pending, map_location='cpu', weights_only=False)
    if (check['optimizer_steps'] != state['optimizer_steps']
            or check['completed_views'] != state['completed_views'] or not check['optimizer']['state']):
        raise ValueError('Actual full local-query optimizer and coverage state did not reload')
    os.replace(pending, path)


def main(args):
    if min(args.epochs, args.steps_per_view) < 1:
        raise ValueError('Complete positive training budgets are required')
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    base = ROOT / 'server_rebuild' / local_name(args.dataset_name)
    dataset = PartitionedFullShardDataset(ROOT, base)
    dataset.require_complete_inputs()
    admission = base / 'local_query_xyz_initialization_control_20261012_v1'
    proof = read(admission / 'receipt.json')
    initialization = ROOT / proof['local_initialization_path']
    if (proof['status'] != 'PASS_ACTUAL_RAW_XYZ_LOCAL_QUERY_ZERO_RESIDUAL_AND_GRADIENT_CONTROLS'
            or not proof['all_actual_named_query_outputs_initially_bitwise_equal']
            or not proof['all_original_weights_unchanged_after_local_only_updates']
            or proof['sources'] != {name:digest(ROOT / name) for name in proof['sources']}
            or digest(initialization) != proof['local_initialization_sha256']):
        raise ValueError('Actual raw-point zero-residual and gradient admission controls are required')
    initial = torch.load(initialization, map_location='cpu', weights_only=False)
    if initial['architecture'] != ARCHITECTURE or initial['optimizer_steps'] != 0:
        raise ValueError('A preserved zero-residual initialization is required')
    out = base / local_name(args.experiment_name); out.mkdir(exist_ok=True)
    with (out / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = {name:digest(ROOT / name) for name in SOURCES}
        settings = dict(stage='COMPLETE_TRAIN_LOCAL_RAW_XYZ_DENSE_QUERY_GEOMETRY', architecture=ARCHITECTURE,
            registry_sha256=dataset.registry_sha256, sources=sources, epochs=args.epochs,
            steps_per_view=args.steps_per_view, registered_versions=len(dataset.versions),
            registered_families=len(dataset.families), complete_raw_xyz_required=True,
            query_local_feature_columns=72, query_assignment='EXISTING_DEPLOYED_DENSE_INPUT_GRID',
            local_initialization_sha256=digest(initialization), complete_original_initialization_preserved=True,
            original_mean_std_frozen=True, original_complete_geometry_field_frame_and_link_loss_retained=True,
            original_loss_weights_retained=True, metric_only_loss_change_not_adopted=True,
            learning_rate=.0001, gradient_clip=10., seed=args.seed,
            ordering='Actually shuffled input jobs each epoch; every separate complete version retained',
            same_input_versions='Separate expected losses with original family/version weights; no label merging',
            unassigned_queries_not_physical_background=True, original_run446_unchanged=True,
            validation_or_test_label_reads=0, diagnostic64_fit=False, model_not_promoted=True)
        protocol = out / 'protocol.json'
        if protocol.exists():
            if read(protocol)['settings'] != settings:
                raise ValueError('Independent complete local training recipe changed')
        else:
            write(protocol, dict(settings=settings, source_commit=git_head(ROOT),
                                 recorded_utc=datetime.now(timezone.utc).isoformat()))
        warm = out / 'warm_initialization.pt'
        if not warm.exists():
            shutil.copyfile(initialization, warm)
        if digest(warm) != proof['local_initialization_sha256']:
            raise ValueError('Preserved local initialization changed')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = LocalQueryWeldModel().to(device); model.load_state_dict(initial['state_dict'], strict=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0001, weight_decay=.0001)
        state = dict(architecture=ARCHITECTURE, protocol_sha256=digest(protocol), sources=sources,
            optimizer_steps=0, completed_views=[], view_evidence={}, mean=initial['mean'], std=initial['std'],
            inherited_optimizer_steps=initial['inherited_optimizer_steps'])
        last = out / 'last.pt'
        if last.exists():
            saved = torch.load(last, map_location=device, weights_only=False)
            if saved['protocol_sha256'] != state['protocol_sha256'] or saved['sources'] != sources:
                raise ValueError('Complete local optimizer recovery belongs to another recipe')
            model.load_state_dict(saved.pop('state_dict')); optimizer.load_state_dict(saved.pop('optimizer'))
            torch.set_rng_state(saved.pop('torch_rng').cpu()); state = saved
        completed = set(state['completed_views']); model.train()
        try:
            for epoch in range(args.epochs):
                keys = list(dataset.jobs)
                order = np.random.default_rng(args.seed + epoch).permutation(len(keys))
                for order_index in order:
                    job_key = keys[int(order_index)]; job = dataset.jobs[job_key]
                    remaining = [v for v in job['versions'] if f'{epoch}:{v["key"]}' not in completed]
                    if not remaining:
                        continue
                    if sources != {name:digest(ROOT / name) for name in sources}:
                        raise ValueError('Live local training source changed')
                    raw = ROOT.parent / 'data/G_Data' / job['workpiece_source']
                    if digest(raw) != job['workpiece_sha256']:
                        raise ValueError('Complete raw XYZ source changed before use')
                    xyz = ply_points(raw)
                    if len(xyz) != job['original_points']:
                        raise ValueError('Full raw point count changed')
                    inp = dataset.input_case(job_key); neighborhoods = RawQueryNeighborhood(xyz)
                    builder = DeployedGridTrainingAssignments(inp)
                    converter = LiveLocalTrainingTensors(initial['mean'], initial['std'], device, neighborhoods)
                    for version in remaining:
                        key = version['key']; view = dataset.supervision(key)
                        folder = base / 'full_task_supervision_v1' / job_key / key
                        verify(folder, dataset, version)
                        with np.load(folder / 'owner_witnesses.npz', allow_pickle=False) as arrays:
                            witnesses = {name:arrays[name] for name in arrays.files}
                        results, reports, contexts = [], [], []
                        for step in range(args.steps_per_view):
                            sample = builder.build(view, witnesses, args.seed + epoch * 100000 + step)
                            with threadpool_limits(limits=2):
                                result = fit_step(model, converter, inp, sample, optimizer,
                                    version['family_equal_version_weight'] * len(dataset.versions))
                            state['optimizer_steps'] += 1
                            results.append(result); reports.append(sample['report']); contexts.append(converter.last_context_report)
                            atomic(out / 'progress.json', dict(status='ACTUAL_COMPLETE_TRAIN_LOCAL_XYZ_QUERY_UPDATES',
                                controller_pid=os.getpid(), epoch=epoch, epochs=args.epochs, version_key=key,
                                optimizer_steps=state['optimizer_steps'], local_step=step+1,
                                steps_per_view=args.steps_per_view, completed_views=len(completed),
                                registered_versions=len(dataset.versions), reference_children=view.target['original_children'],
                                complete_raw_points=len(xyz), held_reference_reads=0, unix_time=time.time()))
                        identity = f'{epoch}:{key}'; completed.add(identity)
                        state['completed_views'] = sorted(completed)
                        state['view_evidence'][identity] = dict(job_key=job_key,
                            input_receipt_sha256=digest(dataset.shards / job_key / 'receipt.json'),
                            label_receipt_sha256=version['label_receipt_sha256'],
                            owner_witness_receipt_sha256=digest(folder / 'receipt.json'),
                            reference_children=view.target['original_children'],
                            family_equal_version_weight=version['family_equal_version_weight'],
                            raw_xyz_sha256=job['workpiece_sha256'], complete_raw_xyz_points=len(xyz),
                            reports=reports, fit_results=results, local_context_reports=contexts)
                        save(last, model, optimizer, state)
                        atomic(out / 'checkpoint_receipt.json', dict(status='PARTIAL_COMPLETE_LOCAL_QUERY_COVERAGE_NOT_PROMOTED',
                            checkpoint_sha256=digest(last), completed_views=len(completed),
                            optimizer_steps=state['optimizer_steps'], protocol_sha256=digest(protocol),
                            validation_or_test_label_reads=0))
                        print('FULL_LOCAL_XYZ_TRAIN_VIEW', epoch, key, len(completed), '/', args.epochs * len(dataset.versions),
                              'updates', state['optimizer_steps'], 'children', view.target['original_children'], flush=True)
                        del sample, view, witnesses
                    del inp, neighborhoods, builder, converter, xyz
                    gc.collect()
                snapshot = out / f'epoch_{epoch:03d}.pt'
                if not snapshot.exists():
                    save(snapshot, model, optimizer, state)
            if len(completed) != args.epochs * len(dataset.versions):
                raise ValueError('A registered complete TRAIN view was omitted')
            atomic(out / 'receipt.json', dict(status=COMPLETE_STATUS, architecture=ARCHITECTURE,
                checkpoint_sha256=digest(last), protocol_sha256=digest(protocol), epochs=args.epochs,
                optimizer_steps=state['optimizer_steps'], training_versions=len(dataset.versions),
                training_families=len(dataset.families), completed_views=len(completed),
                complete_reference_children=sum(v['children'] for _,v in dataset.versions.values()),
                validation_or_test_label_reads=0, model_output_accuracy_evaluations=0,
                complete_selection_and_validation_required=True))
            loaded = torch.load(last, map_location='cpu', weights_only=False)
            verified = verify_complete_local_query_fit(ROOT, base, loaded, digest(last), out / 'receipt.json')
            atomic(out / 'complete_fit_verification.json', verified)
            print('ALL_COMPLETE_LOCAL_XYZ_QUERY_TRAINING_FINISHED', state['optimizer_steps'], flush=True)
        except BaseException:
            atomic(out / 'failure.json', dict(traceback=traceback.format_exc(), completed_views=len(completed),
                optimizer_steps=state['optimizer_steps'], model_not_promoted=True))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', default='full_dataset_v2')
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--steps-per-view', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1031)
    main(parser.parse_args())
