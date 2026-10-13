"""Three complete TRAIN passes of finite interval/field/directed-link learning.

A resumable optimization stage, not a complete extraction/acceptance report.
Epoch checkpoints await complete output and ordered held-evaluation gates.
"""
import argparse
import fcntl
import gc
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from tools.export_full_dataset_split_view import local_name
from tools.prepare_full_task_supervision import verify
from tools.probe_multiview_visual_assistance import load_xyz
from tools.train_full_weld_reference_recall import TrainingTensors
from weld_pipeline.server_interval_endpoint_checkpoint import verify_complete_interval_endpoint_fit
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.server_predicted_parent_pool import FrozenRun446Priors
from weld_pipeline.server_supported_interval_events import FiniteInputIntervalEvents
from weld_pipeline.server_supported_interval_training import CompleteIntervalView, complete_view_step
from weld_pipeline.server_supported_interval_weld_model import SupportedIntervalWeldModel, ARCHITECTURE
from weld_pipeline.server_visual_augmented_input import VisualAugmentedInput
from weld_pipeline.server_visual_augmented_training_samples import VisualAugmentedTrainingAssignments
from weld_pipeline.server_visual_curve_context import VisualCurveContext
from weld_pipeline.sjg_local_runtime_904 import ROOT, digest, read, write

BASE = ROOT / 'server_rebuild/full_dataset_v2'
VISUAL = BASE / 'full_171_visual_assistance_and_candidates_20261013_v1'
SUPERVISION = BASE / 'full_171_visual_augmented_supervision_20261013_v1'
NATIVE_EVENTS = BASE / 'full_171_supported_interval_training_events_20261013_v1/banks'
WARM = BASE / 'full_interval_endpoint_geometry_v5_round04/last.pt'
WARM_SHA = '1f1488106cc00501c39522a251b069fdc3ccf865de0d3e382d7f0f10649cc0aa'
SOURCES = ['tools/train_supported_interval_full_dataset.py',
    'weld_pipeline/server_supported_interval_training.py',
    'weld_pipeline/server_supported_interval_weld_model.py',
    'weld_pipeline/server_visual_augmented_training_samples.py',
    'weld_pipeline/server_visual_augmented_input.py', 'weld_pipeline/server_visual_curve_context.py',
    'weld_pipeline/server_supported_interval_events.py', 'weld_pipeline/server_stable_endpoint_pairs.py',
    'weld_pipeline/server_stable_joint_interval_residual.py', 'weld_pipeline/server_supported_directed_parent_pool.py',
    'weld_pipeline/server_tiled_multiview_visual_features.py']


class TrainingJob:
    def __init__(self, dataset, key, visual_records):
        self.dataset, self.key = dataset, key
        item = visual_records[key]; path = VISUAL / item['path']
        if digest(path) != item['sha256']: raise ValueError('Frozen visual candidates changed')
        addition = read(path)
        self.input_sha = digest(dataset.shards / key / 'receipt.json')
        if (addition['original_input_receipt_sha256'] != self.input_sha
                or addition['original_profiles_sha256'] != digest(dataset.shards / key / 'profiles.json')):
            raise ValueError('Visual and original input identity changed')
        self.inp = VisualAugmentedInput(dataset.input_case(key), addition)
        self.xyz, _ = load_xyz(dataset.jobs[key]); self.context = VisualCurveContext(self.xyz)
        self.producer = FiniteInputIntervalEvents(self.xyz, self.input_sha)
        self.assigner = VisualAugmentedTrainingAssignments(self.inp)
        self.bank_paths = {}
        for folder in (NATIVE_EVENTS / key, SUPERVISION / key / 'events'):
            receipt = read(folder / 'receipt.json')
            sha = receipt.get('input_receipt_sha256', receipt.get('shared_raw_input_receipt_sha256'))
            if sha != self.input_sha: raise ValueError('Input event identity changed')
            for row in receipt['profiles']:
                self.bank_paths[row['profile_id']] = (folder / row['path'], row)
        self.generated = 0

    def bank(self, pid):
        if pid in self.bank_paths:
            path, metadata = self.bank_paths[pid]
            if digest(path) != metadata['sha256']: raise ValueError('Frozen event bank changed')
            with np.load(path) as z: bank = {k: z[k].copy() for k in z.files}
            bank.update(input_receipt_sha256=self.input_sha,
                        curve_representation=metadata['curve_representation'], length_mm=metadata['length_mm'])
            return bank
        # New assignment to a native profile uses the identical input producer.
        if pid >= self.inp.original_profile_count: raise ValueError('An added input event bank is absent')
        a, b = self.inp.original.offsets[pid:pid + 2]; self.generated += 1
        return self.producer.build(self.inp.profiles[pid], self.inp.original.spans[a:b, :2])

    def view(self, key, seed):
        job, version = self.dataset.versions[key]
        if job != self.key: raise ValueError('View belongs to a different raw workpiece')
        prior = BASE / 'full_task_supervision_v1' / job / key; verify(prior, self.dataset, version)
        folder = SUPERVISION / job / key; receipt = read(folder / 'receipt.json')
        if (digest(folder / 'owner_witnesses.npz') != receipt['witness_sha256']
                or digest(prior / 'receipt.json') != receipt['original_supervision_receipt_sha256']):
            raise ValueError('Complete augmented owner targets changed')
        view = self.dataset.supervision(key)
        with np.load(folder / 'owner_witnesses.npz') as z: witnesses = {k: z[k].copy() for k in z.files}
        sample = self.assigner.build(view, witnesses, seed)
        return CompleteIntervalView(self.inp, self.context, sample, view, self.bank)


def checkpoint(path, model, optimizer, state):
    temporary = path.with_suffix('.incomplete')
    torch.save(dict(state, state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []), temporary)
    os.replace(temporary, path)


def run(args):
    torch.set_num_threads(2); torch.manual_seed(1031); np.random.seed(1031)
    dataset = PartitionedFullShardDataset(ROOT, BASE)
    out = BASE / local_name(args.experiment_name); out.mkdir(exist_ok=True)
    lock = (out / 'controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = {p: digest(ROOT / p) for p in SOURCES}
    if digest(WARM) != WARM_SHA: raise ValueError('Complete warm initialization changed')
    initial = torch.load(WARM, map_location='cpu', weights_only=False)
    verify_complete_interval_endpoint_fit(ROOT, BASE, initial, WARM_SHA, WARM.parent / 'receipt.json')
    visual = read(VISUAL / 'receipt.json'); supervision = read(SUPERVISION / 'receipt.json')
    if ({r['job_key'] for r in visual['records']} != set(dataset.jobs)
            or {r['version_key'] for r in supervision['records']} != set(dataset.versions)):
        raise ValueError('All TRAIN raw inputs and qualified complete views are required')
    visual_records = {r['job_key']: r for r in visual['records']}
    if args.control_version and any(k not in dataset.versions for k in args.control_version):
        raise ValueError('Controls must use registered TRAIN versions')
    protocol = dict(architecture=ARCHITECTURE, sources=sources, registry_sha256=dataset.registry_sha256,
        warm_sha256=WARM_SHA, visual_receipt_sha256=digest(VISUAL / 'receipt.json'),
        supervision_receipt_sha256=digest(SUPERVISION / 'receipt.json'), passes=3,
        steps_per_view=args.steps_per_view, learning_rate=args.learning_rate,
        control_versions=args.control_version, device=args.device, seed=1031,
        all_event_endpoint_supervision=True, whole_pair_topk_per_side=8,
        multiple_owners_retained_as_whole_alternatives=True, runtime_directed_link_head_trained=True,
        no_free_geometry_regression=True, GT_coordinates_inserted_into_input=False,
        complete_extraction_output_pending=True, held_or_test_reads=0,
        evaluation_order=['complete_selection_geometry_fields_and_original_five_regression',
                          'freeze', 'protected_two', 'final_test_seven_once_and_csv'])
    if (out / 'protocol.json').exists() and read(out / 'protocol.json') != protocol:
        raise ValueError('Resume requires the identical registered training protocol and sources')
    write(out / 'protocol.json', protocol)
    model = SupportedIntervalWeldModel().to(args.device); model.inherit_complete_query(initial['state_dict'])
    converter = TrainingTensors(initial['mean'], initial['std'], args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    state = dict(mean=initial['mean'], std=initial['std'], architecture=ARCHITECTURE,
                 protocol_sha256=digest(out / 'protocol.json'), completed=[], optimizer_steps=0)
    if (out / 'last.pt').exists():
        saved = torch.load(out / 'last.pt', map_location=args.device, weights_only=False)
        if saved['protocol_sha256'] != state['protocol_sha256']: raise ValueError('Resume protocol mismatch')
        model.load_state_dict(saved['state_dict'], strict=True); optimizer.load_state_dict(saved['optimizer'])
        state.update(completed=saved['completed'], optimizer_steps=saved['optimizer_steps'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if args.device.startswith('cuda'): torch.cuda.set_rng_state_all([r.cpu() for r in saved['cuda_rng']])
        del saved
    del initial
    priors = FrozenRun446Priors(ROOT); active_job = None; completed = set(state['completed'])
    records = read(out / 'progress.json').get('records', []) if (out / 'progress.json').exists() else []
    try:
        for epoch in range(1 if args.control_version else 3):
            plan = dataset.epoch_plan(epoch)
            if args.control_version: plan = [r for r in plan if r['version_key'] in args.control_version]
            # Group same input versions so complete XYZ images are built once per job.
            job_order = {r['job_key']: i for i, r in reversed(list(enumerate(plan)))}
            plan.sort(key=lambda r: job_order[r['job_key']])
            for item in plan:
                key = item['version_key']; visit = f'{epoch}:{key}'
                if visit in completed: continue
                write(out / 'live.json', dict(status='PREPARING_COMPLETE_VIEW', pid=os.getpid(),
                    epoch=epoch, version=key, completed_views=len(completed), optimizer_steps=state['optimizer_steps'], unix_time=time.time()))
                if active_job is None or active_job.key != item['job_key']:
                    active_job = None; gc.collect()
                    with threadpool_limits(limits=2): active_job = TrainingJob(dataset, item['job_key'], visual_records)
                with threadpool_limits(limits=2):
                    data = active_job.view(key, 1031 + epoch)
                    data.priors = priors
                history = []
                for step in range(args.steps_per_view):
                    result = complete_view_step(model, converter, data, optimizer,
                        loss_mass=item['loss_weight'] * len(dataset.versions))
                    state['optimizer_steps'] += 1; history.append(result)
                    write(out / 'live.json', dict(status='FITTING_COMPLETE_VIEW', pid=os.getpid(), epoch=epoch,
                        version=key, step=step + 1, completed_views=len(completed),
                        optimizer_steps=state['optimizer_steps'], latest=result, unix_time=time.time()))
                    print('SUPPORTED_INTERVAL_FIT', epoch, key, step + 1, result['loss'], result['coverage'], flush=True)
                completed.add(visit); state['completed'].append(visit)
                record = dict(epoch=epoch, version_key=key, job_key=item['job_key'], children=data.count,
                    known_attributes=int(data.view.label_known.sum()), known_frames=int(data.view.frame_algebraically_known.sum()),
                    assignment_report=data.sample['report'], directed_links=data.link_audit, updates=history)
                records.append(record); checkpoint(out / 'last.pt', model, optimizer, state)
                write(out / 'progress.json', dict(status='RUNNING_COMPLETE_TRAIN_OPTIMIZATION', records=records,
                    completed_views=len(completed), optimizer_steps=state['optimizer_steps'], checkpoint_sha256=digest(out / 'last.pt')))
                del data; gc.collect()
            if not args.control_version:
                epoch_rows = [r for r in records if r['epoch'] == epoch]
                if ({r['version_key'] for r in epoch_rows} != set(dataset.versions)
                        or sum(r['children'] for r in epoch_rows) != 9964):
                    raise ValueError('Incomplete full pass cannot be called complete')
                # Immutable epoch weights share storage with the just-written file.
                epoch_path = out / f'epoch_{epoch + 1}.pt'
                if not epoch_path.exists(): os.link(out / 'last.pt', epoch_path)
                write(out / f'epoch_{epoch + 1}_receipt.json', dict(status='ALL_171_TRAIN_VIEWS_OPTIMIZED_FULL_OUTPUT_REQUIRED',
                    children=9964, records=epoch_rows, checkpoint_sha256=digest(epoch_path),
                    complete_output_and_csv_pending=True, extraction_accuracy_claimed=False))
        write(out / 'receipt.json', dict(status=('ACTUAL_COMPLETE_VIEW_TRAINING_CONTROL_FINISHED' if args.control_version
            else 'THREE_FULL_TRAIN_PASSES_OPTIMIZED_COMPLETE_OUTPUT_EVALUATION_PENDING'),
            completed_views=len(completed), optimizer_steps=state['optimizer_steps'], records=records,
            protocol_sha256=digest(out / 'protocol.json'), checkpoint_sha256=digest(out / 'last.pt'),
            held_or_test_reads=0, model_promoted=False, full_extraction_accuracy_claimed=False))
    except BaseException as exc:
        write(out / 'failure.json', dict(error=str(exc), traceback=traceback.format_exc(),
            completed_views=len(completed), optimizer_steps=state['optimizer_steps']))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--experiment-name', required=True)
    p.add_argument('--steps-per-view', type=int, default=8); p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--control-version', action='append', default=[])
    a = p.parse_args()
    if a.steps_per_view < 1 or a.learning_rate <= 0: p.error('Positive optimization settings required')
    run(a)
