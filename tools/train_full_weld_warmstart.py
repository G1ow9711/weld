"""Roll every admitted TRAIN shard into a complete-reference downstream fit.

This first phase fits free line/arc geometry, process states, raw frames and
teacher ordered-parent structure. It is not deployment output selection or an
accuracy evaluation. Whole-prediction parent selection and regression checks
must follow; no model is promoted by this receipt.
"""
import argparse
import fcntl
import gc
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.server_full_training_samples import FullTrainingAssignments
from weld_pipeline.server_full_weld_model import FullWeldModel, complete_child_losses
from weld_pipeline.sjg_local_runtime_904 import digest, read, write
from tools.prepare_full_task_supervision import verify as verify_supervision

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'
WARM = ROOT / 'server_rebuild/expansion_v1/positive_query_head_v1/joint_reference_v1'
SOURCES = ['weld_pipeline/server_full_shard_dataset.py',
           'weld_pipeline/server_full_reference_witnesses.py',
           'weld_pipeline/server_full_training_samples.py',
           'weld_pipeline/server_full_weld_model.py',
           'tools/train_full_weld_warmstart.py']


def load_query_initialization():
    receipt = read(WARM / 'receipt.json')
    if digest(WARM / 'model.pt') != receipt['model_sha256']:
        raise ValueError('Query initialization checkpoint changed')
    state = torch.load(WARM / 'model.pt', map_location='cpu', weights_only=False)
    if state['mean'].shape != (457,) or state['std'].shape != (457,) or not np.all(state['std'] > 0):
        raise ValueError('Legacy TRAIN normalization changed')
    return state


class TrainingTensors:
    def __init__(self, mean, std, device):
        self.device = device
        self.mean = self.tensor(mean)
        self.std = self.tensor(std)

    def tensor(self, values):
        array = np.asarray(values)
        if array.dtype == np.bool_:
            dtype = torch.bool
        elif np.issubdtype(array.dtype, np.integer):
            dtype = torch.long
        else:
            dtype = torch.float32
        return torch.tensor(array, device=self.device, dtype=dtype)

    def features(self, raw):
        return torch.clamp((self.tensor(raw) - self.mean) / self.std, -20, 20)

    def normalize_tokens(self, raw):
        # Original feature columns use original TRAIN-only statistics. Order
        # coordinates keep their meaning; log length has a declared fixed scale.
        return torch.cat((torch.clamp((raw[:, :456] - self.mean[:456]) / self.std[:456], -20, 20),
                          raw[:, 456:458], raw[:, 458:459] / 10), 1)

    def batch(self, inp, sample):
        tokens = {int(pid): self.tensor(inp.tokens(int(pid))[0])
                  for pid in np.unique(sample['profile_ids'])}
        return dict(features=self.features(sample['features']),
                    profile_ids=self.tensor(sample['profile_ids']), token_sequences=tokens,
                    normalize_tokens=self.normalize_tokens, geometry=self.tensor(sample['geometry']))


def fit_step(model, converter, inp, sample, optimizer, loss_mass):
    batch = converter.batch(inp, sample)
    hidden = model.encode_queries(**batch)
    prediction = model.predict(hidden, converter.tensor(sample['features'][:, 456]))
    targets = {name: converter.tensor(array) for name, array in sample['targets'].items()}
    losses = complete_child_losses(prediction, targets)
    parent_logits = model.parent_logits(hidden, sample['parent_children'])
    losses['parent_closure'] = F.cross_entropy(parent_logits, converter.tensor(sample['parent_closed']))
    if len(sample['link_left']):
        link = model.link_logits(hidden, converter.tensor(sample['link_left']),
                                 converter.tensor(sample['link_right']),
                                 converter.tensor(sample['link_relative']))
        # Balance the two kinds of known-reference adjacency evidence.
        labels = converter.tensor(sample['link_labels'])
        error = F.cross_entropy(link, labels, reduction='none')
        terms = [error[labels == value].mean() for value in (0, 1) if (labels == value).any()]
        losses['reference_child_link'] = torch.stack(terms).mean()
    else:
        losses['reference_child_link'] = hidden.sum() * 0
    # The reference classifier has no warm-start loss and its own weights are
    # frozen. Shared encoder updates can change its scores; complete-output
    # reference costs and regression checks must fit/validate selection later.
    weights = dict(line_points=10., arc_shape=3., interval=1.,
                   attribute_state=.5, attribute_mm=.5, frame_axes=.5,
                   frame_offsets=2., parent_closure=.5, reference_child_link=.5)
    objective = sum(value * weights.get(name, 1.) for name, value in losses.items())
    if not torch.isfinite(objective):
        raise ValueError('Complete-reference training loss became nonfinite')
    optimizer.zero_grad(set_to_none=True)
    (objective * loss_mass).backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(objective.detach()), gradient_norm=float(grad),
                components={name: float(value.detach()) for name, value in losses.items()})


def save_checkpoint(path, model, optimizer, state):
    temporary = path.with_suffix('.incomplete')
    payload = dict(state, state_dict={name: value.detach().cpu() for name, value in model.state_dict().items()},
                   optimizer=optimizer.state_dict())
    torch.save(payload, temporary)
    # Receipt is emitted only after a real reload of the atomic checkpoint.
    reloaded = torch.load(temporary, map_location='cpu', weights_only=False)
    if reloaded['optimizer_steps'] != state['optimizer_steps'] or reloaded['completed_views'] != state['completed_views']:
        raise ValueError('Checkpoint did not preserve actual fit progress')
    os.replace(temporary, path)


def main(epochs, steps_per_view):
    if epochs < 1 or steps_per_view < 1:
        raise ValueError('Positive complete-coverage epoch and step counts required')
    out = BASE / 'full_weld_warmstart_v1'
    out.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(1031)
    with (out / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dataset = FullShardDataset(ROOT)
        initial = load_query_initialization()
        source = {name: digest(ROOT / name) for name in SOURCES}
        settings = dict(stage='FULL_REFERENCE_GEOMETRY_PROCESS_FRAME_PARENT_TEACHER_WARMSTART',
                        registry_sha256=dataset.registry_sha256,
                        sources=source, epochs=epochs, steps_per_view=steps_per_view,
                        registered_versions=len(dataset.versions), registered_families=len(dataset.families),
                        point_features='Frozen original run162/run26 input; no PTv3 gradient through NumPy',
                        query_initialization_sha256=digest(WARM / 'model.pt'),
                        query_initialization_is_final_run446_selector=False,
                        original_run446_unchanged=True, learning_rate=.0001, seed=1031,
                        every_view_full_reference_children=True,
                        missing_carrier_policy='Explicit weak input query assignment plus complete free-geometry regression; not D3-positive',
                        same_input_version_policy='Expected loss of separate complete TRAIN views; no cross-version parent or field merging',
                        family_version_weights='Registry loss weight times all registered version count at each view step',
                        physical_background_labels=False, physical_total_parent_count_loss=False,
                        parent_order='TRAIN-only teacher sequences; deployment parent proposal/selection training remains required',
                        reference_selector_fit=False, reference_classifier_weights_frozen=True,
                        reference_scores_can_change_with_shared_encoder=True,
                        validation_or_test_label_reads=0,
                        diagnostic64_fit=False, no_promotion_without_full_output_evaluation=True)
        protocol_path = out / 'protocol.json'
        if protocol_path.exists():
            protocol = read(protocol_path)
            if protocol['settings'] != settings:
                raise ValueError('Training protocol changed; preserve this experiment and declare a new version')
        else:
            commit = subprocess.run(['git', 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
            protocol = dict(settings=settings, source_commit=commit, started_unix=time.time())
            write(protocol_path, protocol)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = FullWeldModel().to(device)
        model.inherit_query(initial['state_dict'])
        for parameter in model.reference.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0001, weight_decay=.0001)
        converter = TrainingTensors(initial['mean'], initial['std'], device)
        checkpoint_path = out / 'last.pt'
        completed, optimizer_steps, view_evidence = set(), 0, {}
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if checkpoint['protocol_sha256'] != digest(protocol_path):
                raise ValueError('Checkpoint training protocol binding changed')
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            completed = set(checkpoint['completed_views'])
            optimizer_steps = checkpoint['optimizer_steps']
            view_evidence = checkpoint['view_evidence']
            if set(checkpoint['sources']) != set(source) or checkpoint['sources'] != source:
                raise ValueError('Checkpoint training sources changed')
        model.train()
        for epoch in range(epochs):
            plan = dataset.epoch_plan(epoch, require_ready=False)
            order = {entry['version_key']: entry for entry in plan}
            while any(f'{epoch}:{key}' not in completed for key in order):
                progressed = False
                ready = set(dataset.ready_jobs())
                for job_key, job in dataset.jobs.items():
                    if job_key not in ready:
                        continue
                    pending = [v for v in job['versions'] if f'{epoch}:{v["key"]}' not in completed]
                    if not pending or not all((BASE / 'full_task_supervision_v1' / job_key / v['key'] / 'receipt.json').exists() for v in pending):
                        continue
                    if {name: digest(ROOT / name) for name in SOURCES} != source:
                        raise ValueError('Training source edited during active fit')
                    inp = dataset.input_case(job_key)
                    builder = FullTrainingAssignments(inp)
                    for version in sorted(pending, key=lambda v: list(order).index(v['key'])):
                        key = version['key']
                        folder = BASE / 'full_task_supervision_v1' / job_key / key
                        verify_supervision(folder, dataset, version)
                        supervision = dataset.supervision(key)
                        with np.load(folder / 'owner_witnesses.npz', allow_pickle=False) as arrays:
                            witnesses = {name: arrays[name] for name in arrays.files}
                        results, reports = [], []
                        for step in range(steps_per_view):
                            # Deterministic seed controls carrier alternatives,
                            # never chooses a geometry coordinate as an input.
                            seed = np.random.SeedSequence([1031, epoch, list(dataset.versions).index(key), step])
                            with threadpool_limits(limits=2):
                                sample = builder.build(supervision, witnesses, seed)
                            result = fit_step(model, converter, inp, sample, optimizer,
                                              order[key]['loss_weight'] * len(dataset.versions))
                            optimizer_steps += 1
                            results.append(result)
                            reports.append(sample['report'])
                            write(out / 'progress.json', dict(status='OPTIMIZER_RUNNING_COMPLETE_REFERENCE_WARMSTART',
                                  controller_pid=os.getpid(), epoch=epoch, epochs=epochs, version_key=key,
                                  optimizer_steps=optimizer_steps, completed_views=len(completed),
                                  versions_fitted=len({value.split(':', 1)[1] for value in completed}),
                                  registered_versions=len(dataset.versions), device=device,
                                  actual_model_parameter_updates=True, model_output_accuracy_not_evaluated=True,
                                  current_loss=result['loss'], unix_time=time.time()))
                        completed.add(f'{epoch}:{key}')
                        view_evidence[f'{epoch}:{key}'] = dict(job_key=job_key,
                              input_receipt_sha256=digest(dataset.shards / job_key / 'receipt.json'),
                              label_receipt_sha256=version['label_receipt_sha256'],
                              owner_witness_receipt_sha256=digest(folder / 'receipt.json'),
                              family_equal_version_weight=order[key]['loss_weight'],
                              reference_children=supervision.target['original_children'],
                              reports=reports, fit_results=results)
                        checkpoint = dict(protocol_sha256=digest(protocol_path), sources=source,
                                          optimizer_steps=optimizer_steps, completed_views=sorted(completed),
                                          view_evidence=view_evidence, mean=initial['mean'], std=initial['std'])
                        save_checkpoint(checkpoint_path, model, optimizer, checkpoint)
                        write(out / 'checkpoint_receipt.json', dict(status='PARTIAL_COVERAGE_WARMSTART_NOT_PROMOTED',
                              checkpoint_sha256=digest(checkpoint_path), optimizer_steps=optimizer_steps,
                              completed_views=len(completed), registered_versions=len(dataset.versions),
                              source_commit=protocol['source_commit'], protocol_sha256=digest(protocol_path),
                              validation_or_test_label_reads=0, model_output_accuracy_evaluations=0))
                        print('FULL_WELD_VIEW_FITTED', epoch, key, 'children', supervision.target['original_children'],
                              'steps', optimizer_steps, 'loss', results[-1]['loss'], flush=True)
                        progressed = True
                        del sample, witnesses, supervision
                    del inp, builder
                    gc.collect()
                if not progressed:
                    write(out / 'progress.json', dict(status='WAITING_INPUTS_AFTER_ACTUAL_OPTIMIZER_UPDATES',
                          controller_pid=os.getpid(), epoch=epoch, epochs=epochs,
                          optimizer_steps=optimizer_steps, completed_views=len(completed),
                          versions_fitted=len({value.split(':', 1)[1] for value in completed}),
                          registered_versions=len(dataset.versions), registered_cloud_jobs=len(dataset.jobs),
                          ready_cloud_jobs=len(ready), actual_model_parameter_updates=optimizer_steps > 0,
                          model_output_accuracy_not_evaluated=True, unix_time=time.time()))
                    time.sleep(10)
            # Full coverage is only certified after all registered sources have
            # actually been consumed with their own complete version labels.
            dataset.require_complete_inputs()
            epoch_checkpoint = out / f'epoch_{epoch:03d}.pt'
            if not epoch_checkpoint.exists():
                save_checkpoint(epoch_checkpoint, model, optimizer, checkpoint)
            print('COMPLETE_ALL_TRAIN_VIEWS_WARMSTART_EPOCH', epoch, len(order), flush=True)
        write(out / 'receipt.json', dict(status='ALL_TRAIN_WARMSTART_FITTED_NOT_PROMOTED_NOT_COMPLETE_INFERENCE',
              optimizer_steps=optimizer_steps, epochs=epochs, training_versions=len(dataset.versions),
              training_families=len(dataset.families), checkpoint_sha256=digest(checkpoint_path),
              completed_views=len(completed), protocol_sha256=digest(protocol_path),
              complete_reference_children=sum(v['children'] for _, v in dataset.versions.values()),
              reference_selection_training_and_complete_output_evaluation_remaining=True,
              validation_or_test_label_reads=0, model_output_accuracy_evaluations=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--steps-per-view', type=int, default=8)
    args = parser.parse_args()
    try:
        main(args.epochs, args.steps_per_view)
    except BaseException:
        out = BASE / 'full_weld_warmstart_v1'
        out.mkdir(exist_ok=True)
        write(out / 'failure.json', dict(error=traceback.format_exc(), unix_time=time.time()))
        raise
