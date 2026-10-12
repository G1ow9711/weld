"""Continue every qualified TRAIN view with geometry in three-mm units.

The actual completed alternating local-XYZ model is the warm start. Every
original child, class, field mask, frame, parent and link loss remains. Only
geometry penalties change to the existing tolerance-scaled implementation.
Memoized preparation reproduces the original computations and seed variants.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import gc
import os
from pathlib import Path
import shutil
import sys
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
from tools.train_full_local_query_geometry import SOURCES as ORIGINAL_SOURCES, save
from tools.compare_metric_geometry_loss import metric_step as fit_step
from weld_pipeline.server_cached_local_query_training import CachedLocalQueryInput, CachedLocalTrainingTensors
from weld_pipeline.server_memoized_training_queries import MemoizedTrainingAssignments
from weld_pipeline.server_local_query_checkpoint import COMPLETE_STATUS, verify_complete_local_query_fit
from weld_pipeline.server_local_query_weld_model import LocalQueryWeldModel, ARCHITECTURE
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.server_memoized_training_queries import MemoizedRawQueryNeighborhood
from weld_pipeline.sjg_local_runtime_904 import digest, read, write

ROOT = Path(__file__).resolve().parents[1]
SOURCES = list(dict.fromkeys(['tools/train_full_local_query_geometry_metric.py',
    'tools/check_local_metric_warmstart.py',
    'tools/compare_metric_geometry_loss.py', 'weld_pipeline/server_metric_geometry_losses.py',
    'weld_pipeline/server_memoized_training_queries.py', 'tools/check_full_train_query_memoization.py',
    'weld_pipeline/server_cached_local_query_training.py', 'tools/check_cached_local_query_training.py',
    'tools/diagnose_full_query_geometry.py', *ORIGINAL_SOURCES]))


def memory_guard():
    limit = Path('/sys/fs/cgroup/memory.max')
    current = Path('/sys/fs/cgroup/memory.current')
    if limit.exists() and current.exists() and limit.read_text().strip() != 'max':
        if int(limit.read_text()) - int(current.read_text()) < 20 * 1024**3:
            raise RuntimeError('Complete mixed-view input cache cannot retain 20 GiB process headroom')


def new_object_bytes(value, seen):
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(new_object_bytes(k, seen) + new_object_bytes(v, seen) for k,v in value.items())
    elif isinstance(value, (list, tuple)):
        size += sum(new_object_bytes(v, seen) for v in value)
    elif isinstance(value, CachedLocalQueryInput):
        size += new_object_bytes(vars(value), seen)
    elif isinstance(value, np.ndarray) and isinstance(value.base, np.ndarray):
        size += new_object_bytes(value.base, seen)
    return size


def cache_epoch(dataset, epoch, args, state, out, sources):
    cached, seen, cache_bytes = {}, set(), 0
    for job_key, job in dataset.jobs.items():
        pending = [v for v in job['versions'] if f'{epoch}:{v["key"]}' not in state['completed_views']]
        if not pending:
            continue
        memory_guard()
        if sources != {name:digest(ROOT / name) for name in sources}:
            raise ValueError('Mixed-view source changed before complete input admission')
        raw = ROOT.parent / 'data/G_Data' / job['workpiece_source']
        if digest(raw) != job['workpiece_sha256']:
            raise ValueError('Actual registered raw XYZ changed before mixed-view input caching')
        xyz = ply_points(raw)
        if len(xyz) != job['original_points']:
            raise ValueError('Complete raw XYZ point count differs')
        inp = dataset.input_case(job_key)
        neighborhoods = MemoizedRawQueryNeighborhood(xyz, inp)
        builder = MemoizedTrainingAssignments(inp)
        shared_tokens = {}
        for version in pending:
            key = version['key']; identity = f'{epoch}:{key}'
            view = dataset.supervision(key)
            folder = dataset.base / 'full_task_supervision_v1' / job_key / key
            verify(folder, dataset, version)
            with np.load(folder / 'owner_witnesses.npz', allow_pickle=False) as arrays:
                witnesses = {name:arrays[name] for name in arrays.files}
            evidence = dict(job_key=job_key, input_receipt_sha256=digest(dataset.shards / job_key / 'receipt.json'),
                label_receipt_sha256=version['label_receipt_sha256'], owner_witness_receipt_sha256=digest(folder / 'receipt.json'),
                reference_children=view.target['original_children'], family_equal_version_weight=version['family_equal_version_weight'],
                raw_xyz_sha256=job['workpiece_sha256'], complete_raw_xyz_points=len(xyz))
            previous = state['view_evidence'].get(identity)
            if previous is not None and any(previous[name] != value for name,value in evidence.items()):
                raise ValueError('Recovered mixed-view fitting input or labels changed')
            if previous is None:
                state['view_evidence'][identity] = dict(evidence, reports=[], fit_results=[], local_context_reports=[],
                                                       cached_input_bindings=[])
            done_steps = len(state['view_evidence'][identity]['fit_results'])
            variants = {}
            for step in range(done_steps, args.steps_per_view):
                memory_guard()
                with threadpool_limits(limits=2):
                    sample = builder.build(view, witnesses, args.seed + epoch * 100000 + step)
                    actual = CachedLocalQueryInput(inp, sample, neighborhoods, shared_tokens)
                actual.verify(sample)
                for name in ('features', 'profile_ids', 'geometry', 'input_query_positions_mm'):
                    sample[name].flags.writeable = False
                entry = (actual, sample)
                cache_bytes += new_object_bytes(entry, seen)
                if cache_bytes > args.max_cache_gib * 1024**3:
                    raise RuntimeError('Declared complete mixed-view cache budget exceeded; no view or seed is truncated')
                variants[step] = entry
            cached[key] = variants
            atomic(out / 'progress.json', dict(status='PREPARING_ALL_TRAIN_MIXED_VIEW_INPUTS', controller_pid=os.getpid(),
                epoch=epoch, cached_versions=len(cached), registered_versions=len(dataset.versions),
                optimizer_steps=state['optimizer_steps'], cache_bytes=cache_bytes, held_reference_reads=0, unix_time=time.time()))
            print('MIXED_COMPLETE_VIEW_CACHED', epoch, key, len(cached), '/', len(dataset.versions), flush=True)
            del view, witnesses, sample, actual, entry
        del inp, neighborhoods, builder, shared_tokens, xyz
        gc.collect()
    return cached, cache_bytes


def main(args):
    if min(args.epochs, args.steps_per_view, args.max_cache_gib) < 1:
        raise ValueError('Complete positive training and cache budgets are required')
    if args.control_run and (args.epochs, args.steps_per_view) != (1, 1):
        raise ValueError('The all-input loop control must perform exactly one real update per TRAIN view')
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    base = ROOT / 'server_rebuild' / local_name(args.dataset_name)
    dataset = PartitionedFullShardDataset(ROOT, base)
    dataset.require_complete_inputs()
    control_path = base / 'cached_local_query_input_controls_20261012_v2/receipt.json'
    control = read(control_path)
    if (control['status'] != 'PASS_ACTUAL_COMPLETE_TRAIN_CACHED_LOCAL_INPUT_AND_TWO_UPDATE_PARITY'
            or not control['all_named_predictions_bitwise_equal']
            or not control['all_complete_task_losses_and_two_parameter_updates_bitwise_equal']
            or not control['non_deployed_queries_rejected'] or not control['changed_shared_tokens_rejected']
            or control['held_reference_reads'] != 0
            or control['sources'] != {name:digest(ROOT / name) for name in control['sources']}):
        raise ValueError('Actual complete TRAIN cached-input and update parity controls are required')
    memo_folder = base / 'all_train_query_memoization_parity_20261012_v1'
    memo = read(memo_folder / 'receipt.json')
    memo_protocol = read(memo_folder / 'protocol.json')
    if (memo['status'] != 'PASS_ALL_ACTUAL_TRAIN_QUERY_MEMOIZATION_BITWISE_PARITY'
            or memo['totals']['versions'] != len(dataset.versions)
            or memo['totals']['input_jobs'] != len(dataset.jobs)
            or memo['totals']['complete_batches'] != len(dataset.versions) * 64
            or memo['totals']['children_per_update_round'] != sum(v['children'] for _, v in dataset.versions.values())
            or not memo['actual_full_view_update_parity']['all_losses_parameters_and_optimizer_states_bitwise_equal']
            or memo['held_reference_reads'] != 0 or memo['production_optimizer_updates'] != 0
            or memo_protocol['sources'] != {name:digest(ROOT / name) for name in memo_protocol['sources']}):
        raise ValueError('Actual complete all-TRAIN preparation and update equivalence is required')
    initial_path = (ROOT / args.warm_checkpoint).resolve(strict=True)
    initial_path.relative_to(base)
    if digest(initial_path) != args.warm_sha256:
        raise ValueError('The declared completed warm-start model changed')
    initial = torch.load(initial_path, map_location='cpu', weights_only=False)
    initial_fit = verify_complete_local_query_fit(ROOT, base, initial,
        args.warm_sha256, initial_path.parent / 'receipt.json')
    if initial['architecture'] != ARCHITECTURE or initial['optimizer_steps'] < 1:
        raise ValueError('An actual completed local-XYZ model is required')
    metric_control_folder = base / 'local_query_metric_warmstart_control_20261012_v1'
    metric_control = read(metric_control_folder / 'receipt.json')
    metric_control_protocol = read(metric_control_folder / 'protocol.json')
    if (metric_control['status'] != 'PASS_ACTUAL_WARM_COMPLETE_VIEW_METRIC_LOSS_AND_GEOMETRY_GRADIENTS'
            or metric_control['warm_checkpoint_sha256'] != args.warm_sha256
            or not metric_control['all_frozen_original_parameters_bitwise_preserved']
            or not metric_control['all_non_geometry_child_losses_bitwise_preserved']
            or not all(np.isfinite(v) and v > 0 for v in metric_control['actual_geometry_head_gradient_sums'].values())
            or metric_control['actual_control_optimizer_updates'] != 2
            or metric_control['held_reference_reads'] != 0 or metric_control['production_optimizer_updates'] != 0
            or metric_control_protocol['sources'] != {name:digest(ROOT / name) for name in metric_control_protocol['sources']}):
        raise ValueError('Actual complete-view metric updates and preserved frozen parameters are required')
    out = base / local_name(args.experiment_name); out.mkdir(exist_ok=True)
    with (out / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = {name:digest(ROOT / name) for name in SOURCES}
        settings = dict(read(base / 'full_local_query_geometry_grid_v4_round01/protocol.json')['settings'])
        original_budget = (settings['epochs'], settings['steps_per_view'])
        settings.update(stage='COMPLETE_TRAIN_LOCAL_RAW_XYZ_TOLERANCE_SCALED_GEOMETRY',
            sources=sources, epochs=args.epochs, steps_per_view=args.steps_per_view, seed=args.seed,
            registry_sha256=dataset.registry_sha256, registered_versions=len(dataset.versions),
            registered_families=len(dataset.families), local_initialization_sha256=digest(initial_path),
            ordering='Each update round shuffles every separate complete TRAIN view; original per-view seed variations retained',
            per_view_consecutive_updates=1, complete_cached_raw_xyz_query_inputs=True,
            original_complete_loss_retained=False,
            original_class_field_frame_parent_and_link_losses_retained=True,
            geometry_loss_recipe='Euclidean line endpoints, 65 analytic ordered arc samples and known carrier bounds in 3-mm units',
            geometry_loss_source_sha256=digest(ROOT / 'weld_pipeline/server_metric_geometry_losses.py'),
            geometry_loss_weights=dict(line_points=1., arc_shape=1., interval=1.),
            warm_start_complete_fit=initial_fit,
            warm_start_checkpoint_path=str(initial_path.relative_to(ROOT)),
            warm_start_checkpoint_sha256=args.warm_sha256,
            inherited_query_optimizer_steps=initial['optimizer_steps'],
            new_optimizer_for_changed_geometry_loss=True,
            memoization_control_receipt_sha256=digest(memo_folder / 'receipt.json'),
            metric_warmstart_control_receipt_sha256=digest(metric_control_folder / 'receipt.json'),
            memoized_input_preparation_only=True,
            original_per_view_budget_unchanged=(args.epochs, args.steps_per_view) == original_budget,
            cache_control_receipt_sha256=digest(control_path),
            max_cache_gib=args.max_cache_gib, cache_overflow_policy='Fail; never discard a view, child, profile or seed',
            all_input_loop_control_only=args.control_run,
            diagnostic_training_control_not_main_candidate=args.control_run)
        protocol = out / 'protocol.json'
        if protocol.exists():
            if read(protocol)['settings'] != settings:
                raise ValueError('Mixed-view recipe changed; declare a new independent experiment')
        else:
            write(protocol, dict(settings=settings, source_commit=git_head(ROOT),
                recorded_utc=datetime.now(timezone.utc).isoformat()))
        warm = out / 'warm_initialization.pt'
        if not warm.exists():
            shutil.copyfile(initial_path, warm)
        if digest(warm) != digest(initial_path):
            raise ValueError('The exact original initialization changed')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = LocalQueryWeldModel().to(device); model.load_state_dict(initial['state_dict'], strict=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0001, weight_decay=.0001)
        state = dict(architecture=ARCHITECTURE, protocol_sha256=digest(protocol), sources=sources,
            optimizer_steps=0, completed_views=[], view_evidence={}, mean=initial['mean'], std=initial['std'],
            inherited_optimizer_steps=initial.get('inherited_optimizer_steps', 0) + initial['optimizer_steps'])
        last = out / 'last.pt'
        if last.exists():
            saved = torch.load(last, map_location=device, weights_only=False)
            if saved['protocol_sha256'] != state['protocol_sha256'] or saved['sources'] != sources:
                raise ValueError('Mixed-view recovery belongs to a different data or source recipe')
            model.load_state_dict(saved.pop('state_dict')); optimizer.load_state_dict(saved.pop('optimizer'))
            torch.set_rng_state(saved.pop('torch_rng').cpu()); state = saved
            if sum(len(v['fit_results']) for v in state['view_evidence'].values()) != state['optimizer_steps']:
                raise ValueError('Recovered actual mixed-view updates and evidence differ')
        converter = CachedLocalTrainingTensors(initial['mean'], initial['std'], device)
        keys = list(dataset.versions); model.train()
        try:
            for epoch in range(args.epochs):
                if all(f'{epoch}:{key}' in state['completed_views'] for key in keys):
                    continue
                cached, cache_bytes = cache_epoch(dataset, epoch, args, state, out, sources)
                for step in range(args.steps_per_view):
                    if sources != {name:digest(ROOT / name) for name in sources}:
                        raise ValueError('Live mixed-view training source changed')
                    memory_guard()
                    order = np.random.default_rng(np.random.SeedSequence([args.seed, epoch, step, 33])).permutation(len(keys))
                    for position in order:
                        key = keys[int(position)]; identity = f'{epoch}:{key}'
                        if identity in state['completed_views']:
                            continue
                        view = state['view_evidence'][identity]
                        if len(view['fit_results']) > step:
                            continue
                        if len(view['fit_results']) != step:
                            raise ValueError('A complete mixed-view update round omitted an earlier step')
                        inp, sample = cached[key][step]; inp.verify(sample)
                        with threadpool_limits(limits=2):
                            result = fit_step(model, converter, inp, sample, optimizer,
                                view['family_equal_version_weight'] * len(dataset.versions))
                        view['fit_results'].append(result); view['reports'].append(sample['report'])
                        view['local_context_reports'].append(converter.last_context_report)
                        view['cached_input_bindings'].append(inp.bindings); state['optimizer_steps'] += 1
                        if len(view['fit_results']) == args.steps_per_view:
                            state['completed_views'].append(identity); state['completed_views'].sort()
                        atomic(out / 'progress.json', dict(status='ACTUAL_ALL_TRAIN_ALTERNATING_COMPLETE_VIEW_UPDATES',
                            controller_pid=os.getpid(), epoch=epoch, update_round=step+1,
                            update_rounds_per_epoch=args.steps_per_view, version_key=key,
                            optimizer_steps=state['optimizer_steps'], completed_views=len(state['completed_views']),
                            registered_versions=len(dataset.versions), reference_children=view['reference_children'],
                            complete_raw_points=view['complete_raw_xyz_points'], cache_bytes=cache_bytes,
                            held_reference_reads=0, unix_time=time.time()))
                    save(last, model, optimizer, state)
                    print('ALL_TRAIN_ALTERNATING_UPDATE_ROUND', epoch, step+1, 'updates', state['optimizer_steps'], flush=True)
                save(out / f'epoch_{epoch:03d}.pt', model, optimizer, state)
                del cached
                gc.collect()
            expected = {f'{epoch}:{key}' for epoch in range(args.epochs) for key in keys}
            if set(state['completed_views']) != expected or state['optimizer_steps'] != len(expected)*args.steps_per_view:
                raise ValueError('A registered complete TRAIN view or real update was omitted')
            status = ('ALL_TRAIN_ALTERNATING_ONE_STEP_CONTROL_NOT_MAIN_TRAINING'
                      if args.control_run else COMPLETE_STATUS)
            atomic(out / 'receipt.json', dict(status=status, architecture=ARCHITECTURE,
                checkpoint_sha256=digest(last), protocol_sha256=digest(protocol), epochs=args.epochs,
                optimizer_steps=state['optimizer_steps'], training_versions=len(dataset.versions),
                training_families=len(dataset.families), completed_views=len(expected),
                complete_reference_children=sum(v['children'] for _,v in dataset.versions.values()),
                validation_or_test_label_reads=0, model_output_accuracy_evaluations=0,
                complete_selection_and_validation_required=True, alternating_complete_views=True,
                diagnostic_training_control_not_main_candidate=args.control_run,
                tolerance_scaled_geometry_loss=True, warm_start_checkpoint_sha256=args.warm_sha256))
            loaded = torch.load(last, map_location='cpu', weights_only=False)
            if args.control_run:
                if (sum(len(v['fit_results']) for v in loaded['view_evidence'].values()) != len(dataset.versions)
                        or sum(v['reference_children'] for v in loaded['view_evidence'].values())
                            != sum(v['children'] for _,v in dataset.versions.values())
                        or any(len(v['local_context_reports']) != 1
                               or v['local_context_reports'][0]['input_neighborhood_rows'] != v['reference_children']
                               or v['local_context_reports'][0]['raw_point_tree_size'] != v['complete_raw_xyz_points']
                               for v in loaded['view_evidence'].values())):
                    raise ValueError('The actual all-input one-step control omitted complete supervision or raw context')
                try:
                    verify_complete_local_query_fit(ROOT, base, loaded, digest(last), out / 'receipt.json')
                except ValueError:
                    pass
                else:
                    raise ValueError('A diagnostic one-step control entered main full-model admission')
                atomic(out / 'control_verification.json', dict(
                    status='PASS_ACTUAL_ALL_TRAIN_MIXED_VIEW_INPUT_LOOP_NOT_MODEL_ACCURACY',
                    training_versions=len(dataset.versions),
                    reference_children=sum(v['reference_children'] for v in loaded['view_evidence'].values()),
                    actual_optimizer_steps=loaded['optimizer_steps'], all_raw_xyz_context_rows_verified=True,
                    all_original_supervision_retained=True, original_full_model_admission_rejected_control=True,
                    validation_or_test_label_reads=0, full_training_not_claimed=True))
                print('ALL_TRAIN_ALTERNATING_ONE_STEP_CONTROL_FINISHED', state['optimizer_steps'], flush=True)
                return
            verified = verify_complete_local_query_fit(ROOT, base, loaded, digest(last), out / 'receipt.json')
            atomic(out / 'complete_fit_verification.json', verified)
            print('ALL_TRAIN_ALTERNATING_LOCAL_XYZ_QUERY_TRAINING_FINISHED', state['optimizer_steps'], flush=True)
        except BaseException:
            if state['optimizer_steps'] > 0:
                save(last, model, optimizer, state)
            atomic(out / 'failure.json', dict(traceback=traceback.format_exc(), optimizer_steps=state['optimizer_steps'],
                completed_views=len(state['completed_views']), model_not_promoted=True))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', default='full_dataset_v2')
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--warm-checkpoint', required=True)
    parser.add_argument('--warm-sha256', required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--steps-per-view', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1031)
    parser.add_argument('--max-cache-gib', type=float, default=16.)
    parser.add_argument('--control-run', action='store_true')
    main(parser.parse_args())
