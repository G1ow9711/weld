"""Independent all-TRAIN query candidate with alternating complete views.

The original complete loss, initialization, raw XYZ inputs and per-view budget
are preserved. Each update round shuffles all versions instead of completing
64 consecutive updates on one workpiece. Input caching is admitted by actual
prediction/loss/update parity, and every original seed variation is retained.
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
from tools.train_full_weld_reference_recall import fit_step
from weld_pipeline.server_cached_local_query_training import CachedLocalQueryInput, CachedLocalTrainingTensors
from weld_pipeline.server_deployed_query_assignments import DeployedGridTrainingAssignments
from weld_pipeline.server_local_query_checkpoint import COMPLETE_STATUS, verify_complete_local_query_fit
from weld_pipeline.server_local_query_weld_model import LocalQueryWeldModel, ARCHITECTURE
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.server_query_point_neighborhood import RawQueryNeighborhood
from weld_pipeline.sjg_local_runtime_904 import digest, read, write

ROOT = Path(__file__).resolve().parents[1]
SOURCES = list(dict.fromkeys(['tools/train_full_local_query_geometry_mixed.py',
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
        neighborhoods = RawQueryNeighborhood(xyz)
        builder = DeployedGridTrainingAssignments(inp)
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
    initial_control = read(base / 'local_query_xyz_initialization_control_20261012_v1/receipt.json')
    initial_path = ROOT / initial_control['local_initialization_path']
    if (initial_control['status'] != 'PASS_ACTUAL_RAW_XYZ_LOCAL_QUERY_ZERO_RESIDUAL_AND_GRADIENT_CONTROLS'
            or initial_control['sources'] != {name:digest(ROOT / name) for name in initial_control['sources']}
            or digest(initial_path) != initial_control['local_initialization_sha256']):
        raise ValueError('Preserved original zero-residual local-query initialization is required')
    initial = torch.load(initial_path, map_location='cpu', weights_only=False)
    if initial['architecture'] != ARCHITECTURE or initial['optimizer_steps'] != 0:
        raise ValueError('The independent alternating candidate must preserve the original initialization')
    out = base / local_name(args.experiment_name); out.mkdir(exist_ok=True)
    with (out / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = {name:digest(ROOT / name) for name in SOURCES}
        settings = dict(read(base / 'full_local_query_geometry_grid_v4_round01/protocol.json')['settings'])
        original_budget = (settings['epochs'], settings['steps_per_view'])
        settings.update(stage='COMPLETE_TRAIN_LOCAL_RAW_XYZ_ALTERNATING_VIEW_GEOMETRY',
            sources=sources, epochs=args.epochs, steps_per_view=args.steps_per_view, seed=args.seed,
            registry_sha256=dataset.registry_sha256, registered_versions=len(dataset.versions),
            registered_families=len(dataset.families), local_initialization_sha256=digest(initial_path),
            ordering='Each update round shuffles every separate complete TRAIN view; original per-view seed variations retained',
            per_view_consecutive_updates=1, complete_cached_raw_xyz_query_inputs=True,
            original_complete_loss_retained=True,
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
            inherited_optimizer_steps=initial['inherited_optimizer_steps'])
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
                diagnostic_training_control_not_main_candidate=args.control_run))
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
                    original_complete_loss_retained=True, original_full_model_admission_rejected_control=True,
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
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--steps-per-view', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1031)
    parser.add_argument('--max-cache-gib', type=float, default=16.)
    parser.add_argument('--control-run', action='store_true')
    main(parser.parse_args())
