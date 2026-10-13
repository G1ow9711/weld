"""Freeze a complete native+visual input pool before any reference is opened.

Output is compatible with the existing nullable parent-set training consumer.
A partial-checkpoint control is explicit and cannot claim full TRAIN fitting.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from tools.export_full_dataset_split_view import local_name
from tools.probe_multiview_visual_assistance import load_xyz
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.server_predicted_parent_pool import FrozenRun446Priors
from weld_pipeline.server_supported_directed_parent_pool import build_supported_parent_pool
from weld_pipeline.server_supported_interval_prediction_runtime import SupportedIntervalPredictionRuntime
from weld_pipeline.server_supported_interval_weld_model import SupportedIntervalWeldModel, ARCHITECTURE
from weld_pipeline.server_visual_augmented_input import VisualAugmentedInput
from weld_pipeline.sjg_local_runtime_904 import ROOT, read, write, digest

BASE = ROOT / 'server_rebuild/full_dataset_v2'
VISUAL = BASE / 'full_171_visual_assistance_and_candidates_20261013_v1'


def save_gzip(path, value):
    with gzip.open(path, 'wt', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def run(args):
    torch.set_num_threads(2); dataset = PartitionedFullShardDataset(ROOT, BASE)
    key = args.job_key or min(dataset.jobs, key=lambda k: dataset.jobs[k]['original_points'])
    if key not in dataset.jobs: raise ValueError('Registered TRAIN input required; no held reference route here')
    out = BASE / local_name(args.experiment_name); out.mkdir(exist_ok=True)
    folder = out / 'pools' / key; folder.mkdir(parents=True, exist_ok=False)
    source = (ROOT / args.checkpoint).resolve(strict=True)
    if digest(source) != args.checkpoint_sha256: raise ValueError('Frozen checkpoint fingerprint changed')
    checkpoint = out / 'query_checkpoint.pt'
    if not checkpoint.exists(): shutil.copyfile(source, checkpoint)
    if digest(checkpoint) != args.checkpoint_sha256: raise ValueError('Output pool has a different frozen model')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    training = read(source.parent / 'protocol.json')
    if (state['architecture'] != ARCHITECTURE or state['protocol_sha256'] != digest(source.parent / 'protocol.json')
            or training['registry_sha256'] != dataset.registry_sha256):
        raise ValueError('Wrong model architecture, split or training protocol')
    for name, sha in training['sources'].items():
        if digest(ROOT / name) != sha: raise ValueError('Training/runtime source mismatch: ' + name)
    first_pass = {v.split(':', 1)[1] for v in state['completed'] if v.startswith('0:')}
    complete_first_pass = first_pass == set(dataset.versions)
    if not complete_first_pass and not args.partial_training_control:
        raise ValueError('Complete first TRAIN pass required unless explicitly running a partial training control')
    files = ['tools/prepare_supported_interval_parent_pool.py',
        'weld_pipeline/server_supported_interval_prediction_runtime.py',
        'weld_pipeline/server_supported_interval_weld_model.py',
        'weld_pipeline/server_supported_directed_parent_pool.py']
    protocol = dict(sources={p: digest(ROOT / p) for p in files}, registry_sha256=dataset.registry_sha256,
        query_checkpoint_sha256=args.checkpoint_sha256, architecture=ARCHITECTURE,
        complete_first_TRAIN_pass=complete_first_pass, partial_training_control=args.partial_training_control,
        reference_probability=args.reference_probability, maximum_children=args.maximum_children,
        max_parents=args.maximum_parents, device=args.device, query_batch=16,
        all_profile_one_mm_queries=True, all_top8_cartesian_intervals_scored=True,
        retained_whole_pair_search_beam=getattr(args, 'whole_pair_beam', 64),
        geometry_GT_reads=0, STEP_reads=0, nullable_parent_selection_not_yet_applied=True)
    if (out / 'protocol.json').exists() and read(out / 'protocol.json') != protocol:
        raise ValueError('Frozen output protocol changed')
    write(out / 'protocol.json', protocol)
    raw, _ = load_xyz(dataset.jobs[key]); original = dataset.input_case(key)
    row = next(r for r in read(VISUAL / 'receipt.json')['records'] if r['job_key'] == key)
    addition_path = VISUAL / row['path']
    if digest(addition_path) != row['sha256']: raise ValueError('Frozen visual input changed')
    addition = read(addition_path); input_sha = digest(dataset.shards / key / 'receipt.json')
    if (addition['original_input_receipt_sha256'] != input_sha
            or addition['original_profiles_sha256'] != digest(dataset.shards / key / 'profiles.json')):
        raise ValueError('Native/visual input binding changed')
    inp = VisualAugmentedInput(original, addition)
    model = SupportedIntervalWeldModel().to(args.device).eval(); model.load_state_dict(state['state_dict'], strict=True)
    priors = FrozenRun446Priors(ROOT)
    labels = [str((ROOT / v['label_directory']).resolve()) for _, v in dataset.versions.values()]
    denied = []
    def audit(event, values):
        if event == 'open' and isinstance(values[0], (str, bytes, Path)):
            name = str(Path(values[0]).resolve())
            if any(name == p or name.startswith(p + '/') for p in labels):
                denied.append(name); raise AssertionError('Input prediction attempted to read reference geometry')
    sys.addaudithook(audit)
    started = time.time()
    def progress(visited, total, counts, children):
        write(folder / 'progress.json', dict(status='PREDICTING_ALL_INPUT_PROFILES', pid=os.getpid(),
            profiles_visited=visited, total_profiles=total, children=children, counts=counts, unix_time=time.time()))
        if visited % 128 == 0: print('COMPLETE_INTERVAL_PROFILE', key, visited, total, counts['queries_scored'], children, flush=True)
    try:
        runtime = SupportedIntervalPredictionRuntime(model, state['mean'], state['std'],
            reference_probability=args.reference_probability, maximum_children=args.maximum_children,
            whole_pair_beam=getattr(args, 'whole_pair_beam', 64))
        with threadpool_limits(limits=2): proposals = runtime.collect(inp, raw, input_sha, progress)
        save_gzip(folder / 'child_proposals.json.gz', proposals['children'])
        np.savez_compressed(folder / 'child_inputs.npz', **{k: proposals[k] for k in ['child_tokens', 'query_xyz', 'query_bases']})
        save_gzip(folder / 'shared_events.json.gz', proposals['shared_events'])
        write(folder / 'child_freeze.json', dict(audit=proposals['audit'], geometry_GT_reads=0,
            files={n:digest(folder/n) for n in ['child_proposals.json.gz','child_inputs.npz','shared_events.json.gz']}))
        with threadpool_limits(limits=2):
            pool = build_supported_parent_pool(proposals['children'], proposals['child_tokens'][:, :128],
                priors, model.directed_link, proposals['shared_events'], max_parents=args.maximum_parents)
        save_gzip(folder / 'parent_pool.json.gz', pool['parents'])
        np.savez_compressed(folder / 'learned_links.npz', **{k:pool[k] for k in
            ['link_pairs','link_features','link_prior_probabilities','learned_link_logits','link_legal']})
        write(folder / 'input_audit.json', dict(query=proposals['audit'], parent=pool['audit']))
        names = ['child_proposals.json.gz','child_inputs.npz','shared_events.json.gz','child_freeze.json',
                 'parent_pool.json.gz','learned_links.npz','input_audit.json']
        write(folder / 'receipt.json', dict(status='ACTUAL_PREDICTION_PARENT_POOL_FROZEN_BEFORE_TRAIN_REFERENCE_READ',
            job_key=key, registry_sha256=dataset.registry_sha256, input_receipt_sha256=input_sha,
            query_checkpoint_sha256=args.checkpoint_sha256, query_checkpoint_optimizer_steps=state['optimizer_steps'],
            prior_run446_sha256=priors.checkpoint_sha256, geometry_GT_reads=0, STEP_reads=0,
            competition_mode='lazy', competition_neighbors=16, max_active_competition_pairs=4_000_000,
            children=len(proposals['children']), parents=len(pool['parents']),
            queries_scored=proposals['audit']['counts']['queries_scored'],
            complete_first_TRAIN_pass=complete_first_pass, partial_training_control=args.partial_training_control,
            files={n:digest(folder/n) for n in names}, reference_file_open_attempts=denied,
            seconds=time.time()-started, complete_output_selection_pending=True, full_contract_pass=False))
        print('COMPLETE_SUPPORTED_INTERVAL_POOL_FROZEN', key, len(proposals['children']), len(pool['parents']), flush=True)
    except BaseException as exc:
        write(folder / 'failure.json', dict(error=str(exc), traceback=traceback.format_exc(),
            reference_file_open_attempts=denied, complete_output_claimed=False))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--experiment-name', required=True)
    p.add_argument('--checkpoint', required=True); p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--job-key'); p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--partial-training-control', action='store_true')
    p.add_argument('--whole-pair-beam', type=int, default=64)
    p.add_argument('--reference-probability', type=float, default=.5)
    p.add_argument('--maximum-children', type=int, default=500_000); p.add_argument('--maximum-parents', type=int, default=500_000)
    run(p.parse_args())
