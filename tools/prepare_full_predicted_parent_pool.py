"""Freeze actual complete input predictions before consulting TRAIN references.

One job entry is useful for runtime controls; --all-ready processes every ready
registered training cloud. No fixed pilot list is built into this consumer.
Prepared pools use an immutable checkpoint snapshot and genuine run446 priors.
"""
import argparse
import gc
import gzip
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.server_full_weld_model import FullWeldModel
from weld_pipeline.server_legacy_anchored_full_weld_model import LegacyAnchoredFullWeldModel
from weld_pipeline.server_runtime_aligned_training_samples import FIXED_QUERY_FRACTIONS
from weld_pipeline.server_full_prediction_runtime import CompleteQueryProposals
from weld_pipeline.server_predicted_parent_pool import FrozenRun446Priors, build_parent_pool, parent_competition_pairs
from weld_pipeline.sjg_local_runtime_904 import read, write, digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'
NAME = 'full_predicted_parent_initial_v1'


def save_gzip(path, value):
    with gzip.open(path, 'wt', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def load_gzip(path):
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        return json.load(stream)


def initialize(name=NAME, model_kind='free_v1'):
    if (Path(name).name != name or not name or name in ('.', '..')
            or model_kind not in ('free_v1', 'anchored_v2')):
        raise ValueError('A local experiment name and declared model architecture are required')
    out = BASE / name
    out.mkdir(exist_ok=True)
    storage_link = out / 'pools'
    if not storage_link.exists():
        temporary = Path(tempfile.mkdtemp(prefix='weld-predicted-parent-pools-', dir='/tmp'))
        storage_link.symlink_to(temporary, target_is_directory=True)
        write(out / 'storage_binding.json', dict(project_link=str(storage_link), temporary_directory=str(temporary),
              scope='Rebuildable compact predictions and real input tensors only; immutable checkpoint remains persistent'))
    storage = storage_link.resolve()
    if storage.parent != Path('/tmp') or not storage.name.startswith('weld-predicted-parent-pools-'):
        raise ValueError('Unexpected task-owned prediction cache')
    checkpoint = out / 'query_checkpoint.pt'
    if not checkpoint.exists():
        staging = out / 'query_checkpoint.incomplete'
        training_name = 'full_weld_anchored_v2' if model_kind == 'anchored_v2' else 'full_weld_warmstart_v1'
        shutil.copyfile(BASE / training_name / 'last.pt', staging)
        # The source writer replaces its checkpoint atomically. A copied inode
        # is a coherent snapshot, even if subsequent warm-start steps continue.
        state = torch.load(staging, map_location='cpu', weights_only=False)
        if state['optimizer_steps'] <= 0:
            raise ValueError('No fitted complete-reference query head available')
        os.replace(staging, checkpoint)
    return out, storage, checkpoint


def prepare(dataset, job_key, model, state, priors, output, query_mode,
            fixed_query_fractions=(.125, .375, .625, .875)):
    destination = output / 'pools' / job_key
    if destination.exists():
        receipt = read(destination / 'receipt.json')
        if receipt['query_mode'] != query_mode or receipt['query_checkpoint_sha256'] != digest(output / 'query_checkpoint.pt'):
            raise ValueError('Existing actual prediction recipe differs; use a new declared experiment')
        for name, value in receipt['files'].items():
            if digest(destination / name) != value:
                raise ValueError('Frozen actual prediction artifact changed')
        return receipt
    if shutil.disk_usage(output / 'pools').free < 5 * 1024**3:
        raise RuntimeError('Actual prediction cache lacks five GiB temporary headroom')
    start = time.perf_counter()
    inp = dataset.input_case(job_key)
    generator = CompleteQueryProposals(model, state['mean'], state['std'],
                                      device='cuda' if torch.cuda.is_available() else 'cpu', mode=query_mode,
                                      fixed_query_fractions=fixed_query_fractions)
    def progress(pid, profiles, queries, candidates):
        write(output / 'progress.json', dict(stage='ACTUAL_INPUT_QUERY_PREDICTIONS', job_key=job_key,
              pid=os.getpid(), profile=pid, profiles=profiles, queries=queries, candidates=candidates,
              GT_geometry_reads=0, unix_time=time.time()))
        print('ACTUAL_FULL_PROFILE', job_key, pid, '/', profiles, 'queries', queries, 'children', candidates, flush=True)
    with threadpool_limits(limits=2):
        proposals = generator.build(inp, progress)
        write(output / 'progress.json', dict(stage='PREDICTED_PARENT_CHAINS', job_key=job_key,
              children=len(proposals['children']), GT_geometry_reads=0, unix_time=time.time()))
        print('ACTUAL_CHILD_POOL_READY', job_key, len(proposals['children']), flush=True)
        pool = build_parent_pool(proposals['children'], priors)
        pairs = parent_competition_pairs(pool['parents'])
        print('ACTUAL_PARENT_POOL_READY', job_key, len(pool['parents']), 'competition', len(pairs), flush=True)
        pair_x, duplicate = priors.duplicate_probabilities(pool['parents'], pairs)
    counts = np.array([p['child_count'] for p in pool['parents']], np.int64)
    penalties = duplicate * 2 * np.minimum(counts[pairs[:, 0]], counts[pairs[:, 1]])
    with tempfile.TemporaryDirectory(prefix=job_key + '-', dir=output / 'pools') as temporary:
        stage = Path(temporary) / 'result'
        stage.mkdir()
        save_gzip(stage / 'child_proposals.json.gz', proposals['children'])
        save_gzip(stage / 'parent_pool.json.gz', pool['parents'])
        np.savez_compressed(stage / 'child_inputs.npz', child_tokens=proposals['child_tokens'],
                            query_xyz=proposals['query_xyz'], query_bases=proposals['query_bases'])
        np.savez_compressed(stage / 'relations.npz', pairs=pairs, pair_features=pair_x,
                            duplicate_probabilities=duplicate, base_penalties=penalties,
                            link_pairs=pool['link_pairs'], link_features=pool['link_features'],
                            link_probabilities=pool['link_probabilities'])
        write(stage / 'input_audit.json', dict(query=proposals['audit'], parent=pool['audit']))
        receipt = dict(status='ACTUAL_PREDICTION_PARENT_POOL_FROZEN_BEFORE_TRAIN_REFERENCE_READ',
                       job_key=job_key, registry_sha256=dataset.registry_sha256,
                       input_receipt_sha256=digest(dataset.shards / job_key / 'receipt.json'),
                       query_checkpoint_sha256=digest(output / 'query_checkpoint.pt'),
                       query_checkpoint_optimizer_steps=state['optimizer_steps'],
                       prior_run446_sha256=priors.checkpoint_sha256, query_mode=query_mode,
                       fixed_query_fractions=list(fixed_query_fractions),
                       original_input_profiles=len(inp.profiles), children=len(proposals['children']),
                       parents=len(pool['parents']), geometry_GT_reads=0, STEP_reads=0, seconds=time.perf_counter() - start,
                       files={p.name: digest(p) for p in stage.iterdir()})
        write(stage / 'receipt.json', receipt)
        stage.rename(destination)
    print('ACTUAL_FULL_PARENT_POOL_FROZEN', job_key, receipt['parents'], 'seconds', receipt['seconds'], flush=True)
    del inp, proposals, pool
    gc.collect()
    return receipt


def main(job, all_ready, query_mode, model_kind='free_v1', experiment_name=None):
    if not job and not all_ready:
        raise ValueError('Choose a registered job or all ready registered training inputs')
    torch.set_num_threads(2)
    dataset = FullShardDataset(ROOT)
    name = experiment_name or (NAME if model_kind == 'free_v1' else 'full_predicted_parent_anchored_initial_v2')
    output, _, checkpoint = initialize(name, model_kind)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    anchored_state = 'legacy.trunk.0.weight' in state['state_dict']
    if anchored_state != (model_kind == 'anchored_v2'):
        raise ValueError('Frozen checkpoint architecture differs from declared runtime')
    model = LegacyAnchoredFullWeldModel() if anchored_state else FullWeldModel()
    model.load_state_dict(state['state_dict'])
    fractions = FIXED_QUERY_FRACTIONS if anchored_state else (.125, .375, .625, .875)
    priors = FrozenRun446Priors(ROOT)
    protocol = dict(query_checkpoint_sha256=digest(checkpoint), optimizer_steps=state['optimizer_steps'],
                    registry_sha256=dataset.registry_sha256, original_run446_prior_sha256=priors.checkpoint_sha256,
                    query_mode=query_mode, model_kind=model_kind, fixed_query_fractions=list(fractions),
                    all_profiles_visited=True,
                    GT_geometry_reads=0, original_final_validity_head_526_columns_not_faked=True,
                    reference_selector_training_and_full_evaluation_remaining=True,
                    source_sha256=digest(__file__), sources={name: digest(ROOT / 'weld_pipeline' / name)
                    for name in ['server_full_prediction_runtime.py', 'server_predicted_parent_pool.py',
                                 'server_ordered_parent_set_model.py', 'server_full_weld_model.py',
                                 'server_legacy_anchored_full_weld_model.py',
                                 'server_runtime_aligned_training_samples.py']})
    if (output / 'protocol.json').exists():
        if read(output / 'protocol.json') != protocol:
            raise ValueError('Actual prediction protocol changed')
    else:
        write(output / 'protocol.json', protocol)
    jobs = [job] if job else dataset.ready_jobs()
    for key in jobs:
        prepare(dataset, key, model, state, priors, output, query_mode, fractions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job')
    parser.add_argument('--all-ready', action='store_true')
    parser.add_argument('--model-kind', choices=['free_v1', 'anchored_v2'], default='free_v1')
    parser.add_argument('--experiment-name')
    parser.add_argument('--query-mode', choices=['fragment4', 'grid', 'fixed_and_grid'], default='fixed_and_grid')
    args = parser.parse_args()
    main(args.job, args.all_ready, args.query_mode, args.model_kind, args.experiment_name)
