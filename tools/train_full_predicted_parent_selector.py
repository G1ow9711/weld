"""Exact full-reference set loss on frozen actual predicted ordered parents.

Every selected job retains all its TRAIN label views. Their reference utility
is evaluated separately and averaged; parent coordinates are never merged or
rewritten using a reference. A partial pool fit is not full-dataset training.
"""
import argparse
import gc
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.server_ordered_parent_set_model import OrderedParentSetModel, parent_batch
from weld_pipeline.server_predicted_parent_pool import decoded_subset
from weld_pipeline.server_reference_parent_selection import greedy_nullable_set
from weld_pipeline.server_offline_parent_set_reference import compile_reference_table, evaluate_reference_subset
from weld_pipeline.server_offline_parent_set_comparator import reference_comparator
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents
from weld_pipeline.sjg_local_runtime_904 import read, write, digest
from tools.prepare_full_predicted_parent_pool import load_gzip

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'
POOLS = BASE / 'full_predicted_parent_initial_v1'
DEST = BASE / 'full_predicted_parent_selector_initial_v1'


def pack_tokens(pool, proposals):
    rows = []
    counts = np.array([p['child_count'] for p in pool['parents']], np.int64)
    for start in range(0, len(counts), 256):
        indices = np.arange(start, min(start + 256, len(counts)))
        tokens, valid = parent_batch(pool, proposals, indices, 'cpu')
        rows.append(tokens.numpy()[valid.numpy()])
    flat = np.concatenate(rows) if rows else np.empty((0, 151), np.float32)
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
    assert len(flat) == offsets[-1]
    return dict(flat=flat, offsets=offsets, counts=counts)


def packed_batch(data, indices, device):
    counts = data['counts'][indices]
    length = int(counts.max())
    tokens = np.zeros((len(indices), length, 151), np.float32)
    valid = np.zeros((len(indices), length), bool)
    for row, i in enumerate(indices):
        a, b = data['offsets'][int(i):int(i) + 2]
        tokens[row, :b - a] = data['flat'][a:b]
        valid[row, :b - a] = True
    return torch.tensor(tokens, device=device), torch.tensor(valid, device=device)


def forward_scores(model, data):
    device = next(model.parameters()).device
    unary = []
    with torch.inference_mode():
        for start in range(0, len(data['counts']), 256):
            ids = np.arange(start, min(start + 256, len(data['counts'])))
            tokens, valid = packed_batch(data, ids, device)
            unary.append(model.parent_utility(tokens, valid).cpu().numpy())
        penalties = []
        for start in range(0, len(data['pairs']), 8192):
            stop = start + 8192
            penalties.append(model.pair_penalty(torch.tensor(data['pair_features'][start:stop], device=device),
                              torch.tensor(data['base_penalties'][start:stop], device=device, dtype=torch.float32),
                              torch.tensor(data['pair_scale'][start:stop], device=device, dtype=torch.float32)).cpu().numpy())
    return (np.concatenate(unary).astype(float) if unary else np.empty(0),
            np.concatenate(penalties).astype(float) if penalties else np.empty(0))


def utility_difference(model, data, predicted, teacher):
    device = next(model.parameters()).device
    coefficient = np.zeros(len(data['counts']), np.float32)
    coefficient[np.asarray(predicted, int)] += 1
    coefficient[np.asarray(teacher, int)] -= 1
    nodes = np.flatnonzero(coefficient)
    result = torch.zeros((), device=device)
    for start in range(0, len(nodes), 256):
        ids = nodes[start:start + 256]
        tokens, valid = packed_batch(data, ids, device)
        result = result + (model.parent_utility(tokens, valid)
                           * torch.tensor(coefficient[ids], device=device)).sum()
    pm, tm = np.zeros(len(coefficient), bool), np.zeros(len(coefficient), bool)
    pm[np.asarray(predicted, int)] = True
    tm[np.asarray(teacher, int)] = True
    a, b = data['pairs'].T
    weights = (tm[a] & tm[b]).astype(np.float32) - (pm[a] & pm[b]).astype(np.float32)
    edges = np.flatnonzero(weights)
    for start in range(0, len(edges), 8192):
        ids = edges[start:start + 8192]
        penalty = model.pair_penalty(torch.tensor(data['pair_features'][ids], device=device),
                     torch.tensor(data['base_penalties'][ids], device=device, dtype=torch.float32),
                     torch.tensor(data['pair_scale'][ids], device=device, dtype=torch.float32))
        result = result + (penalty * torch.tensor(weights[ids], device=device)).sum()
    return result


def load_job(dataset, job_key, pools=POOLS):
    folder = pools / 'pools' / job_key
    receipt = read(folder / 'receipt.json')
    if receipt['geometry_GT_reads'] != 0 or receipt['registry_sha256'] != dataset.registry_sha256:
        raise ValueError('Candidate pool was not frozen from approved input alone')
    if receipt['query_checkpoint_sha256'] != digest(pools / 'query_checkpoint.pt'):
        raise ValueError('Actual query checkpoint changed')
    for name, value in receipt['files'].items():
        if digest(folder / name) != value:
            raise ValueError('Actual candidate pool changed before reference read')
    pool = dict(parents=load_gzip(folder / 'parent_pool.json.gz'))
    with np.load(folder / 'child_inputs.npz', allow_pickle=False) as arrays:
        proposals = {name: arrays[name] for name in arrays.files}
    data = pack_tokens(pool, proposals)
    with np.load(folder / 'relations.npz', allow_pickle=False) as arrays:
        for name in ('pairs', 'pair_features', 'base_penalties'):
            data[name] = arrays[name]
    counts, pairs = data['counts'], data['pairs']
    data['pair_scale'] = 2 * np.minimum(counts[pairs[:, 0]], counts[pairs[:, 1]])
    data['pc'] = [p['input_children'] for p in pool['parents']]
    data['job_key'], data['pool'] = job_key, pool
    data['pool_receipt_sha256'] = digest(folder / 'receipt.json')
    # Only now can approved TRAIN references be opened.
    data['views'] = []
    decoded = decoded_subset(pool['parents'], np.arange(len(pool['parents'])))
    for version in dataset.jobs[job_key]['versions']:
        supervision = dataset.supervision(version['key'])
        target = supervision.target
        table, score = compile_reference_table(decoded, target)
        data['views'].append(dict(key=version['key'], table=table, G=target['original_children'],
                                  target=target, weight=version['family_equal_version_weight'],
                                  pool_capacity_reference_evaluation=score['categories']['overall']))
        print('ACTUAL_POOL_REFERENCE_CAPACITY', version['key'], score['categories']['overall'], flush=True)
    return data


def expected_utility(data, selected):
    mass = sum(view['weight'] for view in data['views'])
    return sum(view['weight'] / mass * evaluate_reference_subset(view['table'], selected)['reference_utility']
               / max(1, view['G']) for view in data['views'])


def main(jobs, steps, pool_name=POOLS.name, experiment_name=DEST.name):
    if steps < 1:
        raise ValueError('Positive recorded optimizer budget required')
    if any(not name or name in ('.', '..') or Path(name).name != name
           for name in (pool_name, experiment_name)):
        raise ValueError('Prediction pool and experiment names must be local directory names')
    pools, dest = BASE / pool_name, BASE / experiment_name
    torch.set_num_threads(2)
    torch.manual_seed(1031)
    dataset = FullShardDataset(ROOT)
    if not jobs:
        jobs = [key for key in dataset.jobs if (pools / 'pools' / key / 'receipt.json').exists()]
    if not jobs or not set(jobs) <= set(dataset.jobs):
        raise ValueError('No approved actual TRAIN prediction pool available')
    dest.mkdir(exist_ok=False)
    source = {name: digest(ROOT / 'weld_pipeline' / name) for name in
               ['server_ordered_parent_set_model.py', 'server_offline_parent_set_reference.py',
                'server_offline_parent_set_comparator.py', 'server_reference_parent_selection.py']}
    protocol = dict(status='ACTUAL_COMPLETE_PREDICTION_REFERENCE_SET_SELECTOR_TRAINING_PARTIAL_COVERAGE',
                    jobs=jobs, registered_jobs=len(dataset.jobs), registered_versions=len(dataset.versions),
                    steps=steps, seed=1031, learning_rate=.0003, source_sha256=digest(__file__), sources=source,
                    pool_name=pool_name, query_checkpoint_sha256=digest(pools / 'query_checkpoint.pt'),
                    loss='relu(expected normalized full D3 utility advantage + differentiable full set score difference)',
                    same_input_versions='Separate complete expected reference utility, no parent coordinate or child mixing',
                    physical_negative_labels=False, physical_parent_count_loss=False,
                    candidate_pool_frozen_before_reference_read=True, held_or_diagnostic_fit_calls=0,
                    original_run446_unchanged=True, full_dataset_fit_not_claimed=True)
    write(dest / 'protocol.json', protocol)
    state = torch.load(pools / 'query_checkpoint.pt', map_location='cpu', weights_only=False)
    parent_state = {name.removeprefix('parent_encoder.'): value for name, value in state['state_dict'].items()
                    if name.startswith('parent_encoder.')}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = OrderedParentSetModel(parent_state).to(device)
    data = [load_job(dataset, key, pools) for key in jobs]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    rng = np.random.default_rng(1031)
    history, actual_updates = [], 0
    order = []
    for step in range(steps):
        if step % len(data) == 0:
            order = rng.permutation(len(data)).tolist()
        row = data[order[step % len(data)]]
        model.eval()
        unary, penalties = forward_scores(model, row)
        predicted = greedy_nullable_set(unary, row['pairs'], penalties, row['pc'])
        teacher_candidates = [predicted]
        for view in row['views']:
            teacher, _ = reference_comparator(view['table'], row['pc'], unary)
            teacher_candidates.append(teacher)
        teacher = max(teacher_candidates, key=lambda selected: expected_utility(row, selected))
        advantage = expected_utility(row, teacher) - expected_utility(row, predicted)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if advantage > 0:
            # Scores are normalized by the same expected full reference G;
            # exact matching choices are detached, the score difference is not.
            normalizer = sum(v['weight'] / max(1, v['G']) for v in row['views']) / sum(v['weight'] for v in row['views'])
            job_mass = sum(v['weight'] for v in row['views'])
            loss = torch.relu(utility_difference(model, row, predicted, teacher) * normalizer + advantage)
            # Uniform job visits recover registered family/version objective
            # mass across a complete pool pass. Partial controls keep that mass.
            loss = loss * job_mass * len(data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            actual_updates += 1
        else:
            loss = torch.zeros((), device=device)
        record = dict(step=step + 1, job_key=row['job_key'], loss=float(loss.detach()),
                      predicted_parents=len(predicted), comparator_parents=len(teacher),
                      expected_prediction_utility=expected_utility(row, predicted),
                      expected_comparator_utility=expected_utility(row, teacher), parameter_updates=actual_updates)
        history.append(record)
        write(dest / 'progress.json', record)
        print('FULL_REFERENCE_ORDERED_SET_FIT', record, flush=True)
    checkpoint = dest / 'model.pt'
    torch.save(dict(state_dict={name: value.detach().cpu() for name, value in model.state_dict().items()},
                    protocol_sha256=digest(dest / 'protocol.json')), checkpoint)
    evaluations = []
    model.eval()
    for row in data:
        unary, penalty = forward_scores(model, row)
        selected = greedy_nullable_set(unary, row['pairs'], penalty, row['pc'])
        freeze = dict(selected_parent_indices=selected, checkpoint_sha256=digest(checkpoint),
                      pool_receipt_sha256=row['pool_receipt_sha256'], model_scores_from_reference=False)
        write(dest / (row['job_key'] + '_prediction_freeze.json'), freeze)
        np.savez_compressed(dest / (row['job_key'] + '_scores.npz'), unary=unary, penalty=penalty)
        for view in row['views']:
            score = evaluate_reference_subset(view['table'], selected)
            direct = score_native_parents(decoded_subset(row['pool']['parents'], selected), view['target'],
                                           [False] * len(view['target']['parents']))['categories']['overall']
            if direct['TP'] != score['TP'] or direct['FP'] != score['FP']:
                raise ValueError('Selected complete prediction table differs from original direct D3')
            evaluations.append(dict(version_key=view['key'], G=view['G'], **score))
    write(dest / 'receipt.json', dict(status='PARTIAL_TRAIN_POOL_SET_FIT_NOT_PROMOTED' if actual_updates else
          'REFERENCE_POOL_CONTROL_NO_PARAMETER_UPDATES_NOT_PROMOTED',
          jobs=jobs, processed_reference_versions=sum(len(row['views']) for row in data),
          registered_training_versions=len(dataset.versions), steps=steps, parameter_updates=actual_updates,
          checkpoint_sha256=digest(checkpoint), protocol_sha256=digest(dest / 'protocol.json'),
          history=history, same_training_pool_evaluations=evaluations, independent_model_evaluations=0,
          full_dataset_training_remaining=True, held_or_diagnostic_fit_calls=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', action='append', default=[])
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--pool-name', default=POOLS.name)
    parser.add_argument('--experiment-name', default=DEST.name)
    args = parser.parse_args()
    main(args.job, args.steps, args.pool_name, args.experiment_name)
