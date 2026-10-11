"""Complete ordered-parent prediction without a reference-reading interface.

The batching and score arithmetic match the existing full TRAIN consumer.
This module loads only a frozen input pool, fitted weights and genuine run446
relation priors. References and the D3 scorer belong to a separate caller.
"""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .server_lazy_parent_competition import LazyParentCompetition
from .server_ordered_parent_set_model import OrderedParentSetModel, parent_batch
from .server_predicted_parent_pool import FrozenRun446Priors, decoded_subset
from .server_reference_parent_selection import greedy_nullable_set
from .sjg_local_runtime_904 import digest, read


def load_gzip(path):
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        return json.load(stream)


def pack_tokens(pool, proposals):
    rows = []
    counts = np.array([p['child_count'] for p in pool['parents']], np.int64)
    for start in range(0, len(counts), 256):
        indices = np.arange(start, min(start + 256, len(counts)))
        tokens, valid = parent_batch(pool, proposals, indices, 'cpu')
        rows.append(tokens.numpy()[valid.numpy()])
    flat = np.concatenate(rows) if rows else np.empty((0, 151), np.float32)
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
    if len(flat) != offsets[-1] or not np.isfinite(flat).all():
        raise ValueError('Complete ordered-parent tokens must be finite and untruncated')
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


def load_input_pool(dataset, job_key, pools, priors=None):
    """Authenticate actual input artifacts; never call a dataset reference API."""
    pools = Path(pools)
    if job_key not in dataset.jobs or Path(job_key).name != job_key:
        raise ValueError('An explicitly registered input job is required')
    folder = pools / 'pools' / job_key
    receipt = read(folder / 'receipt.json')
    if (receipt['job_key'] != job_key or receipt['geometry_GT_reads'] != 0
            or receipt['STEP_reads'] != 0 or receipt['registry_sha256'] != dataset.registry_sha256):
        raise ValueError('Pool is not bound to this input-only registration')
    if receipt['input_receipt_sha256'] != digest(dataset.shards / job_key / 'receipt.json'):
        raise ValueError('Original complete input receipt changed')
    if receipt['query_checkpoint_sha256'] != digest(pools / 'query_checkpoint.pt'):
        raise ValueError('Frozen query checkpoint changed')
    for name, sha in receipt['files'].items():
        if Path(name).name != name or digest(folder / name) != sha:
            raise ValueError('Frozen complete prediction artifact changed')
    pool = dict(parents=load_gzip(folder / 'parent_pool.json.gz'))
    with np.load(folder / 'child_inputs.npz', allow_pickle=False) as arrays:
        proposals = {name: arrays[name] for name in arrays.files}
    data = pack_tokens(pool, proposals)
    if len(pool['parents']) != receipt['parents'] or len(proposals['child_tokens']) != receipt['children']:
        raise ValueError('Complete pool counts disagree with the frozen receipt')
    if receipt.get('competition_mode', 'eager') == 'lazy':
        priors = priors or FrozenRun446Priors(dataset.root)
        if priors.checkpoint_sha256 != receipt['prior_run446_sha256']:
            raise ValueError('Genuine frozen relation prior changed')
        data['lazy_competition'] = LazyParentCompetition(pool['parents'], priors,
            neighbors=receipt['competition_neighbors'],
            max_active_pairs=receipt.get('max_active_competition_pairs', 4_000_000))
        data.update(data['lazy_competition'].relation_arrays([]))
    elif receipt.get('competition_mode', 'eager') == 'eager':
        with np.load(folder / 'relations.npz', allow_pickle=False) as arrays:
            for name in ('pairs', 'pair_features', 'base_penalties'):
                data[name] = arrays[name]
    else:
        raise ValueError('Unknown frozen competition recipe')
    pairs, counts = data['pairs'], data['counts']
    data['pair_scale'] = 2 * np.minimum(counts[pairs[:, 0]], counts[pairs[:, 1]])
    data['pc'] = [p['input_children'] for p in pool['parents']]
    data.update(job_key=job_key, pool=pool, receipt=receipt,
                pool_receipt_sha256=digest(folder / 'receipt.json'))
    return data


def load_selector(checkpoint, sha256, protocol, query_sha256, device='cpu'):
    """Reject a selector trained against another query checkpoint/token space."""
    checkpoint, protocol = Path(checkpoint), Path(protocol)
    if digest(checkpoint) != sha256:
        raise ValueError('Declared selector checkpoint fingerprint differs')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    training = read(protocol)
    if (state['protocol_sha256'] != digest(protocol)
            or training['query_checkpoint_sha256'] != query_sha256):
        raise ValueError('Selector and query are not a bound fitted model pair')
    if training.get('held_or_diagnostic_fit_calls', 0) != 0:
        raise ValueError('Held/diagnostic fitting is forbidden')
    model = OrderedParentSetModel().to(device)
    model.load_state_dict(state['state_dict'], strict=True)
    model.eval()
    return model, state, training


def forward_scores(model, data):
    device = next(model.parameters()).device
    unary = []
    if model.training:
        raise ValueError('Complete prediction requires evaluation mode')
    with torch.inference_mode():
        for start in range(0, len(data['counts']), 256):
            ids = np.arange(start, min(start + 256, len(data['counts'])))
            tokens, valid = packed_batch(data, ids, device)
            unary.append(model.parent_utility(tokens, valid).cpu().numpy())
        unary = np.concatenate(unary).astype(float) if unary else np.empty(0)
        if 'lazy_competition' in data:
            data.update(data['lazy_competition'].relation_arrays(np.flatnonzero(unary > 0)))
        penalties = []
        for start in range(0, len(data['pairs']), 8192):
            stop = start + 8192
            penalties.append(model.pair_penalty(
                torch.tensor(data['pair_features'][start:stop], device=device),
                torch.tensor(data['base_penalties'][start:stop], device=device, dtype=torch.float32),
                torch.tensor(data['pair_scale'][start:stop], device=device, dtype=torch.float32)).cpu().numpy())
    penalty = np.concatenate(penalties).astype(float) if penalties else np.empty(0)
    if not np.isfinite(unary).all() or not np.isfinite(penalty).all():
        raise ValueError('Complete prediction scores must be finite')
    return unary, penalty


def predict(model, data):
    unary, penalty = forward_scores(model, data)
    selected = list(map(int, greedy_nullable_set(unary, data['pairs'], penalty, data['pc'])))
    parents = [data['pool']['parents'][i] for i in selected]
    freeze = dict(selected_parent_indices=selected, runtime_uses_reference_geometry=False,
        query_pool_receipt_sha256=data['pool_receipt_sha256'],
        unary_sha256=hashlib.sha256(unary.tobytes()).hexdigest(),
        pairs_sha256=hashlib.sha256(data['pairs'].tobytes()).hexdigest(),
        penalty_sha256=hashlib.sha256(penalty.tobytes()).hexdigest(),
        pool_parents=len(data['counts']), selected_parents=len(selected),
        total_prediction_children=sum(p['child_count'] for p in parents),
        reference_API_calls=0, STEP_reads=0, full_contract_pass=False)
    return parents, decoded_subset(data['pool']['parents'], selected), freeze
