"""Whole-carrier sequence pilot with occupancy and explicit internal break heads.

TRAIN labels alone define normalization, losses and targets. Predictions remain
unassembled frontend proposals; they do not establish final contract acceptance.
"""
import argparse
import gc
import json
import math
import pickle
import time
import traceback
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from weld_pipeline import sjg_local_interval_pilot_1238 as prior
from weld_pipeline.sjg_finite_extent_sequence_1240 import SCHEMA
from weld_pipeline.sjg_local_runtime_904 import OUT, read, write, digest, check_resources, reclaim_own_idle_pages
from weld_pipeline.sjg_whole_joint_targets_1229 import TRAIN
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write

prior.INPUT_RUN = 'run390'
prior.EXPECTED_SCHEMA = SCHEMA
EPOCHS = 20
BATCH = 8


def pointwise(layer, x):
    """Exact 1x1 convolution algebra via GEMM, without cuDNN workspaces."""
    return F.linear(x.transpose(1, 2), layer.weight[:, :, 0], layer.bias).transpose(1, 2)


class SymmetricBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.dilation = dilation
        self.center = nn.Conv1d(width, width, 1)
        self.neighbor = nn.Conv1d(width, width, 1, bias=False)

    def forward(self, x, lengths, closed, mask):
        position = torch.arange(x.shape[2], device=x.device)[None, :]
        near = torch.zeros_like(x)
        for sign in (-1, 1):
            j = position + sign * self.dilation
            valid = (j >= 0) & (j < lengths[:, None])
            j = torch.where(closed[:, None], j.remainder(lengths[:, None]), j.clamp(min=0))
            j = torch.minimum(j, lengths[:, None] - 1)
            neighbor = torch.gather(x, 2, j[:, None, :].expand(-1, x.shape[1], -1))
            near = near + neighbor * (valid | closed[:, None])[:, None, :]
        return (x + .2 * F.silu(pointwise(self.center, x) + pointwise(self.neighbor, near * .5))) * mask


class SequenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Conv1d(56, 32, 1)
        self.context = nn.Conv1d(64, 32, 1)
        self.blocks = nn.ModuleList([SymmetricBlock(32, 2**i) for i in range(10)])
        self.occupancy = nn.Conv1d(32, 1, 1)
        self.boundary = nn.Conv1d(32, 1, 1)

    def forward(self, x, lengths, closed):
        mask = (torch.arange(x.shape[2], device=x.device)[None, :] < lengths[:, None])[:, None, :]
        h = F.silu(pointwise(self.input, x)) * mask
        mean = h.sum(2, keepdim=True) / lengths[:, None, None]
        maximum = h.masked_fill(~mask, -torch.inf).amax(2, keepdim=True)
        h = (h + F.silu(pointwise(self.context, torch.cat((mean, maximum), 1)))) * mask
        for block in self.blocks:
            h = block(h, lengths, closed, mask)
        return pointwise(self.occupancy, h)[:, 0], pointwise(self.boundary, (h[:, :, 1:] + h[:, :, :-1]) * .5)[:, 0]


def prepare(dest):
    from weld_pipeline.sjg_supervision_dependency_guard_1269 import require_clean_occupancy_supervision
    # This legacy entry point reads exactly the rejected run380 supervision.
    require_clean_occupancy_supervision(['run392'])
    records = []
    audits = []
    sums = np.zeros(56)
    squares = np.zeros(56)
    total = 0
    for key in TRAIN:
        source, m, receipt = prior.load_inputs(key)
        folder = OUT / f'run380/source_{key}'
        sr = read(folder / 'joint_supervision_receipt.json')
        assert digest(folder / 'joint_count_targets.npz') == sr['labels_sha256']
        assert digest(folder / 'joint_interval_instances.json') == sr['instances_sha256']
        with np.load(folder / 'joint_count_targets.npz') as z:
            counts = z['labels']
        events = defaultdict(list)
        owners, covered = set(), set()
        for event in read(folder / 'joint_interval_instances.json')['instances']:
            owners.add(tuple(event['owner']))
            if event['count_is_known']:
                events[event['row']].append(event)
        lookup = {int(p): i for i, p in enumerate(m['profile_ids'])}
        begin = len(records)
        for row in m['admitted_rows']:
            row = int(row)
            if counts[row] < 0:
                continue
            j = lookup[row // 2]
            a, b = m['offsets'][j:j+2]
            q = m['q_mm'][a:b]
            x = prior.ordered_features(source[a:b], row % 2).copy()
            y = np.zeros(len(q), np.float32)
            weight = np.zeros(len(q), np.float32)
            boundary = np.zeros(len(q)-1, np.float32)
            edge = (q[:-1] + q[1:]) * .5
            event_weight = 0.
            for event in events[row]:
                lo, hi = np.array(event['ratio']) * event['length_mm']
                positive = (q >= lo) & (q <= hi)
                if positive.any():
                    y[positive] = 1
                    weight[positive] += event['weight'] / int(positive.sum())
                    covered.add(tuple(event['owner']))
                    event_weight += event['weight']
                for endpoint in (lo, hi):
                    # Domain endpoints need no artificial internal cut.
                    if endpoint > q[0] + 1e-6 and endpoint < q[-1] - 1e-6:
                        boundary[np.argmin(np.abs(edge-endpoint))] = 1
            records.append(dict(source=key, row=row, x=x, y=y, w=weight, boundary=boundary,
                                closed=bool(x[0, 27] > .5), event_weight=event_weight))
            sums += x.sum(0, dtype=np.float64)
            squares += np.square(x.astype(np.float64)).sum(0)
            total += len(x)
        group = records[begin:]
        positive_mass = sum(float(r['w'].sum()) for r in group)
        negative_count = sum(int((r['y'] == 0).sum()) for r in group)
        total_event_weight = sum(r['event_weight'] for r in group)
        bpos = sum(float(r['boundary'].sum()) for r in group)
        bneg = sum(int((r['boundary'] == 0).sum()) for r in group)
        for r in group:
            pos = r['y'] > 0
            r['w'][pos] /= (len(TRAIN) * 2 * positive_mass)
            r['w'][~pos] = 1 / (len(TRAIN) * 2 * negative_count) if negative_count else 0
            r['dice_weight'] = r['event_weight'] / (len(TRAIN) * total_event_weight)
            r['bw'] = np.where(r['boundary'] > 0,
                               1 / (len(TRAIN)*2*max(bpos, 1)),
                               1 / (len(TRAIN)*2*max(bneg, 1))).astype(np.float32)
        audit = dict(source=key, known_sequences=len(group), original_children=len(owners),
                     covered_children=len(covered), missing=sorted(owners-covered),
                     boundary_sites=int(bpos), feature_sha256=receipt['features_sha256'])
        audits.append(audit)
        source._mmap.close()
        print('SEQUENCE_SUPERVISION', audit, flush=True)
    assert not any(a['missing'] for a in audits)
    mean = sums / total
    scale = np.sqrt(np.maximum(squares / total - mean*mean, 0))
    scale[scale < 1e-6] = 1
    for r in records:
        r['x'] -= mean.astype(np.float32)
        r['x'] /= scale.astype(np.float32)
        assert np.isfinite(r['x']).all()
    write(dest / 'supervision_receipt.json', dict(cases=audits, G=sum(a['original_children'] for a in audits),
        sequences=len(records), point_rows=total, DEV_reads=0, STEP_reads=0, labels_changed=False))
    return records, mean.astype(np.float32), scale.astype(np.float32)


def batch(records, device):
    size = max(len(r['y']) for r in records)
    n = len(records)
    arrays = dict(x=np.zeros((n, 56, size), np.float32), y=np.zeros((n, size), np.float32),
                  w=np.zeros((n, size), np.float32), boundary=np.zeros((n, size-1), np.float32),
                  bw=np.zeros((n, size-1), np.float32))
    for i, r in enumerate(records):
        length = len(r['y'])
        arrays['x'][i, :, :length] = r['x'].T
        for name in ('y', 'w'):
            arrays[name][i, :length] = r[name]
        for name in ('boundary', 'bw'):
            arrays[name][i, :length-1] = r[name]
    result = {k: torch.from_numpy(v).to(device) for k, v in arrays.items()}
    result.update(lengths=torch.tensor([len(r['y']) for r in records], device=device),
                  closed=torch.tensor([r['closed'] for r in records], device=device),
                  dw=torch.tensor([r['dice_weight'] for r in records], device=device))
    return result


def loss(model, b, sampling_factor):
    logits, boundary = model(b['x'], b['lengths'], b['closed'])
    p = logits.sigmoid()
    valid = torch.arange(p.shape[1], device=p.device)[None, :] < b['lengths'][:, None]
    dice = 1 - (2*(p*b['y']).sum(1)+1e-6) / ((p*valid).sum(1)+b['y'].sum(1)+1e-6)
    occupancy_loss = (F.binary_cross_entropy_with_logits(logits, b['y'], reduction='none') * b['w']).sum()
    boundary_loss = (F.binary_cross_entropy_with_logits(boundary, b['boundary'], reduction='none') * b['bw']).sum()
    return sampling_factor * (occupancy_loss + (dice*b['dw']).sum() + boundary_loss)


def controls(model, dest):
    model.eval()
    generator = torch.Generator().manual_seed(719)
    x = torch.randn(2, 56, 37, generator=generator)
    ordinary = model.input(x)
    equivalent = pointwise(model.input, x)
    equivalence_error = (ordinary-equivalent).abs().max().item()
    assert equivalence_error < 2e-5
    lengths = torch.tensor([37, 37])
    closed = torch.tensor([False, True])
    with torch.no_grad():
        a, b = model(x, lengths, closed)
        ar, br = model(x.flip(2), lengths, closed)
        ap, bp = model(F.pad(x, (0, 17)), lengths, closed)
    reverse = max((a-ar.flip(1)).abs().max().item(), (b-br.flip(1)).abs().max().item())
    padding = max((a-ap[:, :37]).abs().max().item(), (b-bp[:, :36]).abs().max().item())
    assert reverse < 2e-5 and padding < 2e-5, (reverse, padding)
    # An internal break must survive even when occupancy is positive everywhere.
    q = np.arange(0., 26., 5.)
    d = decode_intervals(q, np.ones(6), np.array([0., 0., 1., 0., 0.]), False)
    assert [(r['lo'], r['hi']) for r in d] == [(0., 12.5), (12.5, 25.)]
    write(dest / 'architecture_controls.json', dict(reversal_max_abs=reverse, padding_max_abs=padding,
        pointwise_conv_linear_max_abs=equivalence_error,
        open_and_closed_neighbors_checked=True, contiguous_children_break_preserved=True))


def decode_intervals(q, occupancy, boundary, closed):
    positive = occupancy >= .5
    domains = prior.intervals(q, positive, False)
    # Retain every >=.5 local boundary peak; a plateau uses its central edge.
    peak = (boundary >= .5) & (boundary >= np.r_[-np.inf, boundary[:-1]]) & (boundary >= np.r_[boundary[1:], -np.inf])
    locations = np.flatnonzero(peak)
    groups = np.split(locations, np.flatnonzero(np.diff(locations) > 1)+1)
    cuts = [(q[g[0]] + q[g[0]+1] + q[g[-1]] + q[g[-1]+1])*.25 for g in groups if len(g)]
    result = []
    for domain in domains:
        edges = [domain['lo']] + [float(c) for c in cuts if domain['lo'] < c < domain['hi']] + [domain['hi']]
        result.extend(dict(lo=a, hi=b, closed=False, wrap=False) for a, b in zip(edges[:-1], edges[1:]))
    if closed and positive.all() and not cuts:
        return [dict(lo=float(q[0]), hi=float(q[-1]), closed=True, wrap=False)]
    if closed and len(result) > 1 and positive[0] and positive[-1]:
        result = result[1:-1] + [dict(lo=result[0]['hi'], hi=result[-1]['lo'], closed=False, wrap=True)]
    return result


def train(dest, resume=False):
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(1031)
    np.random.seed(1031)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = SequenceModel()
    controls(model, dest)
    records, mean, scale = prepare(dest)
    check_resources()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # Release this worker's cold DLL pages only; do not terminate other jobs.
    write(dest / 'own_memory_trim.json', reclaim_own_idle_pages())
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    rng = np.random.default_rng(1031)
    trace = []
    elapsed_before = 0.
    first_epoch = 1
    if resume:
        state_path = dest / 'latest_training_state.pt'
        state = torch.load(state_path, map_location=device)
        assert np.array_equal(state['mean'], mean) and np.array_equal(state['scale'], scale)
        model.load_state_dict(state['state_dict'])
        assert all(torch.equal(v, state['state_dict'][k]) for k, v in model.state_dict().items())
        optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state = state['rng']
        trace = state['trace']
        first_epoch = int(state['epoch']) + 1
        elapsed_before = trace[-1]['seconds']
        write(dest / 'resume_receipt.json', dict(checkpoint_sha256=digest(state_path),
            completed_epoch=first_epoch-1, optimizer_restored=True, rng_restored=True,
            normalization_exact=True, weights_exact=True, prior_trace=trace,
            source_sha256=digest(__file__), reason='own cold pages trim and equivalent GEMM pointwise execution; resource guard unchanged'))
        del state
        print('RESUME_FROM_EPOCH', first_epoch, 'PRESERVED', first_epoch-1, flush=True)
    for epoch in range(first_epoch, EPOCHS+1):
        model.train()
        epoch_start = time.perf_counter()
        # Every known sequence once per epoch. Sort only within random buckets
        # to limit padding; no positive/negative or workpiece is omitted.
        ids = rng.permutation(len(records))
        groups = []
        for lo in range(0, len(ids), BATCH*16):
            bucket = sorted(ids[lo:lo+BATCH*16], key=lambda i: len(records[i]['y']))
            groups += [bucket[j:j+BATCH] for j in range(0, len(bucket), BATCH)]
        rng.shuffle(groups)
        value = 0.
        for ids in groups:
            b = batch([records[i] for i in ids], device)
            optimizer.zero_grad(set_to_none=True)
            objective = loss(model, b, len(records)/len(ids))
            assert torch.isfinite(objective)
            objective.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            assert torch.isfinite(norm)
            optimizer.step()
            value += float(objective.detach()) * len(ids) / len(records)
        trace.append(dict(epoch=epoch, loss=value, epoch_seconds=time.perf_counter()-epoch_start,
                          seconds=elapsed_before+time.perf_counter()-start))
        atomic_binary_write(dest / 'latest_training_state.pt', lambda f: torch.save(dict(
            state_dict=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
            rng=rng.bit_generator.state, mean=mean, scale=scale, trace=trace), f))
        write(dest / 'training_progress.json', dict(trace=trace, total_epochs=EPOCHS))
        print('EPOCH', epoch, '/', EPOCHS, 'LOSS', round(value, 6), 'SECONDS', round(trace[-1]['epoch_seconds'], 2), flush=True)
        del b, objective
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        trim = reclaim_own_idle_pages()
        trim.update(epoch=epoch)
        write(dest / 'latest_memory_trim.json', trim)
        check_resources()
    model.cpu().eval()
    controls(model, dest)
    payload = dict(state_dict=model.state_dict(), mean=mean, scale=scale, epochs=EPOCHS,
                   input_run='run390', architecture='SYMMETRIC_TCN_OCCUPANCY_BREAK_V1')
    atomic_binary_write(dest / 'sequence_model.pt', lambda f: torch.save(payload, f))
    reload = SequenceModel()
    reload.load_state_dict(torch.load(dest / 'sequence_model.pt', map_location='cpu')['state_dict'])
    b = batch(records[:2], 'cpu')
    with torch.no_grad():
        assert all(torch.equal(a, c) for a, c in zip(model(b['x'], b['lengths'], b['closed']), reload(b['x'], b['lengths'], b['closed'])))
    write(dest / 'training_receipt.json', dict(status='STRUCTURED_SEQUENCE_TRAINED', TRAIN=list(TRAIN), G=213,
        input_run='run390', epochs=EPOCHS, trace=trace, model_sha256=digest(dest/'sequence_model.pt'),
        threshold=.5, boundary_threshold=.5, DEV_reads=0, STEP_reads=0, reload_exact=True,
        seconds=elapsed_before+time.perf_counter()-start, full_contract_pass=False, device=str(device)))
    print('SEQUENCE_TRAIN_COMPLETE', round(elapsed_before+time.perf_counter()-start, 2), flush=True)


def infer(dest, key, input_run='run390', whole_run='run380'):
    start = time.perf_counter()
    torch.set_num_threads(2)
    receipt = read(dest / 'training_receipt.json')
    assert digest(dest/'sequence_model.pt') == receipt['model_sha256']
    payload = torch.load(dest/'sequence_model.pt', map_location='cpu')
    model = SequenceModel().eval()
    model.load_state_dict(payload['state_dict'])
    old_receipt = read(OUT/'run384/training_receipt.json')
    assert digest(OUT/'run384/whole_joint_model.pkl') == old_receipt['model_sha256']
    with (OUT/'run384/whole_joint_model.pkl').open('rb') as f:
        frozen = pickle.load(f)
    x, m, _ = prior.load_inputs(key,input_run)
    case = prior.load_case(key)
    case['x']._mmap.close()
    graph = case['graph']
    old = np.load(OUT/f'{whole_run}/source_{key}/joint_features.npy', mmap_mode='r')
    lookup = {int(p): i for i, p in enumerate(m['profile_ids'])}
    choices, probabilities, boundaries, offsets, boffsets = [], [], [], [0], [0]
    for row in m['admitted_rows']:
        row = int(row)
        j = lookup[row//2]
        a, b = m['offsets'][j:j+2]
        q = m['q_mm'][a:b]
        curve = graph.curve(row//2)
        closed = curve.kind == 'arc' and curve.arc.is_closed
        features = (prior.ordered_features(x[a:b], row%2) - payload['mean']) / payload['scale']
        with torch.no_grad():
            p, bp = model(torch.from_numpy(features.T.copy())[None], torch.tensor([len(q)]), torch.tensor([closed]))
            p, bp = p[0].sigmoid().numpy(), bp[0].sigmoid().numpy()
        probabilities.append(p); boundaries.append(bp)
        offsets.append(offsets[-1]+len(p)); boffsets.append(boffsets[-1]+len(bp))
        domains = decode_intervals(q, p, bp, closed)
        if not domains:
            continue
        category_x = np.column_stack((np.tile(old[row], (len(domains), 1)), np.arange(1, len(domains)+1)/frozen['K'])).astype(np.float32)
        categories = frozen['models']['category'].predict(category_x)
        for domain, category in zip(domains, categories):
            attr = prior.attributes(category)
            attr.update(forward=True, closed=domain['closed'], wrap=domain['wrap'])
            choices.append(dict(query=len(choices), probability=1., child_count=1, closed=domain['closed'], children=[dict(
                profile=row//2, intervals=[0, 0], fractions=[domain['lo']/curve.length, domain['hi']/curve.length],
                side='UNKNOWN', connection='UNKNOWN', **attr)]))
    result = prior.decode_shared(choices, prior.ContinuousGraphBank(graph))
    result.update(scope='STRUCTURED_SEQUENCE_RAW_PROPOSALS_NOT_FINAL_PARENT_OUTPUT', GT_reads=0, STEP_reads=0)
    folder = dest/f'proposal_{key}'
    folder.mkdir(exist_ok=False)
    with (folder/'decoded.json').open('w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    np.savez(folder/'occupancy.npz', admitted_rows=m['admitted_rows'], offsets=np.array(offsets),
             probabilities=np.concatenate(probabilities), boundary_offsets=np.array(boffsets), boundaries=np.concatenate(boundaries))
    write(folder/'prediction_receipt.json', dict(status='PURE_INPUT_PROPOSALS_FROZEN_BEFORE_EVALUATION', source=key,
        decoded_sha256=digest(folder/'decoded.json'), model_sha256=receipt['model_sha256'],
        proposed_intervals=len(choices), GT_reads=0, STEP_reads=0, seconds=time.perf_counter()-start))
    print('SEQUENCE_PREDICT', key, 'INTERVALS', len(choices), 'SECONDS', round(time.perf_counter()-start, 2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--stage', choices=['train', 'infer'], required=True)
    ap.add_argument('--source')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    dest = OUT/args.run
    if args.stage == 'train' and not args.resume:
        dest.mkdir(exist_ok=False)
        write(dest/'registered_protocol.json', dict(source_sha256=digest(__file__), TRAIN=list(TRAIN), DEV=['64'],
            epochs=EPOCHS, batch=BATCH, optimizer='Adam', learning_rate=.001, seed=1031, threshold=.5,
            frozen_admission='run384', frozen_category='run384', input_run='run390',
            full_sequences=True, all_known_rows_each_epoch=True, final_epoch_only=True,
            best379_preserved=True, external_consultation_paused=True))
    try:
        train(dest, args.resume) if args.stage == 'train' else infer(dest, args.source)
    except BaseException:
        write(dest/f'{args.stage}_{args.source or "all"}_failure.json', dict(error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
