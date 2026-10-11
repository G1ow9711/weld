"""Frozen input event fields -> learned interval assignment; conditional pilot.

Candidate generation never reads weld labels. The original contract evaluator
scores every emitted child after learned parent assembly. STEP remains an
explicit diagnostic input, so these results cannot qualify PCD deployment.
"""
import argparse
import json
import math
import time
import traceback
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F
from weld_pipeline.sjg_geometry_776 import read, write, digest
from weld_pipeline.sjg_prepare_776 import ROOT, OUT, TRAIN, VALID, ARC
from weld_pipeline.sjg_model_786 import SurfaceJointNet786 as SurfaceJointNet777
from weld_pipeline.sjg_local_frame_786 import local_arrays
from weld_pipeline.sjg_heads_support_788 import train_parent
from weld_pipeline.sjg_train_776 import check_resources, utc
from weld_pipeline.sjg_path_781 import ParentLink, assemble
from weld_pipeline.sjg_contract_score_778 import adapt, total
from weld_pipeline.contract_geometry_d3_409 import score_source

BASE = OUT / 'run10'
PATH_BASE = OUT / 'run07'
SEED = 2026100782


class IntervalHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('scale', torch.ones(dim))
        self.net = nn.Sequential(nn.Linear(dim, 128), nn.SiLU(),
                                 nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net((x-self.mean)/self.scale).squeeze(-1)


def build_candidates(si, base):
    folder = OUT / f'source_{si:02d}'
    freeze = read(folder / 'input_freeze.json')
    assert digest(folder/'inputs.npz') == freeze['input_sha256']
    assert digest(folder/'geometry.json') == freeze['geometry_sha256']
    with np.load(folder/'inputs.npz', allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    geometry = read(folder/'geometry.json')
    a['nodes'], a['patches'], a['edge_type'] = local_arrays(a, geometry)
    contacts = geometry['contacts']
    query_node = np.repeat(np.arange(len(contacts), dtype=np.int64), np.diff(a['ptr']))
    probabilities, offsets = [], []
    with torch.no_grad():
        h = base.encode_graph(*[torch.from_numpy(a[k]).cuda()
                              for k in ('nodes', 'edge_index', 'edge_type')])
        joint = base.joint_head(h).sigmoid().squeeze(-1).cpu().numpy()
        for begin in range(0, len(query_node), 2048):
            end = begin+2048
            logits, off = base.decode(h, torch.from_numpy(a['patches'][begin:end]).cuda(),
                torch.from_numpy(a['query'][begin:end]).cuda(),
                torch.from_numpy(query_node[begin:end]).cuda())
            probabilities.append(logits.sigmoid().cpu().numpy())
            offsets.append(off.cpu().numpy())
        h = h.cpu().numpy()
    p = np.concatenate(probabilities); off = np.concatenate(offsets)
    rows, features = [], []
    for ci, c in enumerate(contacts):
        if len(set(c['materials'])) < 2 or joint[ci] < .5:
            continue
        begin, end = a['ptr'][ci:ci+2]
        pp, oo = p[begin:end], off[begin:end]
        events = []
        for head in (1, 2):
            peaks = find_peaks(np.r_[-1, pp[:, head], -1], height=.5, distance=2)[0]-1
            # Finite-contact boundary proposals are input geometry. They still
            # require learned validity; no automatic positive or GT coordinate.
            q = [(int(k), float(np.clip(k*c['query_step_mm']+
                   2*np.clip(oo[k, head-1], -1.5, 1.5), 0, c['length'])), 0.)
                 for k in peaks]
            for k, location in ((0, 0.), (len(pp)-1, c['length'])):
                if not any(abs(location-v[1]) < .05 for v in q):
                    q.append((k, float(location), 1.))
            events.append(sorted(q, key=lambda v: v[1]))
        starts, ends = events
        for s, (sk, lo, boundary_s) in enumerate(starts):
            for e, (ek, hi, boundary_e) in enumerate(ends):
                if hi-lo < 1. or ek < sk:
                    continue
                span = pp[sk:ek+1, 0]
                midpoint = pp[(sk+ek)//2, 0]
                numeric = np.r_[lo/max(c['length'], 1.), hi/max(c['length'], 1.),
                    lo/1000., (c['length']-hi)/1000., (hi-lo)/1000.,
                    np.log1p(hi-lo)/10., pp[sk], pp[ek],
                    span.mean(), span.min(), np.quantile(span, .25), midpoint,
                    joint[ci], boundary_s, boundary_e,
                    np.sum(pp[sk+1:ek, 1:3] >= .5)/max(ek-sk, 1)]
                features.append(np.r_[h[ci], numeric])
                rows.append([ci, s, e, lo, hi])
    return contacts, h, np.asarray(rows, np.float64).reshape(-1, 5), \
        np.asarray(features, np.float32).reshape(-1, h.shape[1]+20)


def training_labels(si, rows):
    """One-to-one target assignment, after all input proposals were frozen."""
    coverage = read(OUT/f'source_{si:02d}/coverage_v2.json')
    y = np.zeros(len(rows), np.float32)
    represented = 0
    for ci in np.unique(rows[:, 0].astype(np.int64)):
        ids = np.flatnonzero(rows[:, 0] == ci)
        truth = [a for a in coverage['assignments'] if a['contact'] == ci]
        if not truth:
            continue
        target = np.asarray([[a['lo'], a['hi']] for a in truth])
        errors = np.abs(rows[ids, None, 3:5]-target[None, :, :]).max(2)
        # Dummy slots make the matching maximize the number within 3 mm before
        # minimizing residuals. Missing targets remain in the original G.
        cost = np.where(errors <= 3., errors, 1e6)
        cost = np.concatenate([cost, np.full((len(ids), len(ids)), 1e4)], axis=1)
        ii, jj = linear_sum_assignment(cost)
        chosen = [int(ids[i]) for i, j in zip(ii, jj)
                  if j < len(truth) and errors[i, j] <= 3.]
        y[chosen] = 1.; represented += len(chosen)
    return y, dict(original_G=coverage['original_G'], candidate_supported=represented)


def assign_intervals(rows, logits):
    """Maximum-weight event matching with explicit unmatched dummy columns."""
    selected = []
    for ci in np.unique(rows[:, 0].astype(np.int64)):
        ids = np.flatnonzero(rows[:, 0] == ci)
        ns, ne = int(rows[ids, 1].max())+1, int(rows[ids, 2].max())+1
        gain = np.full((ns, ne+ns), -1e6, np.float64)
        gain[:, ne:] = 0.
        lookup = {}
        for k in ids:
            s, e = map(int, rows[k, 1:3]); gain[s, e] = logits[k]; lookup[s, e] = int(k)
        ii, jj = linear_sum_assignment(-gain)
        selected.extend(lookup[i, j] for i, j in zip(ii, jj)
                        if j < ne and gain[i, j] > 0.)
    return sorted(selected)


def predictions_from_rows(rows, selected, contacts):
    parents = {}
    for k in selected:
        ci = int(rows[k, 0]); c = contacts[ci]; lo, hi = rows[k, 3:5]
        if ci not in parents:
            parents[ci] = dict(id=str(ci), type='vertical' if abs(c['direction'][2])>.707
                               else 'horizontal', weld_side=None, segments=[])
        d = np.asarray(c['direction']); p = np.asarray(c['p0'])
        parents[ci]['segments'].append(dict(kind='line', points=[(p+lo*d).tolist(), (p+hi*d).tolist()]))
    return list(parents.values())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', default='run11'); args = ap.parse_args()
    out = OUT/args.run; out.mkdir(exist_ok=False)
    updates = 0; parent_updates = 0; started = time.monotonic()
    try:
        check_resources(); torch.set_num_threads(2)
        torch.manual_seed(SEED); np.random.seed(SEED)
        base_path = BASE/'relation_graph_last.pt'
        base_sha = digest(base_path)
        assert base_sha == read(BASE/'receipt.json')['arms']['relation_graph']['checkpoint_sha256']
        with base_path.open('rb') as f: checkpoint = torch.load(f, map_location='cuda')
        base = SurfaceJointNet777(checkpoint['node_dim']).cuda()
        base.load_state_dict(checkpoint['model']); base.eval()
        for parameter in base.parameters(): parameter.requires_grad_(False)
        contexts, frozen = {}, []
        for si in TRAIN+VALID:
            print('BUILD_INPUT_EVENT_PAIRS', si, flush=True)
            contacts, h, rows, x = build_candidates(si, base)
            file = out/f'source_{si:02d}_candidates.npz'
            np.savez_compressed(file, rows=rows, features=x, graph_features=h)
            contexts[si] = (contacts, h, rows, x)
            frozen.append(dict(source=si, path=file.name, sha256=digest(file), count=len(rows)))
            print('FROZEN_INPUT_PAIRS', si, len(rows), flush=True)
            check_resources()
        write(out/'candidate_freeze.json', dict(outputs=frozen, labels_read=0,
              all_candidates_frozen_before_labels=True))
        pieces = [training_labels(si, contexts[si][2]) for si in TRAIN]
        y = np.concatenate([v[0] for v in pieces])
        x = np.concatenate([contexts[si][3] for si in TRAIN])
        assert y.sum() > 0 and (y == 0).sum() > 0
        batch = 256; batches = math.ceil(len(x)/batch); epochs = min(100, 2000//batches)
        assert epochs >= 1, 'A complete epoch exceeds the registered update limit'
        protocol = dict(utc=utc(), stage='LEARNED_EVENT_INTERVAL_MATCHING',
            train=TRAIN, holdout=VALID, seed=SEED, batch=batch, epochs=epochs,
            batches_per_epoch=batches, maximum_updates=2000, actual_training_examples=len(x),
            positive_candidates=int(y.sum()), candidate_support={str(si):v[1] for si,v in zip(TRAIN,pieces)},
            all_candidates_frozen_before_labels=True, output_probability_threshold=.5,
            peak_probability_threshold=.5, endpoint_match_training_mm=3.,
            finite_boundary_proposals=True, raw_peak_pairs_all_lengths=True,
            optimizer='AdamW', learning_rate=.001, frozen_geometry_checkpoint_sha256=base_sha,
            parent_head_retrained_for_this_geometry=True,parent_maximum_updates=2000, source_sha256=digest(__file__),
            prior_failed_round='run10', review='new local graph needs its own interval and parent heads',
            external_messages=0, STEP_conditioned_not_PCD_acceptance=True,
            formal_independent=False, full_contract_pass=False, large_scale_allowed=False)
        write(out/'registered_protocol.json', protocol)
        print('PAIRING_PROTOCOL', json.dumps(protocol['candidate_support']), 'epochs', epochs,
              'batches', batches, 'positive', int(y.sum()), 'total', len(x), flush=True)
        model = IntervalHead(x.shape[1]).cuda()
        model.mean.copy_(torch.from_numpy(x.mean(0)).cuda())
        model.scale.copy_(torch.from_numpy(x.std(0).clip(.01)).cuda())
        xx = torch.from_numpy(x).cuda(); yy = torch.from_numpy(y).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        rng = np.random.default_rng(SEED)
        # Balanced positives prevent the large invalid-pair pool dominating.
        weight = torch.tensor(float((len(y)-y.sum())/y.sum()), device='cuda')
        with (out/'epochs.jsonl').open('w', encoding='utf8') as log:
            for epoch in range(1, epochs+1):
                order = rng.permutation(len(x)); loss_sum = 0.
                for begin in range(0, len(x), batch):
                    ids = torch.as_tensor(order[begin:begin+batch], dtype=torch.long, device='cuda')
                    loss = F.binary_cross_entropy_with_logits(model(xx[ids]), yy[ids], pos_weight=weight)
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite pairing loss')
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
                    updates += 1; loss_sum += float(loss.detach())
                status = dict(utc=utc(), epoch=epoch, epochs=epochs, updates=updates,
                              loss=loss_sum/batches, complete_epoch=True)
                log.write(json.dumps(status)+'\n'); log.flush(); write(out/'progress.json', status)
                print('INTERVAL_TRAIN Epoch', f'{epoch}/{epochs}', 'Updates', updates,
                      'Loss', round(status['loss'], 6), flush=True)
                if epoch % 10 == 0: check_resources()
        with (out/'interval_head.pt').open('wb') as f:
            torch.save(dict(model=model.state_dict(), dim=x.shape[1], protocol=protocol), f)
        model.eval(); outputs = []; raw_predictions = {}; selections = {}; raw_freeze = []
        for si in TRAIN+VALID:
            contacts,h,rows,features=contexts[si]
            with torch.no_grad(): logits=model(torch.from_numpy(features).cuda()).cpu().numpy()
            selected=assign_intervals(rows,logits)
            raw_predictions[si]=predictions_from_rows(rows,selected,contacts)
            selections[si]=(selected,logits)
            path=out/f'source_{si:02d}_intervals.json';write(path,raw_predictions[si])
            raw_freeze.append(dict(source=si,path=path.name,sha256=digest(path)))
        write(out/'interval_freeze.json',dict(outputs=raw_freeze,all_frozen_before_parent_training=True))
        inventory=read(OUT/'inventory.json')
        path_model,parent_updates=train_parent(raw_predictions,contexts,inventory,out)
        for si in TRAIN+VALID:
            contacts, h, rows, features = contexts[si]
            selected,logits=selections[si]
            interval_predictions=raw_predictions[si]
            predictions, links = assemble(interval_predictions, contacts, h, path_model)
            target = out/f'source_{si:02d}_predictions.json'; write(target, predictions)
            write(out/f'source_{si:02d}_selection.json', dict(selected_candidate_ids=selected,
                  logits=logits.tolist(), links=links))
            outputs.append(dict(source=si, path=target.name, sha256=digest(target)))
            print('FROZEN_INTERVAL_OUTPUT', si, 'children', len(selected), flush=True)
        write(out/'prediction_freeze.json', dict(outputs=outputs, all_frozen_before_scoring=True))
        inventory = read(OUT/'inventory.json'); scores = {}
        for row in outputs:
            si = row['source']; target = ROOT/inventory[si]['target']['target_path']
            assert digest(target) == inventory[si]['target']['target_sha256']
            assert digest(out/row['path']) == row['sha256']
            scores[si] = score_source(adapt(read(out/row['path'])), read(target))
            write(out/f'source_{si:02d}_contract_score.json', scores[si])
            print('CONTRACT_SCORE', si, scores[si]['categories']['overall'], flush=True)
        assert digest(base_path) == base_sha
        write(out/'receipt.json', dict(utc=utc(), status='SMALL_INTERVAL_TRAIN_AND_TEST_COMPLETE',
            actual_optimizer_updates=updates+parent_updates,interval_updates=updates,parent_updates=parent_updates, parent_checkpoint_sha256=digest(out/"parent_link.pt"),train=total([scores[i] for i in TRAIN]),
            development_holdout=total([scores[i] for i in VALID]),
            checkpoint_sha256=digest(out/'interval_head.pt'), elapsed_seconds=time.monotonic()-started,
            pure_XYZ_accuracy=None, E16_status='NOT_IMPLEMENTED_AND_INCOMPLETE_REVIEWED_TRUTH',
            arc_G_unimplemented=inventory[ARC[0]]['G'], full_contract_pass=False, large_scale_allowed=False))
        print('INTERVAL_PILOT_COMPLETE | full acceptance not passed', flush=True)
    except BaseException as error:
        write(out/'failure.json', dict(error=str(error), traceback=traceback.format_exc(), updates=updates,parent_updates=parent_updates))
        raise


if __name__ == '__main__':
    main()
