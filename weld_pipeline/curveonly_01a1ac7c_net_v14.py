"""From-scratch small sequence net for complete curved-weld intervals.

Inference uses only the independent34D XYZ/material-curve observations. There
are no case IDs, truth paths, previous fitted models or label-source readers.
"""
from __future__ import annotations
import time, math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial import cKDTree
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_arc_publication_v6 as publication

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def padded_valid(x, lengths):
    index = (lengths-1)[:, None, None].expand(-1, x.shape[1], 1)
    end = x.gather(2, index)
    mask = torch.arange(x.shape[2], device=x.device)[None, None] >= lengths[:, None, None]
    return torch.where(mask, end, x)


class Block(nn.Module):
    def __init__(self, width, dilation):
        super().__init__(); self.pad = 2*dilation
        self.depth = nn.Conv1d(width, width, 5, dilation=dilation, groups=width)
        self.mix = nn.Conv1d(width, width, 1)

    def forward(self, x, lengths):
        p = self.pad
        extended = torch.cat([x[:, :, :1].expand(-1, -1, p), x,
                              x[:, :, -1:].expand(-1, -1, p)], 2)
        y = self.mix(F.silu(self.depth(extended)))
        return padded_valid(x+.5*F.silu(y), lengths)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__(); self.stem = nn.Conv1d(34, 32, 1)
        self.blocks = nn.ModuleList([Block(32, d) for d in (1, 4, 16, 64)])
        self.head = nn.Conv1d(32, 6, 1)

    def forward(self, x, lengths):
        x = padded_valid(F.silu(self.stem(x)), lengths)
        for block in self.blocks:
            x = block(x, lengths)
        return self.head(x)


def batches(lengths, rng=None, token_limit=32768, maximum_batch=32):
    buckets = {}
    for i, n in enumerate(lengths):
        buckets.setdefault(int(math.ceil(math.log2(max(1, n)))), []).append(i)
    groups = []
    for key in sorted(buckets):
        indices = np.asarray(buckets[key])
        if rng is not None:
            rng.shuffle(indices)
        batch = []; longest = 0
        for i in indices:
            n = int(lengths[i]); nextlong = max(longest, n)
            if batch and ((len(batch)+1)*nextlong > token_limit or len(batch) >= maximum_batch):
                groups.append(batch); batch = []; longest = 0
            batch.append(int(i)); longest = max(longest, n)
        if batch:
            groups.append(batch)
    if rng is not None:
        rng.shuffle(groups)
    return groups


def reverse_features(x):
    y = x[::-1].copy(); y[:, 9] = 1.-y[:, 9]
    for a, b in ((23,24), (27,28), (29,30)):
        y[:, [a,b]] = y[:, [b,a]]
    return y


class CurveNet:
    classes_ = np.asarray([0, 1, 2])
    n_features_in_ = 34

    def __init__(self, state, mean, scale):
        self.state = state; self.mean = np.asarray(mean, np.float32); self.scale = np.asarray(scale, np.float32)
        self._module = None; self.last_timing = None

    def __getstate__(self):
        return dict(state=self.state, mean=self.mean, scale=self.scale, _module=None, last_timing=None)

    def module(self):
        if self._module is None:
            m = TinyNet(); m.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in self.state.items()})
            self._module = m.eval()
        return self._module

    def predict(self, features, packet):
        started = time.perf_counter(); module = self.module()
        x = np.clip((features-self.mean)/self.scale, -8., 8.)
        spans = list(zip(packet['offsets'][:-1], packet['offsets'][1:])); lengths = [b-a for a,b in spans]
        result = np.zeros((len(x), 6), np.float32)
        with torch.inference_mode():
            for group in batches(lengths):
                longest = max(lengths[i] for i in group); values = np.zeros((len(group), 34, longest), np.float32)
                ll = []
                for j, ci in enumerate(group):
                    a,b = spans[ci]; n = b-a; ll.append(n)
                    values[j, :, :n] = x[a:b].T
                raw = module(torch.from_numpy(values), torch.tensor(ll, dtype=torch.int64))
                prob = torch.softmax(raw[:, :3], 1)
                distance = F.softplus(raw[:, 3:5])*100.
                closed = torch.sigmoid(raw[:, 5:6])
                out = torch.cat([prob, distance, closed], 1).numpy()
                for j, ci in enumerate(group):
                    a,b = spans[ci]; result[a:b] = out[j, :, :b-a].T
        self.last_timing = dict(cpu_seconds=time.perf_counter()-started, parameters=sum(v.size for v in self.state.values()),
            input_stations=len(x), threads=torch.get_num_threads(), padded_buckets=True)
        return result

    def predict_proba(self, features):
        raise RuntimeError('Curve sequence boundaries require the input packet; call predict(features, packet)')


def native_slice(points, start, end):
    p = np.asarray(points, float); length = float(cad.arclength(p)[-1])
    if end <= length+1e-8:
        return cad.slice_curve(p, start, min(end, length))
    if np.linalg.norm(p[0]-p[-1]) > 1e-6 or end-start > length+1e-6:
        raise ValueError('Unproved or repeated native wrap')
    a = cad.slice_curve(p, start, length); b = cad.slice_curve(p, 0., end-length)
    return np.vstack([a, b[1:]])


def interval_votes(c, predictions, threshold):
    p = np.asarray(c['points']); length = float(cad.arclength(p)[-1])
    s = np.linspace(0., length, len(predictions)); prob = predictions[:, :3]
    cls = np.argmax(prob, axis=1); selected = (cls>0) & (np.max(prob, axis=1)>=threshold)
    intervals = []; closed_native = np.linalg.norm(p[0]-p[-1]) <= 1e-6
    for kind in (1, 2):
        rows = np.flatnonzero(selected & (cls==kind))
        if not len(rows):
            continue
        a = s[rows]-predictions[rows, 3]; b = s[rows]+predictions[rows, 4]
        span = b-a; closed_vote = closed_native & (predictions[rows, 5]>=.5)
        if closed_native:
            a = np.mod(a, length); b = a+span
            a[closed_vote] = 0.; b[closed_vote] = length
            keep = (span > 1e-6) & ((span <= length+1e-7) | closed_vote)
        else:
            keep = (a>=-3.) & (b<=length+3.) & (b>a+1e-6)
            a = np.maximum(a, 0.); b = np.minimum(b, length)
            keep &= b > a+1e-6
        rows = rows[keep]; a = a[keep]; b = b[keep]
        if not len(rows):
            continue
        votes = np.column_stack([a,b]); w = prob[rows, kind]
        keys = np.round(votes/4.).astype(np.int64); unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        mass = np.bincount(inverse, weights=w); order = np.argsort(-mass, kind='stable')
        used = np.zeros(len(votes), bool); accepted = []
        for bin_index in order:
            center = unique[bin_index]*4.
            take = (~used) & (abs(votes-center).max(1)<=4.)
            if not take.any():
                continue
            aa,bb = np.average(votes[take], axis=0, weights=w[take]); nn = int(take.sum())
            support = float(w[take].sum()); expected = max(1., (bb-aa)/2.)
            if nn < min(2, len(rows)) or support < .2*expected:
                continue
            used[take] = True
            # One consensus may straddle adjacent4mm bins; retain one proposal.
            if any(max(abs(aa-x['start']), abs(bb-x['end']))<=4. for x in accepted):
                continue
            accepted.append(dict(start=float(aa), end=float(bb), cls=kind,
                confidence=float(w[take].mean()), support=nn, expected_stations=expected))
        intervals.extend(accepted)
    return intervals


def decode(features, packet, model, threshold=.5):
    started = time.perf_counter(); predictions = model.predict(features, packet); proposals = []
    for ci, c in enumerate(packet['candidates']):
        a,b = packet['offsets'][ci:ci+2]
        for vote in interval_votes(c, predictions[a:b], threshold):
            p = native_slice(c['points'], vote['start'], vote['end'])
            proposals.append(dict(points=p.tolist(), source_candidate=ci, vote=vote,
                weld_type='horizontal' if vote['cls']==1 else 'vertical', confidence=vote['confidence']))
    kept = []
    for proposal in sorted(proposals, key=lambda r:-r['confidence']):
        p = cad.resample(proposal['points'],1.); duplicate = False
        for old in kept:
            q = np.asarray(old['points'])
            if proposal['weld_type'] != old['weld_type'] or np.linalg.norm(p.mean(0)-q.mean(0))>5.:
                continue
            if max(cKDTree(p).query(q,workers=1)[0].max(), cKDTree(q).query(p,workers=1)[0].max())<.5:
                duplicate=True; break
        if not duplicate:
            kept.append(proposal)
    parents = []
    for i, r in enumerate(kept):
        p = np.asarray(r['points']); fit = cad.circle_fit(p)
        segment = dict(segment_index=0, points=p.tolist(), length_mm=float(cad.arclength(p)[-1]),
            geometry_type='arc' if fit else 'bspline_polyline', reason='COMPLETE_INTERVAL_ENDPOINT_VOTES',
            status='NOMINAL_GEOMETRY_OBSERVATION_RECORDED')
        parents.append(dict(parent_id=f'curve_{i:05d}', weld_type=r['weld_type'], weld_side=None,
            is_closed=bool(np.linalg.norm(p[0]-p[-1])<.1), segments=[segment], points=p.tolist(),
            confidence=r['confidence'], source_candidate=r['source_candidate'], circle=fit,
            interval_vote=r['vote'], actual_geometry_status='NOT_INDEPENDENTLY_ACCEPTED'))
    decoded = dict(parents=parents, geometry_units=len(parents), published_segments=len(parents), units='mm',
        feature_columns=34, same_model_and_configuration=True, sequence_model=model.last_timing,
        model_stage_seconds=time.perf_counter()-started)
    return publication.publish(decoded)


def controls():
    torch.manual_seed(20261106); m = TinyNet().eval()
    a = torch.randn(1,34,7); b = torch.randn(1,34,43)
    together = torch.zeros(2,34,43); together[0,:,:7]=a[0]; together[1]=b[0]
    with torch.inference_mode():
        alone = m(a,torch.tensor([7])); batch = m(together,torch.tensor([7,43]))[0,:,:7]
    error = float((alone[0]-batch).abs().max())
    assert error < 1e-5
    x = np.arange(34*5,dtype=np.float32).reshape(5,34)/100
    assert np.max(abs(reverse_features(reverse_features(x))-x))<1e-6
    return dict(passed=True, parameters=sum(v.numel() for v in m.parameters()),
        maximum_batch_padding_difference=error, feature_reverse_involution=True, CPU_threads=torch.get_num_threads())
