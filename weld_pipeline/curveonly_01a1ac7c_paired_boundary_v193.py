"""Paired boundary events, bounded offsets and explicit full native cycles."""
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import curveonly_01a1ac7c_paired_endpoints_v190 as codec

base = codec.previous.previous
OBSERVATIONS = 111
OUTPUT_PERMUTATION = np.array([0, 2, 1, 4, 3, 5], np.int64)
OUTPUT_SIGN = np.array([1, 1, 1, -1, -1, 1], np.float32)


def targets(row, chosen):
    n, step, length = len(row['grid']), row['grid_step_mm'], row['length_mm']
    station = np.arange(n)*step
    fields = np.zeros((6, n), float)
    full, codes = False, []
    start_instance, end_instance = np.full(n, -1), np.full(n, -1)
    for ii, value in enumerate(chosen):
        a, b = value['start_mm'], value['end_mm']
        code = codec.encode(row, a, b)
        codes.append(dict(code, truth=value['truth']))
        if code['full_cycle']:
            assert len(chosen) == 1
            fields[0] = 1.
            full = True
            continue
        for turn in ((-1, 0, 1) if row['closed'] else (0,)):
            fields[0, (station+turn*length >= a) & (station+turn*length <= b)] = 1.
        i, j = code['start_index'], code['end_index']
        assert start_instance[i] == end_instance[j] == -1, 'Same-role grid cell collision'
        start_instance[i], end_instance[j] = ii, ii
        fields[1, i], fields[2, j] = 1., 1.
        fields[3, i], fields[4, j] = code['start_offset'], code['end_offset']
    both = (fields[1] > 0) & (fields[2] > 0)
    fields[5, both] = fields[4, both] < fields[3, both]
    return dict(fields=fields, full=float(full), codes=codes,
                order_known=both, start_instance=start_instance, end_instance=end_instance)


def peaks(probability, closed):
    p = np.asarray(probability, float)
    positive = p > .5
    if not positive.any():
        return []
    groups, current = [], []
    for i in range(len(p)):
        if positive[i]:
            current.append(i)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    if closed and len(groups) > 1 and positive[0] and positive[-1]:
        groups = [groups[-1]+groups[0]]+groups[1:-1]
    return [max(ids, key=lambda i: (p[i], -i)) for ids in groups]


def decode_fields(row, fields, full_probability):
    f = np.asarray(fields, float)
    assert f.shape == (6, len(row['grid'])) and np.isfinite(f).all()
    assert np.abs(f[3:5]).max(initial=0) <= .5+1e-7
    length, step, n = row['length_mm'], row['grid_step_mm'], len(row['grid'])
    if row['closed'] and full_probability > .5:
        confidence = float(np.sqrt(max(0., full_probability*f[0].mean())))
        if confidence > .5:
            return [dict(start_index=0, end_index=0, start_offset=0., end_offset=0.,
                         full_cycle=True, confidence=confidence)], dict(full_cycle=True)
    starts, ends = peaks(f[1], row['closed']), peaks(f[2], row['closed'])
    offsets = f[3:5].copy()
    projected = []
    for i in set(starts) & set(ends):
        left, right = sorted(offsets[:, i])
        if f[5, i] > .5:
            offsets[0, i], offsets[1, i] = right, left
        else:
            offsets[0, i], offsets[1, i] = left, right
        if not np.array_equal(offsets[:, i], f[3:5, i]):
            projected.append(i)
    events = []
    for role, ids in ((0, starts), (1, ends)):
        for i in ids:
            position = (i+offsets[role, i])*step
            position = position % length if row['closed'] else float(np.clip(position, 0, length))
            events.append(dict(role=role, index=i, position=float(position),
                               offset=float(offsets[role, i]), probability=float(f[1+role, i])))
    # End-before-start at an exactly shared coordinate preserves interval order.
    events.sort(key=lambda q: (q['position'], -q['role']))
    groups = []
    for event in events:
        if not groups or groups[-1][0]['role'] != event['role']:
            groups.append([event])
        else:
            groups[-1].append(event)
    if row['closed'] and len(groups) > 1 and groups[0][0]['role'] == groups[-1][0]['role']:
        groups = [groups[-1]+groups[0]]+groups[1:-1]
    representatives = [max(g, key=lambda q: (q['probability'], -q['index'])) for g in groups]
    codes = []
    for k, start in enumerate(representatives):
        if start['role'] != 0:
            continue
        if k+1 == len(representatives) and not row['closed']:
            continue
        end = representatives[(k+1) % len(representatives)]
        if end['role'] != 1:
            continue
        code = dict(start_index=start['index'], end_index=end['index'],
                    start_offset=start['offset'], end_offset=end['offset'], full_cycle=False)
        interval = codec.decode(row, code)
        if interval is None:
            continue
        a, b = interval
        station = np.arange(n)*step
        take = np.zeros(n, bool)
        for turn in ((-1, 0, 1) if row['closed'] else (0,)):
            take |= (station+turn*length >= a) & (station+turn*length <= b)
        if take.any():
            support = float(f[0, take].mean())
        else:
            # A positive sub-grid interval still has endpoint-local support.
            support = float((f[0, start['index']]+f[0, end['index']])*.5)
        confidence = float((max(0., start['probability']*end['probability']*support))**(1/3))
        if confidence > .5:
            codes.append(dict(code, confidence=confidence))
    return codes, dict(full_cycle=False, start_peaks=len(starts), end_peaks=len(ends),
                       ordered_role_groups=len(groups), local_order_projected_cells=projected,
                       complete_pairs=len(codes), no_minimum_weld_or_gap=True)


def pack(rows, mean, scale, device):
    size = np.asarray([len(q['grid']) for q in rows], np.int64)
    width = int(size.max())
    arrays = dict(x=np.zeros((len(rows), OBSERVATIONS, width), np.float32), size=size,
                  closed=np.asarray([q['closed'] for q in rows], bool))
    for key, channels in (('targets', 6), ('class_weight', 3), ('offset_weight', 2)):
        arrays[key] = np.zeros((len(rows), channels, width), np.float32)
    arrays['order_weight'] = np.zeros((len(rows), width), np.float32)
    arrays['full_target'] = np.zeros(len(rows), np.float32)
    arrays['full_weight'] = np.zeros(len(rows), np.float32)
    for i, row in enumerate(rows):
        n = size[i]
        arrays['x'][i, :, :n] = ((row['grid']-mean)/scale).T
        for key in ('targets', 'class_weight', 'offset_weight'):
            if key in row:
                arrays[key][i, :, :n] = row[key]
        if 'order_weight' in row:
            arrays['order_weight'][i, :n] = row['order_weight']
            arrays['full_target'][i] = row['full_target']
            arrays['full_weight'][i] = row['full_weight']
    return {key: torch.as_tensor(value, device=device) for key, value in arrays.items()}


class BoundaryNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        c = base.CHANNELS
        self.input = nn.Conv1d(OBSERVATIONS, c, 1)
        self.blocks = nn.ModuleList([base.BoundaryDepthwise(d) for d in base.DILATIONS])
        self.head = nn.Sequential(nn.Conv1d(3*c, c, 1), nn.SiLU(), nn.Conv1d(c, 6, 1))
        self.full_head = nn.Sequential(nn.Linear(2*c, c), nn.SiLU(), nn.Linear(c, 1))
        self.register_buffer('reverse_channels', torch.as_tensor(codec.previous.PERMUTATION.copy()))
        self.register_buffer('reverse_outputs', torch.as_tensor(OUTPUT_PERMUTATION.copy()))
        self.register_buffer('reverse_sign', torch.as_tensor(OUTPUT_SIGN.copy()))

    def forward(self, x, size, closed):
        mask = (torch.arange(x.shape[-1], device=x.device)[None, None, :] < size[:, None, None]).to(x.dtype)
        z = F.silu(self.input(x))*mask
        for block in self.blocks:
            z = block(z, size, closed, mask)
        average = z.sum(2, keepdim=True)/size[:, None, None]
        maximum = z.masked_fill(mask == 0, -torch.inf).amax(2, keepdim=True)
        point = self.head(torch.cat([z, average.expand_as(z), maximum.expand_as(z)], dim=1))*mask
        full = self.full_head(torch.cat([average[:, :, 0], maximum[:, :, 0]], dim=1))[:, 0]
        return point, full

    def both_directions(self, batch):
        x, size, closed = batch['x'], batch['size'], batch['closed']
        index = base.reverse_index(size, closed, x.shape[-1])
        reverse = x.gather(2, index[:, None, :].expand(-1, OBSERVATIONS, -1))[:, self.reverse_channels]
        a, af = self.forward(x, size, closed)
        b, bf = self.forward(reverse, size, closed)
        b = b.gather(2, index[:, None, :].expand(-1, 6, -1))[:, self.reverse_outputs]
        return a, b*self.reverse_sign[None, :, None], af, bf


def predict_fields(network, rows, mean, scale, device='cpu'):
    if str(device) == 'cpu' and torch.get_num_threads() != 2:
        torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    network.eval()
    result, full = [None]*len(rows), [None]*len(rows)
    with torch.no_grad():
        for ids in base.batches(rows):
            batch = pack([rows[i] for i in ids], mean, scale, device)
            a, b, af, bf = network.both_directions(batch)
            value = (a+b)*.5
            fields = torch.cat([torch.sigmoid(value[:, :3]), .5*torch.tanh(value[:, 3:5]),
                                torch.sigmoid(value[:, 5:6])], dim=1).cpu().numpy()
            cycle = torch.sigmoid((af+bf)*.5).cpu().numpy()
            for j, i in enumerate(ids):
                result[i] = fields[j, :, :len(rows[i]['grid'])].astype(float)
                full[i] = float(cycle[j])
    return result, full


class PairedBoundarySet:
    def __init__(self, network, mean, scale):
        self.network = network.cpu().eval()
        self.mean, self.scale = np.asarray(mean, np.float32), np.asarray(scale, np.float32)
        self.last_states = self.last_fields = self.last_full = self.last_timing = None

    def predict(self, features, packet):
        start = time.perf_counter()
        rows = codec.candidate_rows(features, packet)
        fields, full = predict_fields(self.network, rows, self.mean, self.scale)
        codes, states = [], []
        for row, values, cycle in zip(rows, fields, full):
            items, state = decode_fields(row, values, cycle)
            codes.append(items)
            states.append(dict(candidate=row['candidate'], codes=items, **state))
        self.last_states, self.last_fields, self.last_full = states, fields, full
        self.last_timing = dict(cpu_seconds=time.perf_counter()-start,
            parameter_count=sum(q.numel() for q in self.network.parameters()),
            paired_boundary_locations_local_offsets_and_full_cycle=True,
            no_count_template_CTMC_or_minimum_length=True)
        return codec.proposals(packet, rows, codes)


candidate_rows = codec.candidate_rows
input_curves = codec.input_curves
arclength = codec.arclength
NAMES = codec.previous.NAMES


def decode(features, packet, model, threshold=.5):
    return base.original_runtime.decode(features, packet, model, threshold)
