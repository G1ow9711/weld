"""Learned exact finite-source endpoint types on a frozen paired backbone."""
import time
import numpy as np
import torch
from torch import nn
from . import curveonly_01a1ac7c_instance_order_v197 as previous

core = previous.previous
codec, base = previous.codec, previous.base
OBSERVATIONS = 111
OUTPUT_PERMUTATION = np.array([0, 2, 1, 4, 3, 5, 7, 6], np.int64)
OUTPUT_SIGN = np.array([1, 1, 1, -1, -1, 1, 1, 1], np.float32)


def targets(row, chosen):
    result = previous.targets(row, chosen)
    result['fields'] = np.vstack([result['fields'], np.zeros((2, len(row['grid'])))])
    if not row['closed']:
        tolerance = codec.reference.PROJECTION_NUMERIC_MM
        for choice, code in zip(chosen, result['codes']):
            result['fields'][6, code['start_index']] = abs(choice['start_mm']) <= tolerance
            result['fields'][7, code['end_index']] = abs(choice['end_mm']-row['length_mm']) <= tolerance
    return result


def decode_fields(row, fields, full_probability):
    values = np.asarray(fields, float)
    assert values.shape == (8, len(row['grid']))
    adjusted = values[:6].copy()
    decisions = []
    if not row['closed']:
        for role, boundary, probability_head, offset_head, state_head in (
                ('START', 0, 1, 3, 6), ('END', len(row['grid'])-1, 2, 4, 7)):
            selected = boundary in core.peaks(values[probability_head], False)
            applied = bool(selected and values[state_head, boundary] > .5)
            if applied:
                adjusted[offset_head, boundary] = 0.
            decisions.append(dict(role=role, grid_index=boundary,
                selected_boundary_event=selected, probability=float(values[state_head, boundary]),
                exact_source_endpoint_used=applied,
                original_offset_mm=float(values[offset_head, boundary]*row['grid_step_mm']),
                reason='LEARNED_FINITE_SOURCE_ENDPOINT_TYPE' if applied else 'INDEPENDENT_LOCAL_OFFSET'))
    codes, state = core.decode_fields(row, adjusted, full_probability)
    return codes, dict(state, source_endpoint_decisions=decisions,
                       partial_closed_origin_never_snapped=True)


def pack(rows, mean, scale, device):
    proxy = [dict(row, targets=row['targets'][:6]) if 'targets' in row else row for row in rows]
    batch = core.pack(proxy, mean, scale, device)
    extra = np.zeros((len(rows), 2, batch['x'].shape[-1]), np.float32)
    for i, row in enumerate(rows):
        if 'targets' in row:
            extra[i, :, :len(row['grid'])] = row['targets'][6:8]
    batch['targets'] = torch.cat([batch['targets'], torch.as_tensor(extra, device=device)], dim=1)
    return batch


class BoundaryNetwork(core.BoundaryNetwork):
    def __init__(self):
        super().__init__()
        self.source_head = nn.Conv1d(base.CHANNELS, 2, 1)
        self.reverse_outputs = torch.as_tensor(OUTPUT_PERMUTATION.copy())
        self.reverse_sign = torch.as_tensor(OUTPUT_SIGN.copy())
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith('source_head.'))

    def hidden(self, x, size, closed):
        mask = (torch.arange(x.shape[-1], device=x.device)[None, None, :] < size[:, None, None]).to(x.dtype)
        z = core.F.silu(self.input(x))*mask
        for block in self.blocks:
            z = block(z, size, closed, mask)
        average = z.sum(2, keepdim=True)/size[:, None, None]
        maximum = z.masked_fill(mask == 0, -torch.inf).amax(2, keepdim=True)
        joined = torch.cat([z, average.expand_as(z), maximum.expand_as(z)], dim=1)
        hidden = self.head[1](self.head[0](joined))
        full = self.full_head(torch.cat([average[:, :, 0], maximum[:, :, 0]], dim=1))[:, 0]
        return hidden, full, mask

    def forward(self, x, size, closed):
        hidden, full, mask = self.hidden(x, size, closed)
        point = torch.cat([self.head[2](hidden), self.source_head(hidden)], dim=1)*mask
        return point, full

    def hidden_directions(self, batch):
        x, size, closed = batch['x'], batch['size'], batch['closed']
        index = base.reverse_index(size, closed, x.shape[-1])
        reverse = x.gather(2, index[:, None, :].expand(-1, OBSERVATIONS, -1))[:, self.reverse_channels]
        a = self.hidden(x, size, closed)[0]
        b = self.hidden(reverse, size, closed)[0]
        return a, b.gather(2, index[:, None, :].expand(-1, a.shape[1], -1))

    def both_directions(self, batch):
        x, size, closed = batch['x'], batch['size'], batch['closed']
        index = base.reverse_index(size, closed, x.shape[-1])
        reverse = x.gather(2, index[:, None, :].expand(-1, OBSERVATIONS, -1))[:, self.reverse_channels]
        a, af = self.forward(x, size, closed)
        b, bf = self.forward(reverse, size, closed)
        b = b.gather(2, index[:, None, :].expand(-1, 8, -1))[:, self.reverse_outputs]
        return a, b*self.reverse_sign[None, :, None], af, bf

    def load_backbone(self, state):
        original = {k: v for k, v in state.items() if k not in ('reverse_outputs', 'reverse_sign')}
        result = self.load_state_dict(original, strict=False)
        assert not result.unexpected_keys
        assert set(result.missing_keys) == {'reverse_outputs', 'reverse_sign',
                                             'source_head.weight', 'source_head.bias'}
        assert all(torch.equal(parameter, state[name]) for name, parameter in self.named_parameters()
                   if not name.startswith('source_head.'))


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
            # Preserve the original six elementwise transforms exactly.
            fields = torch.cat([torch.sigmoid(value[:, :3]), .5*torch.tanh(value[:, 3:5]),
                torch.sigmoid(value[:, 5:6]), torch.sigmoid(value[:, 6:8])], dim=1).cpu().numpy()
            cycle = torch.sigmoid((af+bf)*.5).cpu().numpy()
            for j, i in enumerate(ids):
                result[i] = fields[j, :, :len(rows[i]['grid'])].astype(float)
                full[i] = float(cycle[j])
    return result, full


class PairedBoundarySet(core.PairedBoundarySet):
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
            learned_exact_source_endpoint_types=True, original_six_heads_frozen=True,
            no_count_template_CTMC_or_minimum_length=True)
        return codec.proposals(packet, rows, codes)


candidate_rows, input_curves = codec.candidate_rows, codec.input_curves
arclength, NAMES, decode = codec.arclength, previous.NAMES, core.decode
