"""Separate shared classification and millimeter localization networks."""
import numpy as np
import torch
from torch import nn

from weld_pipeline.bounded_event_phase_v2 import FEATURE_COUNT, bounded_features, cell_offsets
from weld_pipeline.canonical_horizontal_patch_heatmaps import canonical_frame
from weld_pipeline.continuous_center_modes import center_modes as or_modes
from weld_pipeline.continuous_center_modes_v2 import center_modes as mean_modes, restore_event_order

VERSION = 'shared_separate_bounded_event_network_v1'
POOLING = {'mean': mean_modes, 'or': or_modes}


class EventField(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.input = nn.Linear(FEATURE_COUNT, 512)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 512),
            nn.GELU(), nn.Linear(512, 512)) for _ in range(3)])
        self.output = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, outputs))
        self.skip = nn.Linear(FEATURE_COUNT, outputs, bias=False)

    def forward(self, values):
        hidden = torch.nn.functional.gelu(self.input(values))
        for block in self.blocks:
            hidden = hidden+block(hidden)
        return self.output(hidden)+self.skip(values)


class BoundedEventNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = EventField(4)
        self.regressor = EventField(3)

    def forward(self, values):
        return torch.cat([self.classifier(values), self.regressor(values)], dim=1)


@torch.no_grad()
def predict_fields(model, data, sequences, mean, standard_deviation, chunk_size=4096):
    model.eval(); device = next(model.parameters()).device
    mean = np.asarray(mean, np.float32); std = np.asarray(standard_deviation, np.float32)
    if mean.shape != (FEATURE_COUNT,) or std.shape != mean.shape or np.any(std<=0):
        raise ValueError('Fixed training-only bounded normalizer required')
    fields = [np.zeros((len(s['xyz']), 7), np.float32) for s in sequences]
    for entry in data['entries']:
        index = entry['sequence']
        if entry['type'] != 'horizontal' or sequences[index].get('curve_kind') == 'arc':
            continue
        seq, _, transform = canonical_frame(sequences[index])
        if transform != entry['transform']:
            raise ValueError('Raw bounded query canonical order changed')
        for start in range(0, len(seq['xyz']), chunk_size):
            stop = min(len(seq['xyz']), start+chunk_size)
            values = bounded_features(seq, data, entry, np.arange(start, stop))
            fields[index][start:stop] = model(torch.from_numpy((values-mean)/std).to(device)).cpu().numpy()
    return fields


def assemble_events(data, sequences, fields, pooling):
    if pooling not in POOLING or len(fields) != len(sequences):
        raise ValueError('One fixed pooling configuration and complete fields required')
    foregrounds = [np.zeros(len(s['xyz']), np.float32) for s in sequences]
    events = [[[], [], []] for s in sequences]
    for entry in data['entries']:
        index = entry['sequence']
        if entry['type'] != 'horizontal' or sequences[index].get('curve_kind') == 'arc':
            continue
        seq, _, transform = canonical_frame(sequences[index])
        if transform != entry['transform']:
            raise ValueError('Raw bounded event canonical order changed')
        raw = np.asarray(fields[index]); step = float(seq['sample_step_mm'])
        if raw.shape != (len(seq['xyz']), 7) or not np.isfinite(raw).all():
            raise ValueError('Complete finite bounded fields required')
        positions = np.arange(len(raw), dtype=float)*step
        offsets = cell_offsets(raw[:, 4:7], positions)
        sigmoid = lambda values: 1./(1.+np.exp(-np.clip(np.asarray(values, float), -60., 60.)))
        hypotheses = POOLING[pooling](sigmoid(raw[:, 1:4]), offsets, step)
        foregrounds[index], events[index] = restore_event_order(sigmoid(raw[:, 0]), hypotheses,
            (len(raw)-1)*step, transform)
    return foregrounds, events


def infer_event_sets(model, data, sequences, mean, standard_deviation, pooling):
    return assemble_events(data, sequences, predict_fields(model, data, sequences, mean, standard_deviation), pooling)
