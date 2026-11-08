"""Append finite material direction observations to the unchanged ordered grammar."""
import types
import numpy as np
import torch
from torch import nn
from . import curveonly_01a1ac7c_ordered_spans_v235 as previous
from . import curveonly_01a1ac7c_material_direction_v160 as direction
from . import curveonly_01a1ac7c as cad
from OCP.BRepCheck import BRepCheck_Analyzer

OBSERVATIONS = 134
CHANNELS = previous.CHANNELS
NAMES = tuple(previous.NAMES)+tuple('direction:'+s for s in direction.NAMES)
PERMUTATION = np.r_[previous.PERMUTATION, np.arange(111,134)]
assert len(NAMES) == OBSERVATIONS and np.array_equal(PERMUTATION[PERMUTATION], np.arange(134))
codec, base, normalize, arclength = previous.codec, previous.base, previous.normalize, previous.arclength
sequence = previous.sequence


def candidate_rows(features, packet):
    rows = previous.candidate_rows(features, packet)
    values = np.asarray(packet['local_material_direction_v160']['values'], np.float32)
    assert values.shape == (packet['offsets'][-1], 23) and np.isfinite(values).all()
    for row in rows:
        grid = np.arange(len(row['grid']))*row['grid_step_mm']
        local = values[row['lo']:row['hi']]
        extra = np.column_stack([np.interp(grid,row['station'],local[:,j]) for j in range(23)]).astype(np.float32)
        row['grid'] = np.column_stack([row['grid'],extra]).astype(np.float32)
        assert row['grid'].shape[1] == OBSERVATIONS
    return rows


pack = types.FunctionType(previous.pack.__code__, dict(previous.__dict__,
    OBSERVATIONS=OBSERVATIONS, PERMUTATION=PERMUTATION))
predict = types.FunctionType(previous.predict.__code__, dict(previous.__dict__, pack=pack))


class OrderedSpanNetwork(previous.OrderedSpanNetwork):
    def __init__(self):
        # Construct exactly the fresh111-channel network first: every common
        # parameter has the same random initialization as the control network.
        super().__init__()
        original = self.input
        augmented = nn.Conv1d(OBSERVATIONS, CHANNELS, 1)
        with torch.no_grad():
            augmented.weight[:,:111].copy_(original.weight)
            augmented.weight[:,111:].zero_()
            augmented.bias.copy_(original.bias)
        self.input = augmented


class OrderedSpanSet(previous.OrderedSpanSet):
    def predict(self, features, packet):
        return types.FunctionType(previous.OrderedSpanSet.predict.__code__,
            dict(previous.__dict__,candidate_rows=candidate_rows,predict=predict))(self,features,packet)


def input_curves(record):
    """Same raw134 observations, capturing the existing material load once."""
    load = cad.load_material
    captured = {}
    def capture(path):
        solids = load(path)
        captured['solids'] = solids
        return solids
    cad.load_material = capture
    try:
        features,packet = previous.input_curves(record)
    finally:
        cad.load_material = load
    solids = captured.get('solids')
    validity = [bool(BRepCheck_Analyzer(s,True).IsValid()) for s in solids] if solids else []
    values,receipt = direction.build(packet,solids if solids and all(validity) else None)
    if solids and not all(validity):
        receipt.update(reason='INVALID_FINITE_MATERIAL_BREP')
    if not solids:
        receipt.update(reason='NO_STEP_INPUT')
    packet = dict(packet,local_material_direction_v160=dict(values=values.tolist(),
        receipt=dict(receipt,real_solid_validity=validity)))
    return features,packet


def decode(features,packet,model,threshold=.5):
    return previous.decode(features,packet,model,threshold)
