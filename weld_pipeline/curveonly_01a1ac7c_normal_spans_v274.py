"""Append qualified finite-face observations to the frozen134 ordered generator."""
import types
import numpy as np
import torch
from torch import nn
from . import curveonly_01a1ac7c_direction_spans_v247 as previous
from . import curveonly_01a1ac7c_qualified_normals_v271 as normals
from .curveonly_01a1ac7c_corner_publication_v256 import decode

OBSERVATIONS, CHANNELS = 167, previous.CHANNELS
NAMES = previous.NAMES+tuple('qualified_normal:'+name for name in normals.NAMES)
PERMUTATION = np.r_[previous.PERMUTATION,np.arange(134,167)]
assert len(NAMES)==OBSERVATIONS and np.array_equal(PERMUTATION[PERMUTATION],np.arange(167))
codec,base,normalize,arclength = previous.codec,previous.base,previous.normalize,previous.arclength
sequence = previous.sequence


def candidate_rows(features,packet):
    rows=previous.candidate_rows(features,packet)
    values=np.asarray(packet['qualified_surface_normals_v271']['values'],np.float32)
    assert values.shape==(packet['offsets'][-1],33) and np.isfinite(values).all()
    for row in rows:
        grid=np.arange(len(row['grid']))*row['grid_step_mm']
        local=values[row['lo']:row['hi']]
        extra=np.column_stack([np.interp(grid,row['station'],local[:,j])for j in range(33)]).astype(np.float32)
        row['grid']=np.column_stack([row['grid'],extra]).astype(np.float32)
        assert row['grid'].shape[1]==OBSERVATIONS
    return rows


pack=types.FunctionType(previous.pack.__code__,dict(previous.__dict__,OBSERVATIONS=OBSERVATIONS,PERMUTATION=PERMUTATION))
predict=types.FunctionType(previous.predict.__code__,dict(previous.__dict__,pack=pack))


class OrderedSpanNetwork(previous.OrderedSpanNetwork):
    def __init__(self):
        super().__init__()
        original=self.input
        augmented=nn.Conv1d(OBSERVATIONS,CHANNELS,1)
        with torch.no_grad():
            augmented.weight[:,:134].copy_(original.weight)
            augmented.weight[:,134:].zero_()
            augmented.bias.copy_(original.bias)
        self.input=augmented


class OrderedSpanSet(previous.OrderedSpanSet):
    def predict(self,features,packet):
        origin=previous.previous
        return types.FunctionType(origin.OrderedSpanSet.predict.__code__,
            dict(origin.__dict__,candidate_rows=candidate_rows,predict=predict))(self,features,packet)


def input_curves(record):
    load=previous.cad.load_material
    captured={}
    def capture(path):
        solids=load(path)
        captured['solids']=solids
        return solids
    previous.cad.load_material=capture
    try:
        features,packet=previous.input_curves(record)
    finally:
        previous.cad.load_material=load
    values,receipt=normals.build(packet,captured.get('solids'))
    packet=dict(packet,qualified_surface_normals_v271=dict(values=values.tolist(),receipt=receipt))
    return features,packet
