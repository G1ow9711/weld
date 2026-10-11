"""Shared deep point/context network with physical signed event offsets."""
import numpy as np
import torch
from torch import nn

from weld_pipeline.canonical_horizontal_patch_heatmaps import canonical_frame, restore_probability_order
from weld_pipeline.forest_event_centers import build_inputs, combine_features, proximity_features
from weld_pipeline.near_event_votes import vote_heatmaps

VERSION = 'shared_deep_physical_event_votes_v1'
FEATURE_COUNT = 368


class PhysicalEventNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(FEATURE_COUNT, 512)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(512), nn.Linear(512,512),
            nn.GELU(), nn.Linear(512,512)) for _ in range(3)])
        self.output = nn.Sequential(nn.LayerNorm(512), nn.Linear(512,7))
        self.skip = nn.Linear(FEATURE_COUNT,7,bias=False)

    def forward(self, values):
        hidden = torch.nn.functional.gelu(self.input(values))
        for block in self.blocks:hidden = hidden+block(hidden)
        return self.output(hidden)+self.skip(values)


def extra_features(sequence, indices, actual_xyz):
    indices = np.asarray(indices, np.int64)
    step = float(sequence['sample_step_mm']); span=(len(sequence['xyz'])-1)*step
    positions=indices.astype(float)*step
    xyz=np.asarray(sequence['xyz'],float)[indices]/10000. if actual_xyz else np.zeros((len(indices),3))
    return np.c_[positions/1000.,(span-positions)/1000.,xyz].astype(np.float32)


def point_features(sequence, data, entry, indices, actual_xyz):
    indices=np.asarray(indices,np.int64); rows=entry['start']+indices
    local=np.c_[data['local'][rows],data['point_context'][rows]]
    physical=proximity_features(sequence,data['geometry'],data['planes'],indices)
    values=np.c_[combine_features(local,data['nodes'][entry['node']],physical),extra_features(sequence,indices,actual_xyz)]
    if values.shape != (len(indices),FEATURE_COUNT) or not np.isfinite(values).all():
        raise ValueError('Finite common neural event features required')
    return values.astype(np.float32)


def physical_votes(foreground, support, offsets, step):
    offsets=np.asarray(offsets,float);support=np.asarray(support,float).copy()
    # A prediction outside the trained 16 mm support radius supplies no vote;
    # clipping it to the radius would invent an event at the window edge.
    support[np.abs(offsets)>1.+1e-6]=0.
    return vote_heatmaps(foreground,support,offsets,step)


@torch.no_grad()
def infer(model, data, sequences, mean, standard_deviation, actual_xyz, chunk_size=8192):
    model.eval();device=next(model.parameters()).device
    mean=np.asarray(mean,np.float32);standard_deviation=np.asarray(standard_deviation,np.float32)
    if mean.shape!=(FEATURE_COUNT,) or standard_deviation.shape!=mean.shape or np.any(standard_deviation<=0):
        raise ValueError('One fixed training normalizer required')
    probabilities=[np.zeros((len(s['xyz']),7),np.float32) for s in sequences]
    for entry in data['entries']:
        index=entry['sequence']
        if entry['type']!='horizontal' or sequences[index].get('curve_kind')=='arc':continue
        seq,_,transform=canonical_frame(sequences[index])
        if transform!=entry['transform']:raise ValueError('Canonical neural point ordering changed')
        size=len(seq['xyz']);raw=np.zeros((size,7),np.float32)
        for a in range(0,size,chunk_size):
            b=min(size,a+chunk_size)
            values=point_features(seq,data,entry,np.arange(a,b),actual_xyz)
            raw[a:b]=model(torch.from_numpy((values-mean)/standard_deviation).to(device)).cpu().numpy()
        sigmoid=lambda x:1./(1.+np.exp(-np.clip(x.astype(float),-60.,60.)))
        votes=physical_votes(sigmoid(raw[:,0]),sigmoid(raw[:,1:4]),raw[:,4:7],float(seq['sample_step_mm']))
        probabilities[index]=restore_probability_order(votes,transform)
    return probabilities
