"""Deep physical offsets decoded as continuous typed center modes."""
import numpy as np
import torch

from weld_pipeline.canonical_horizontal_patch_heatmaps import canonical_frame
from weld_pipeline.continuous_center_modes import center_modes,restore_event_order
from weld_pipeline.neural_physical_event_votes import FEATURE_COUNT,PhysicalEventNet,extra_features,point_features

VERSION='shared_deep_physical_continuous_event_sets_v2'


@torch.no_grad()
def infer_event_sets(model,data,sequences,mean,standard_deviation,actual_xyz,chunk_size=8192):
    model.eval();device=next(model.parameters()).device
    mean=np.asarray(mean,np.float32);standard_deviation=np.asarray(standard_deviation,np.float32)
    if mean.shape!=(FEATURE_COUNT,) or standard_deviation.shape!=mean.shape or np.any(standard_deviation<=0):
        raise ValueError('One fixed training feature normalizer required')
    foregrounds=[np.zeros(len(s['xyz']),np.float32) for s in sequences]
    event_sets=[[[],[],[]] for s in sequences]
    for entry in data['entries']:
        index=entry['sequence']
        if entry['type']!='horizontal' or sequences[index].get('curve_kind')=='arc':continue
        seq,_,transform=canonical_frame(sequences[index])
        if transform!=entry['transform']:raise ValueError('Canonical continuous event ordering differs')
        size=len(seq['xyz']);raw=np.zeros((size,7),np.float32)
        for a in range(0,size,chunk_size):
            b=min(size,a+chunk_size);x=point_features(seq,data,entry,np.arange(a,b),actual_xyz)
            raw[a:b]=model(torch.from_numpy((x-mean)/standard_deviation).to(device)).cpu().numpy()
        sigmoid=lambda x:1./(1.+np.exp(-np.clip(x.astype(float),-60.,60.)))
        events=center_modes(sigmoid(raw[:,1:4]),raw[:,4:7],float(seq['sample_step_mm']))
        foregrounds[index],event_sets[index]=restore_event_order(sigmoid(raw[:,0]),events,(size-1)*float(seq['sample_step_mm']),transform)
    return foregrounds,event_sets
