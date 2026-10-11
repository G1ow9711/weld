"""Shared process-plane position features estimated from training labels.

Inference uses only point-derived features and the checkpoint parameters. No
piece identifier or reference table participates in the forward pass.
"""
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from weld_pipeline.typed_contact_model import TypedContactNet


def estimate_training_planes(pieces,min_count=8,min_groups=2,max_planes=12):
    observations=defaultdict(list)
    for piece in pieces:
        group=piece['record']['name']
        for weld in piece['truth']:
            segments=weld['segments']
            for segment,at_start in [(segments[0],True),(segments[-1],False)]:
                p=np.asarray(segment['points'],float)
                endpoint=p[0] if at_start else p[-1]
                tangent=p[1]-p[0] if at_start else p[-1]-p[-2]
                tangent=tangent/max(float(np.linalg.norm(tangent)),1e-9)
                for axis in range(3):
                    # An endpoint crossing a cut plane contributes; a whole
                    # horizontal weld lying on the same Z level does not.
                    if abs(tangent[axis])<.7:continue
                    key=(axis,round(float(endpoint[axis]),1))
                    observations[key].append((float(endpoint[axis]),group))
    candidates=[]
    for (axis,coordinate),values in observations.items():
        groups=sorted({g for _,g in values})
        if len(values)<min_count or len(groups)<min_groups:continue
        values_mm=np.array([v for v,_ in values])
        candidates.append({'axis':axis,'position_mm':float(np.median(values_mm)),
            'observations':len(values),'training_groups':groups,
            'max_initial_residual_mm':float(np.max(np.abs(values_mm-np.median(values_mm))))})
    return sorted(candidates,key=lambda r:(-r['observations'],r['axis'],r['position_mm']))[:max_planes]


class ProcessPlaneContactNet(nn.Module):
    def __init__(self,features,planes,width=64):
        super().__init__();self.input_features=features
        if features!=70:raise ValueError('Expected v3 point-derived feature layout')
        self.register_buffer('plane_axes',torch.tensor([p['axis'] for p in planes],dtype=torch.long))
        self.cut_positions_mm=nn.Parameter(torch.tensor([p['position_mm'] for p in planes],dtype=torch.float32))
        self.register_buffer('scales_mm',torch.tensor([2.,8.,32.,128.]))
        self.net=TypedContactNet(features+len(planes)*5,width).net

    def forward(self,x):
        if not len(self.plane_axes):return self.net(x)
        # The last three channels are absolute input XYZ / 10000, as declared
        # by typed_sequences. Positions remain shared across every assembly.
        world=x[:,-3:,:].float()*10000.
        delta=world[:,self.plane_axes,:]-self.cut_positions_mm[None,:,None]
        normalized=delta[:,:,None,:]/self.scales_mm[None,None,:,None]
        radial=torch.exp(-.5*normalized.square()).flatten(1,2)
        signed=(delta/128.).clamp(-1,1)
        return self.net(torch.cat([x,radial,signed],dim=1))

    def load_base_state(self,state):
        current=self.state_dict()
        for key,value in state.items():
            if key=='net.0.weight':
                current[key].zero_();current[key][:,:self.input_features].copy_(value)
            elif key in current:
                if current[key].shape!=value.shape:raise ValueError('Base shape mismatch: '+key)
                current[key].copy_(value)
        self.load_state_dict(current)


def model_from_checkpoint(state):
    if state.get('model_name','typed')=='process_planes':
        planes=[{'axis':int(a),'position_mm':float(p)} for a,p in zip(state['model']['plane_axes'],state['model']['cut_positions_mm'])]
        model=ProcessPlaneContactNet(state['features'],planes)
    else:model=TypedContactNet(state['features'])
    model.load_state_dict(state['model'])
    return model
