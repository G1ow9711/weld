"""Frozen independent tiny encoder plus small whole-interval regressors."""
import time
import numpy as np
from . import curveonly_01a1ac7c_net_v15 as sequence
from .curveonly_01a1ac7c_interval_decode_v16 import decode


def observations(encoder, features, packet):
    captured=[];module=encoder.module()
    def before_head(layer,args):
        captured.append(args[0].detach().numpy().copy())
    handle=module.head.register_forward_pre_hook(before_head)
    try:
        result=encoder.predict(features,packet)
    finally:
        handle.remove()
    lengths=np.diff(packet['offsets']);groups=sequence.batches(lengths)
    hidden=np.empty((len(features),32),np.float32)
    assert len(captured)==len(groups)
    for group,batch in zip(groups,captured):
        for j,ci in enumerate(group):
            a,b=packet['offsets'][ci:ci+2];hidden[a:b]=batch[j,:,:b-a].T
    return result,hidden


class CurveHybrid:
    classes_=np.array([0,1,2]);n_features_in_=34

    def __init__(self,encoder,start,end,raw_skip=False):
        self.encoder=encoder;self.start=start;self.end=end;self.raw_skip=raw_skip;self.last_timing=None

    def predict(self,features,packet):
        tick=time.perf_counter();out,hidden=observations(self.encoder,features,packet)
        x=np.column_stack([features,hidden])if self.raw_skip else hidden
        active=(np.argmax(out[:,:3],axis=1)>0)&(out[:,:3].max(1)>=.5)
        bounds=np.zeros((len(x),2),float)
        if active.any():
            bounds[active,0]=self.start.predict(x[active]);bounds[active,1]=self.end.predict(x[active])
        for ci,c in enumerate(packet['candidates']):
            first,last=packet['offsets'][ci:ci+2];p=np.array(c['points']);L=float(sequence.cad.arclength(p)[-1])
            s=np.linspace(0.,L,last-first);a=bounds[first:last,0]*L;b=bounds[first:last,1]*L
            if np.linalg.norm(p[0]-p[-1])<=1e-6:
                extent=b-a;a=np.mod(a,L);b=a+extent;s=np.where(s<a,s+L,s)
            out[first:last,3]=s-a;out[first:last,4]=b-s
        self.last_timing=dict(cpu_seconds=time.perf_counter()-tick,encoder_parameters=6310,
            endpoint_feature_columns=x.shape[1],endpoint_tree_iterations=[self.start.n_iter_,self.end.n_iter_],
            selection_and_closure_encoder_unchanged=True,input_stations=len(x))
        return out
