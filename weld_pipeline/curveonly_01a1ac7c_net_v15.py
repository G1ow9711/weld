"""Direct dimensionless interval endpoints; known arclength conversion outside NN.

The6310-parameter architecture and geometric decoder are the pretraining v14
controls. No v14 neural weights were trained or used.
"""
import time
import numpy as np
import torch
from . import curveonly_01a1ac7c_net_v14 as base
from .curveonly_01a1ac7c_net_v14 import TinyNet,batches,reverse_features,decode,controls,native_slice
cad=base.cad


class CurveNet(base.CurveNet):
    def predict(self, features, packet):
        started=time.perf_counter();module=self.module()
        x=np.clip((features-self.mean)/self.scale,-8.,8.)
        spans=list(zip(packet['offsets'][:-1],packet['offsets'][1:]));lengths=[b-a for a,b in spans]
        result=np.zeros((len(x),6),np.float32)
        with torch.inference_mode():
            for group in batches(lengths):
                longest=max(lengths[i]for i in group);values=np.zeros((len(group),34,longest),np.float32);ll=[]
                for j,ci in enumerate(group):
                    a,b=spans[ci];n=b-a;ll.append(n);values[j,:,:n]=x[a:b].T
                raw=module(torch.from_numpy(values),torch.tensor(ll,dtype=torch.int64))
                probabilities=torch.softmax(raw[:,:3],1).numpy();bounds=raw[:,3:5].numpy()
                closed=torch.sigmoid(raw[:,5]).numpy()
                for j,ci in enumerate(group):
                    first,last=spans[ci];n=last-first;p=np.asarray(packet['candidates'][ci]['points'])
                    length=float(base.cad.arclength(p)[-1]);station=np.linspace(0.,length,n)
                    a=bounds[j,0,:n].astype(float)*length;b=bounds[j,1,:n].astype(float)*length
                    if np.linalg.norm(p[0]-p[-1])<=1e-6:
                        extent=b-a;a=np.mod(a,length);b=a+extent
                        station=np.where(station<a,station+length,station)
                    result[first:last,:3]=probabilities[j,:,:n].T
                    result[first:last,3]=station-a;result[first:last,4]=b-station
                    result[first:last,5]=closed[j,:n]
        self.last_timing=dict(cpu_seconds=time.perf_counter()-started,parameters=sum(v.size for v in self.state.values()),
            input_stations=len(x),threads=torch.get_num_threads(),padded_buckets=True,
            head='NORMALIZED_NATIVE_INTERVAL_ENDPOINTS_WITH_EXACT_LENGTH_CONVERSION')
        return result
