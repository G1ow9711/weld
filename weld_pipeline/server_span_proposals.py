"""Inference-only span proposals/features from learned continuous query outputs."""
import numpy as np
from weld_pipeline.server_geometry_events import geometry
from weld_pipeline.contract_curve_geometry import CircularArc


def role_peaks(positions,outputs,role,length):
    ids=np.flatnonzero(outputs[:,role]>=0);order=ids[np.argsort(-outputs[ids,role],kind='stable')];keep=[];s=[]
    for i in order:
        t=float(np.clip(positions[i]+outputs[i,4+role],0,length))
        if any(abs(t-v)<1. for v in s):continue
        keep.append(int(i));s.append(t)
    return list(zip(keep,s))


def representation(candidate,start,end):
    length,point,_,arc=geometry(candidate['representation'])
    if arc is None:return dict(kind='line',endpoints_mm=point(np.array([start,end])).tolist())
    sub=arc.subarc(min(start,end),max(start,end))
    if end<start:
        sub=CircularArc(sub.center_mm,sub.radius_mm,sub.plane_normal,(sub.point_at(sub.length_mm)-sub.center_mm)/sub.radius_mm,-sub.sweep_rad,sub.is_closed)
    return dict(kind='arc',arc_parameters=sub.to_parameters())


def build(candidates,metadata,features,outputs):
    boundaries=np.r_[0,np.flatnonzero(np.diff(metadata[:,0]))+1,len(metadata)];rows=[];xs=[]
    for ci,(begin,end) in enumerate(zip(boundaries[:-1],boundaries[1:])):
        assert metadata[begin,0]==ci
        candidate=candidates[ci];length,_,_,arc=geometry(candidate['representation']);s=metadata[begin:end,1];p=outputs[begin:end];x=features[begin:end]
        for typ in range(2):
            starts=role_peaks(s,p,2*typ,length);ends=role_peaks(s,p,2*typ+1,length)
            for ia,sa in starts:
                for ib,sb in ends:
                    if abs(sa-sb)<=1e-6:continue
                    query=np.linspace(sa,sb,17);occupancy=np.interp(query,s,1/(1+np.exp(-np.clip(p[:,8+typ],-60,60))))
                    shape=np.array([np.log1p(abs(sb-sa)),abs(sb-sa)/length,float(sb>sa),float(typ),float(arc is not None),float(occupancy.mean()),float(occupancy.min()),float(occupancy.max())])
                    xs.append(np.r_[x[ia],x[ib],p[ia],p[ib],occupancy,shape]);rows.append(dict(candidate=ci,start=sa,end=sb,weld_type='vertical' if typ else 'horizontal',representation=representation(candidate,sa,sb)))
    return rows,np.asarray(xs,np.float32)
