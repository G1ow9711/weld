"""Cheap finite-carrier interval sets from pooled existing geometric observations."""
import time
import numpy as np
from scipy.spatial import cKDTree
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_arc_publication_v6 as publication
from .curveonly_01a1ac7c_interval_decode_v16 import native_slice

PROFILE=(5,6,7,8,11,19,20,29,30)


def reverse_features(x):
    y=x[::-1].copy();y[:,9]=1.-y[:,9]
    for a,b in ((23,24),(27,28),(29,30),(12,14)):
        y[:,[a,b]]=y[:,[b,a]]
    return y


def pooled(x):
    x=np.asarray(x,np.float32);q=np.linspace(0.,len(x)-1,13)
    profile=np.column_stack([np.interp(q,np.arange(len(x)),x[:,j])for j in PROFILE])
    return np.r_[x.mean(0),x.std(0),profile.ravel()].astype(np.float32)


def query(row,count,index):
    return np.r_[row,np.log1p(count),index/max(count,1),(index+.5)/max(count,1)].astype(np.float32)


class CurveSet:
    def __init__(self,count,interval,style):
        self.count=count;self.interval=interval;self.style=style;self.last_timing=None

    def predict(self,features,packet):
        start=time.perf_counter();offsets=packet['offsets']
        if not packet['candidates']:return []
        x=np.array([pooled(features[a:b])for a,b in zip(offsets[:-1],offsets[1:])])
        probabilities=self.count.predict_proba(x);labels=self.count.classes_[probabilities.argmax(1)]
        confidence=probabilities.max(1);rows=[];meta=[]
        for ci,(n,p)in enumerate(zip(labels,confidence)):
            if n<=0 or p<.5:continue
            for j in range(int(n)):rows.append(query(x[ci],int(n),j));meta.append((ci,int(n),j,float(p)))
        out=[]
        if rows:
            q=np.array(rows);bounds=self.interval.predict(q);sp=self.style.predict_proba(q)
            style=self.style.classes_[sp.argmax(1)];score=sp.max(1)
            for (ci,n,j,cf),bb,kind,cp in zip(meta,bounds,style,score):
                p=np.array(packet['candidates'][ci]['points']);L=float(cad.arclength(p)[-1]);a,b=bb*L
                closed_native=np.linalg.norm(p[0]-p[-1])<=1e-6
                if kind%2:
                    if not closed_native:continue
                    a,b=0.,L
                elif closed_native:
                    extent=b-a;a=float(a%L);b=a+extent
                    if not 1e-6<b-a<=L+1e-7:continue
                else:
                    if a<-3. or b>L+3. or b<=a+1e-6:continue
                    a=max(0.,a);b=min(L,b)
                if b<=a+1e-6:continue
                out.append(dict(candidate=ci,points=native_slice(p,a,b).tolist(),
                    weld_type='horizontal'if kind//2==1 else'vertical',confidence=cf*float(cp),
                    interval=dict(start_mm=float(a),end_mm=float(b),index=j,predicted_count=n)))
        self.last_timing=dict(cpu_seconds=time.perf_counter()-start,pooled_features=185,query_features=188,
            input_curves=len(x),interval_queries=len(rows),new_geometric_queries=0,neural_parameters=0)
        return out


def decode(features,packet,model,threshold=.5):
    tick=time.perf_counter();proposals=model.predict(features,packet);kept=[]
    # Identical final geometry is suppressed only at the old0.5mm criterion.
    for r in sorted(proposals,key=lambda r:-r['confidence']):
        p=cad.resample(r['points'],1.);duplicate=False
        for old in kept:
            q=np.asarray(old['points'])
            if r['weld_type']!=old['weld_type']or np.linalg.norm(p.mean(0)-q.mean(0))>5.:continue
            if max(cKDTree(p).query(q,workers=1)[0].max(),cKDTree(q).query(p,workers=1)[0].max())<.5:
                duplicate=True;break
        if not duplicate:kept.append(r)
    parents=[]
    for i,r in enumerate(kept):
        p=np.array(r['points']);fit=cad.circle_fit(p)
        segment=dict(segment_index=0,points=p.tolist(),length_mm=float(cad.arclength(p)[-1]),
            geometry_type='arc'if fit else'bspline_polyline',reason='FINITE_CARRIER_INTERVAL_SET',status='NOMINAL_GEOMETRY_OBSERVATION_RECORDED')
        parents.append(dict(parent_id=f'curve_{i:05d}',weld_type=r['weld_type'],weld_side=None,
            is_closed=bool(np.linalg.norm(p[0]-p[-1])<.1),segments=[segment],points=p.tolist(),
            confidence=r['confidence'],source_candidate=r['candidate'],circle=fit,interval=r['interval'],
            actual_geometry_status='NOT_INDEPENDENTLY_ACCEPTED'))
    result=dict(parents=parents,geometry_units=len(parents),published_segments=len(parents),units='mm',
        feature_columns=34,same_model_and_configuration=True,interval_model=model.last_timing,
        model_stage_seconds=time.perf_counter()-tick)
    return publication.publish(result)
