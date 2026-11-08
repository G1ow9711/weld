"""Physical-window station classifier with an unbounded number of intervals."""
import time
import numpy as np
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_set_v18 as base
from . import curveonly_01a1ac7c_local_material_v94 as material

# Local geometry/observation columns. Global XYZ height, shape length/radius,
# normalized station, source size and whole-count shortcuts are not inputs.
RAW_COLUMNS=(5,6,11,12,13,14,15,19,20,21,22,23,24,25,26,27,28,29,30,31,32)
WINDOW_MM=(-35.,-15.,-5.,0.,5.,15.,35.)
FEATURES=(len(RAW_COLUMNS)+len(material.NAMES))*len(WINDOW_MM)+len(WINDOW_MM)


def observations(features,packet):
    block=packet.get('local_finite_material_v94')
    if block is None:raise ValueError('Local finite evidence packet is missing')
    values=np.asarray(block['values'],np.float32)
    assert values.shape==(len(features),len(material.NAMES))
    return np.column_stack([features[:,RAW_COLUMNS],values]).astype(np.float32)


def encode(features,packet):
    source=observations(features,packet);out=[]
    for ci,candidate in enumerate(packet['candidates']):
        lo,hi=packet['offsets'][ci:ci+2];x=source[lo:hi];s=np.asarray(candidate['station_s_mm']);L=float(s[-1])
        points=np.asarray(candidate['points']);closed=np.linalg.norm(points[0]-points[-1])<=1e-6
        columns=[];valid=[]
        for delta in WINDOW_MM:
            query=s+delta
            if closed:
                query=query%L;ss=np.r_[s[:-1]-L,s,s[1:]+L];xx=np.concatenate([x[:-1],x,x[1:]])
                good=np.ones(len(s),np.float32)
            else:ss=s;xx=x;good=((query>=0)&(query<=L)).astype(np.float32)
            columns.append(np.column_stack([np.interp(query,ss,xx[:,j])for j in range(x.shape[1])]))
            valid.append(good)
        out.append(np.column_stack([*columns,*valid]).astype(np.float32))
    result=np.concatenate(out)if out else np.empty((0,FEATURES),np.float32)
    assert result.shape==(len(features),FEATURES)and np.isfinite(result).all()
    return result


def intervals(station,probabilities,classes,closed,length):
    """Interpolate class decision crossings; retain every positive interval."""
    s=np.asarray(station);p=np.asarray(probabilities).copy();classes=np.asarray(classes)
    if closed:p[-1]=p[0]
    label=classes[p.argmax(1)];label[p.max(1)<.5]=0
    cuts=np.r_[0,np.flatnonzero(label[1:]!=label[:-1])+1,len(label)];runs=[]
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        kind=int(label[lo])
        if kind<=0:continue
        index=int(np.flatnonzero(classes==kind)[0]);zero=int(np.flatnonzero(classes==0)[0])
        def crossing(a,b):
            other=zero if label[a]==0 or label[b]==0 else int(np.flatnonzero(classes==label[a if label[a]!=kind else b])[0])
            y0=p[a,index]-p[a,other];y1=p[b,index]-p[b,other]
            return float(s[a]+(s[b]-s[a])*np.clip(-y0/(y1-y0),0,1))if abs(y1-y0)>1e-12 else float((s[a]+s[b])/2)
        a=0. if lo==0 else crossing(lo-1,lo);b=length if hi==len(s)else crossing(hi-1,hi)
        runs.append(dict(start_mm=a,end_mm=b,kind=kind,confidence=float(p[lo:hi,index].mean())))
    if closed and len(runs)>1 and runs[0]['start_mm']==0. and runs[-1]['end_mm']==length and runs[0]['kind']==runs[-1]['kind']:
        first,last=runs[0],runs[-1];a=last['end_mm']-last['start_mm'];b=first['end_mm']
        combined=dict(start_mm=last['start_mm'],end_mm=length+first['end_mm'],kind=first['kind'],confidence=(a*last['confidence']+b*first['confidence'])/max(a+b,1e-12))
        runs=[*runs[1:-1],combined]
    return [q for q in runs if q['end_mm']>q['start_mm']+1e-6]


class LocalCurveSet:
    def __init__(self,classifier):self.classifier=classifier;self.last_timing=None;self.last_states=None

    def predict(self,features,packet):
        tick=time.perf_counter();x=encode(features,packet);prob=self.classifier.predict_proba(x)if len(x)else np.empty((0,len(self.classifier.classes_)))
        proposals=[];states=[]
        for ci,candidate in enumerate(packet['candidates']):
            lo,hi=packet['offsets'][ci:ci+2];points=np.asarray(candidate['points']);L=float(cad.arclength(points)[-1]);closed=np.linalg.norm(points[0]-points[-1])<=1e-6
            runs=intervals(candidate['station_s_mm'],prob[lo:hi],self.classifier.classes_,closed,L)
            states.append(dict(candidate=ci,intervals=runs))
            for j,q in enumerate(runs):
                proposals.append(dict(candidate=ci,points=base.native_slice(points,q['start_mm'],q['end_mm']).tolist(),
                    weld_type='horizontal'if q['kind']==1 else'vertical',confidence=q['confidence'],
                    interval=dict(start_mm=q['start_mm'],end_mm=q['end_mm'],index=j,predicted_count=len(runs))))
        self.last_states=states;self.last_timing=dict(cpu_seconds=time.perf_counter()-tick,local_features=FEATURES,
            physical_window_mm=list(WINDOW_MM),input_curves=len(packet['candidates']),input_stations=len(features),
            learned_count_or_ordinal_endpoint_template=False,all_positive_intervals_retained=True)
        return proposals


class LocalRefinement:
    """Preserve existing fine partitions, replace only when local count grows.

    This uses decoded input proposals, never labels or a workpiece identity.
    It also permits a local interval on a source rejected by the global head.
    The baseline is this experiment's own TRAIN-complement model in controls.
    """
    def __init__(self,baseline,local):
        self.baseline=baseline;self.local=local;self.last_timing=None;self.last_changes=None

    def predict(self,features,packet):
        tick=time.perf_counter();old=self.baseline.predict(features,packet);new=self.local.predict(features,packet)
        old_groups={};new_groups={}
        for q in old:old_groups.setdefault(q['candidate'],[]).append(q)
        for q in new:new_groups.setdefault(q['candidate'],[]).append(q)
        out=[];changes=[]
        for ci in range(len(packet['candidates'])):
            a=old_groups.get(ci,[]);b=new_groups.get(ci,[]);replace=len(b)>len(a)
            out.extend(b if replace else a)
            if replace:changes.append(dict(candidate=ci,old_raw_intervals=len(a),new_local_intervals=len(b)))
        self.last_changes=changes;self.last_timing=dict(cpu_seconds=time.perf_counter()-tick,
            old_model=self.baseline.last_timing,local_model=self.local.last_timing,
            local_refinement_changes=changes,rule='Use local intervals only when their count exceeds decoded global count; otherwise retain global intervals.',
            no_answer_or_name_lookup=True)
        return out
