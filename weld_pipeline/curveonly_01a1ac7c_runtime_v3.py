"""Independent boundary head on the same lightweight curve observations."""
from .curveonly_01a1ac7c_runtime_v2 import *
from . import curveonly_01a1ac7c_runtime_v2 as previous
import numpy as np

class CurveHeads:
    def __init__(self,occupancy,boundary):
        self.occupancy=occupancy;self.boundary=boundary;self.classes_=occupancy.classes_;self.n_features_in_=occupancy.n_features_in_
    def predict_proba(self,x):return self.occupancy.predict_proba(x)

def publish_piece(parent,points,index):
    length=arclength(points)[-1];circle=circle_fit(points);n=1 if circle is None else max(1,int(math.ceil(abs(circle['sweep'])/(math.pi/6)-1e-10)));segments=[]
    for j in range(n):
        p=slice_curve(points,length*j/n,length*(j+1)/n)
        segments.append(dict(segment_index=j,points=p.tolist(),length_mm=float(arclength(p)[-1]),geometry_type='arc'if circle else 'bspline_polyline',reason='ARC_TURN_AT_MOST_30_DEGREES'if n>1 else 'PREDICTED_CURVE_INTERVAL',status='NOMINAL_GEOMETRY_OBSERVATION_RECORDED'))
    return dict(parent,parent_id=f'curve_{index:05d}',points=points.tolist(),segments=segments,circle=circle,is_closed=bool(np.linalg.norm(points[0]-points[-1])<.1),boundary_source='TRAIN_ONLY_BOUNDARY_HEAD')

def decode(features,packet,model,threshold=.5):
    original=previous.decode(features,packet,model.occupancy,threshold);events=model.boundary.predict_proba(features)[:,1]if len(features)else np.empty(0);parents=[]
    for parent in original['parents']:
        ci=parent['source_candidate'];candidate=packet['candidates'][ci];a,b=packet['offsets'][ci:ci+2];s=np.array(candidate['station_s_mm']);p=np.array(candidate['stations']);prob=events[a:b];points=np.array(parent['points']);length=arclength(points)[-1]
        _,indices=cKDTree(p).query(points[[0,-1]]);lo,hi=sorted(s[indices]);cuts=[];start=0
        while start<len(prob):
            if prob[start]<threshold:start+=1;continue
            end=start+1
            while end<len(prob)and prob[end]>=threshold:end+=1
            # Clusters touching a published outer endpoint are the same event,
            # not a new short weld. No minimum weld-length deletion is used.
            if s[start]>lo+1. and s[end-1]<hi-1.:
                peak=np.flatnonzero(prob[start:end]>=prob[start:end].max()-1e-10)+start;where=float((s[peak[0]]+s[peak[-1]])/2-lo)
                if 1e-6<where<length-1e-6:cuts.append(where)
            start=end
        bounds=[0.]+sorted(cuts)+[float(length)]
        for begin,end in zip(bounds[:-1],bounds[1:]):
            pp=slice_curve(points,begin,end);parents.append(publish_piece(parent,pp,len(parents)))
    return dict(original,parents=parents,geometry_units=len(parents),published_segments=sum(len(p['segments'])for p in parents),boundary_model=True)
