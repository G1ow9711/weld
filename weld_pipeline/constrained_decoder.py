"""Global, training-calibrated straight-weld peak decoding."""
import numpy as np
from scipy.signal import find_peaks


def refined_peaks(heat,step,threshold,distance_mm=4.):
    indices,_=find_peaks(heat,height=threshold,
                         distance=max(1,int(np.ceil(distance_mm/step))),prominence=.05)
    values=[]
    for i in indices:
        a,b,c=np.log(np.clip(heat[i-1:i+2],1e-7,1.))
        denom=a-2*b+c
        offset=np.clip(.5*(a-c)/denom,-.5,.5) if abs(denom)>1e-8 else 0.
        values.append((float(i+offset),float(heat[i])))
    return values


def decode_constrained(sequences,probabilities,endpoint_threshold=.5,internal_threshold=.5,
                       minimum_segment_mm=1.5,internal_distance_mm=4.):
    result=[]
    for seq,prob in zip(sequences,probabilities):
        step=seq['sample_step_mm']
        starts=refined_peaks(prob[:,1],step,endpoint_threshold)
        ends=refined_peaks(prob[:,2],step,endpoint_threshold)
        internal=refined_peaks(prob[:,3],step,internal_threshold)
        used=set()
        for a,_ in starts:
            possible=[(i,b) for i,(b,_) in enumerate(ends) if i not in used and b-a>3/step]
            if not possible:continue
            i,b=min(possible,key=lambda v:v[1]);used.add(i)
            mask=prob[int(np.ceil(a)):int(np.floor(b))+1,0]
            if not len(mask) or mask.mean()<.5:continue
            eligible=[(c,h) for c,h in internal if a+minimum_segment_mm/step<c<b-minimum_segment_mm/step]
            selected=[]
            for c,h in sorted(eligible,key=lambda v:(-v[1],v[0])):
                if all(abs(c-q)*step>=max(minimum_segment_mm,internal_distance_mm) for q in selected):
                    selected.append(c)
            divisions=[a]+sorted(selected)+[b]
            origin=seq['xyz'][0].astype(float);direction=seq['direction'].astype(float)*step
            segments=[{'kind':'line','points':[(origin+direction*x).tolist(),(origin+direction*y).tolist()]}
                      for x,y in zip(divisions[:-1],divisions[1:])]
            result.append({'id':f'weld_{len(result)+1:04d}','type':seq['type'],'segments':segments,
                           'confidence':float(min(prob[round(a),1],prob[round(b),2]))})
    return result
