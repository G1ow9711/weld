"""Pairing that respects a predicted endpoint followed by local background."""
import bisect
import numpy as np
from weld_pipeline.constrained_decoder import refined_peaks
from weld_pipeline.curved_sequences import arc_points


def integral(values):
    return np.r_[0.,np.cumsum((values[:-1]+values[1:])*.5)]


def event_peaks(heat,step,threshold):
    result=[]
    for position,height in refined_peaks(heat,step,threshold):
        i=int(round(position));offset=position-i
        a,b,c=np.log(np.clip(heat[i-1:i+2].astype(float),1e-7,1.))
        # Labels are Gaussian heatmaps: score the interpolated peak amplitude,
        # not a sample that may sit halfway between the actual peak and grid.
        amplitude=float(np.clip(np.exp(b-.25*(a-c)*offset),1e-4,1-1e-4))
        result.append([position,amplitude])
    return result


def select_intervals(prob,step,threshold=.5,weld_penalty=4.,mask_weight=.5,event_gap_mm=0.):
    if not np.isfinite(event_gap_mm) or event_gap_mm<0:
        raise ValueError('Background gap limit must be finite and nonnegative; zero disables the constraint')
    starts=event_peaks(prob[:,1],step,threshold)
    ends=event_peaks(prob[:,2],step,threshold)
    if not starts or not ends:return []
    # Match mutual nearest start/end peaks representing the same physical cut.
    # Small opposite subpixel shifts must not make adjacent welds overlap.
    start_positions=np.array([a for a,_ in starts]);end_positions=np.array([b for b,_ in ends])
    distances=abs(start_positions[:,None]-end_positions[None,:])
    for i,start in enumerate(starts):
        j=int(distances[i].argmin())
        if int(distances[:,j].argmin())==i and distances[i,j]*step<=1.5:
            position=(start[0]*start[1]+ends[j][0]*ends[j][1])/(start[1]+ends[j][1])
            start[0]=position;ends[j][0]=position
    mask=np.clip(prob[:,0].astype(float),1e-4,1-1e-4)
    transitions=np.diff(np.r_[False,mask<.5,False].astype(np.int8))
    gap_starts=np.flatnonzero(transitions==1)-.5
    gap_ends=np.flatnonzero(transitions==-1)-.5
    prefix=integral(np.log(mask)-np.log1p(-mask));mass=integral(mask);grid=np.arange(len(mask))
    candidates=[]
    logit=lambda p:float(np.log(np.clip(p,1e-4,1-1e-4))-np.log1p(-np.clip(p,1e-4,1-1e-4)))
    for a,pa in starts:
        for b,pb in ends:
            if (b-a)*step<=3.:continue
            if event_gap_mm and len(gap_starts):
                blocked=False
                # Require a learned endpoint AND a neighboring background run.
                # A dip alone neither rejects a weld nor creates a breakpoint.
                for endpoint,_ in ends:
                    if not a+1.5/step<endpoint<b-1.5/step:continue
                    overlap=np.maximum(0.,np.minimum(gap_ends,b-1.5/step)-np.maximum(gap_starts,endpoint))
                    adjacent=(gap_starts-endpoint)*step
                    if np.any((adjacent>=-1.5)&(adjacent<=4.)&(overlap*step>event_gap_mm)):
                        blocked=True;break
                if not blocked:
                    for startpoint,_ in starts:
                        if not a+1.5/step<startpoint<b-1.5/step:continue
                        overlap=np.maximum(0.,np.minimum(gap_ends,startpoint)-np.maximum(gap_starts,a+1.5/step))
                        adjacent=(startpoint-gap_ends)*step
                        if np.any((adjacent>=-1.5)&(adjacent<=4.)&(overlap*step>event_gap_mm)):
                            blocked=True;break
                if blocked:continue
            mean=(np.interp(b,grid,mass)-np.interp(a,grid,mass))/(b-a)
            if mean<.5:continue
            foreground=(np.interp(b,grid,prefix)-np.interp(a,grid,prefix))*step
            weight=foreground*mask_weight+logit(pa)+logit(pb)-weld_penalty
            if weight>0:candidates.append((b,a,weight,pa,pb))
    candidates.sort();endpoints=[c[0] for c in candidates]
    best=np.zeros(len(candidates)+1);previous=[];take=[]
    for i,(b,a,weight,pa,pb) in enumerate(candidates):
        prior=bisect.bisect_right(endpoints,a+1e-9,0,i)
        selected=best[prior]+weight>best[i]
        best[i+1]=best[prior]+weight if selected else best[i]
        previous.append(prior);take.append(selected)
    result=[];i=len(candidates)
    while i:
        if take[i-1]:
            b,a,weight,pa,pb=candidates[i-1];result.append((a,b,pa,pb));i=previous[i-1]
        else:i-=1
    return result[::-1]


def decode_intervals(sequences,probabilities,threshold=.5,weld_penalty=4.,mask_weight=.5,event_gap_mm=0.):
    result=[]
    for seq,prob in zip(sequences,probabilities):
        step=seq['sample_step_mm'];internal=refined_peaks(prob[:,3],step,threshold)
        for a,b,pa,pb in select_intervals(prob,step,threshold,weld_penalty,mask_weight,event_gap_mm):
            cuts=[a]+[p for p,h in internal if a+1.5/step<p<b-1.5/step]+[b];segments=[]
            for u,v in zip(cuts[:-1],cuts[1:]):
                if seq.get('curve_kind')=='arc':
                    t=np.linspace(u*step,v*step,max(2,int(np.ceil((v-u)*step))+1))
                    segments.append({'kind':'arc','points':arc_points(seq,t).tolist(),'length_mm':float((v-u)*step)})
                else:
                    q=seq['xyz'][0].astype(float);direction=seq['direction']*step
                    segments.append({'kind':'line','points':[(q+u*direction).tolist(),(q+v*direction).tolist()]})
            result.append({'id':f'weld_{len(result)+1:04d}','type':seq['type'],'segments':segments,'confidence':float(min(pa,pb))})
    return result
