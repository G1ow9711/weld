"""Direct continuous center modes with spacing adapted to vote evidence.

No fixed millimeter peak spacing, minimum weld length, or opposite-role merge.
The tiny coordinate quantization is derived from float32 offset precision.
"""
import numpy as np

from weld_pipeline.near_event_votes import RADIUS_MM

VERSION = 'direct_continuous_center_modes_v1'
RESOLUTION_MM = 8.*RADIUS_MM*np.finfo(np.float32).eps


def center_modes(support, offsets, step, threshold=.5):
    support=np.asarray(support,float);offsets=np.asarray(offsets,float)
    if (support.shape!=offsets.shape or support.ndim!=2 or support.shape[1]!=3
            or not np.isfinite(support).all() or not np.isfinite(offsets).all() or step<=0.):
        raise ValueError('Three finite typed support and physical-offset fields required')
    locations=np.arange(len(support),dtype=float)*step;span=locations[-1]
    result=[]
    for role in range(3):
        positions=locations+offsets[:,role]*RADIUS_MM
        known=(support[:,role]>=threshold)&(np.abs(offsets[:,role])<=1.+1e-6)&(positions>=-RESOLUTION_MM)&(positions<=span+RESOLUTION_MM)
        centers=positions[known];weights=np.clip(support[known,role],0.,1.)
        if not len(centers):result.append([]);continue
        order=np.argsort(centers,kind='stable');centers=centers[order];weights=weights[order]
        gaps=np.diff(centers);breaks=[]
        for i,gap in enumerate(gaps):
            # Exclude the candidate gap itself when estimating its local scale.
            neighbors=np.r_[gaps[max(0,i-8):i],gaps[i+1:min(len(gaps),i+9)]]
            scale=float(np.median(neighbors)) if len(neighbors) else 0.
            if gap>max(RESOLUTION_MM,8.*scale):breaks.append(i+1)
        divisions=np.r_[0,breaks,len(centers)].astype(int);events=[]
        for a,b in zip(divisions[:-1],divisions[1:]):
            weight=weights[a:b];center=float(np.average(centers[a:b],weights=weight))
            center=float(np.clip(np.rint(center/RESOLUTION_MM)*RESOLUTION_MM,0.,span))
            confidence=float(-np.expm1(np.log1p(-np.clip(weight,0.,1.-1e-7)).sum()))
            if confidence>=threshold:events.append([center,min(confidence,1.-1e-4)])
        result.append(events)
    return result


def restore_event_order(foreground, events, span, transform):
    foreground=np.asarray(foreground,np.float32)
    if not transform['reverse']:return foreground.copy(),events
    restored=[sorted([[span-p,c] for p,c in events[role]]) for role in (1,0,2)]
    return foreground[::-1].copy(),restored
