"""Numerical coincidence groups, independent of weld length/segmentation rules."""
import numpy as np


def coincident_event_groups(positions,tolerance_mm=1e-6):
    """Connected runs merge only when their entire diameter <= tolerance.

    Long chains of small but positive gaps remain separate, avoiding transitive
    extension. 1e-6mm matches the existing strict circle-fit numerical tolerance;
    it is not a minimum weld length or the customer 35mm clearance rule.
    """
    q=np.asarray(positions,float)
    if q.ndim!=1 or not np.isfinite(q).all() or np.any(np.diff(q)<0):raise ValueError('Finite ordered positions required')
    if not np.isfinite(tolerance_mm) or tolerance_mm<=0:raise ValueError('Positive tolerance required')
    if not len(q):return np.zeros(1,np.int64),np.empty(0)
    boundaries=np.r_[0,np.flatnonzero(np.diff(q)>tolerance_mm)+1,len(q)];offsets=[0];centers=[]
    for a,b in zip(boundaries[:-1],boundaries[1:]):
        if q[b-1]-q[a]<=tolerance_mm:
            centers.append(float(q[a]+np.mean(q[a:b]-q[a])));offsets.append(int(b))
        else:
            centers.extend(q[a:b].tolist());offsets.extend(range(int(a)+1,int(b)+1))
    return np.asarray(offsets,np.int64),np.asarray(centers)
