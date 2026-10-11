"""Point-only observed height context; not an exact customer plate-height label."""
from collections import deque
import numpy as np
from weld_pipeline.indexed_surface_features import PlaneIndex


def window_extrema(position,height,queries,half_width=10.):
    """Exact inclusive window min/max/count for sorted or unsorted samples."""
    position=np.asarray(position,float);height=np.asarray(height,float);queries=np.asarray(queries,float)
    if len(position)!=len(height) or np.any(np.diff(queries)<0):raise ValueError('Invalid window input')
    if half_width<=0:raise ValueError('Positive half width required')
    order=np.argsort(position,kind='stable');t=position[order];z=height[order]
    minima=deque();maxima=deque();left=right=0
    low=np.full(len(queries),np.nan);high=low.copy();counts=np.zeros(len(queries),int)
    for i,q in enumerate(queries):
        while right<len(t) and t[right]<=q+half_width:
            while minima and z[minima[-1]]>=z[right]:minima.pop()
            while maxima and z[maxima[-1]]<=z[right]:maxima.pop()
            minima.append(right);maxima.append(right);right+=1
        while left<right and t[left]<q-half_width:left+=1
        while minima and minima[0]<left:minima.popleft()
        while maxima and maxima[0]<left:maxima.popleft()
        counts[i]=right-left
        if counts[i]:low[i]=z[minima[0]];high[i]=z[maxima[0]]
    return low,high,counts


def append_local_height_context(xyz,sequences,half_width_mm=10.):
    index=PlaneIndex(xyz);output=[];total_valid=0
    for seq in sequences:
        base=seq['features'];q=seq['xyz'].astype(float)
        if base.shape[1]!=84:raise ValueError('Expected original 84 features')
        context=np.zeros((len(q),4),np.float32)
        if seq['type']=='horizontal':
            normal=base[len(q)//2,60:63].astype(float);normal/=np.linalg.norm(normal)
            points=index.select(normal,float(q[0]@normal))
            direction=np.asarray(seq['direction'],float)
            lo,hi,count=window_extrema(points@direction,points[:,2],q@direction,half_width_mm)
            valid=count>0
            context[valid,0]=(hi[valid]-q[valid,2])/1000.
            context[valid,1]=(q[valid,2]-lo[valid])/1000.
            context[valid,2]=np.log1p(count[valid])/10.
            context[valid,3]=1.
            total_valid+=int(valid.sum())
        output.append({**seq,'features':np.column_stack([base,context]).astype(np.float32)})
    return output,{'feature_version':'surface_height_context_prototype_v1','channels':88,
                   'half_width_mm':half_width_mm,'valid_horizontal_samples':total_valid,
                   'scope':'Observed local coplanar Z extents, not exact plate height or process clearance. Not integrated into trained predictor.'}
