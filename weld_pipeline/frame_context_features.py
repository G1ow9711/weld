"""Point-only cross-surface extents at local, broad and whole-plane scales."""
import numpy as np
from weld_pipeline.indexed_surface_features import PlaneIndex
from weld_pipeline.local_height_features import window_extrema

FEATURE_VERSION='typed_contact_surface_context_v1'
WINDOWS_MM=(35.,300.)


def append_frame_context(xyz,sequences):
    index=PlaneIndex(xyz);cache={};output=[];valid_count=0
    for seq in sequences:
        base=seq['features'];q=seq['xyz'].astype(float)
        if base.shape[1]!=84:raise ValueError('Expected 84 original features')
        u=np.asarray(seq['direction'],float)
        n1,n2=[base[len(q)//2,a:b].astype(float) for a,b in [(60,63),(63,66)]]
        n1/=np.linalg.norm(n1);n2/=np.linalg.norm(n2)
        arrays=[]
        for normal,cross in [(n1,n2),(n2,n1)]:
            offset=float(q[0]@normal)
            if normal[np.argmax(np.abs(normal))]<0:normal=-normal;offset=-offset
            key=tuple(np.round(normal,9))+(round(offset,4),)
            if key not in cache:cache[key]=index.select(normal,offset)
            p=cache[key];cross_origin=q@cross
            along=p@u;across=p@cross;query=q@u
            for radius in (*WINDOWS_MM,None):
                f=np.zeros((len(q),3),np.float32)
                if len(p):
                    if radius is None:
                        low=np.full(len(q),across.min());high=np.full(len(q),across.max());count=np.full(len(q),len(p))
                    else:low,high,count=window_extrema(along,across,query,radius)
                    valid=count>0
                    f[valid,0]=(cross_origin[valid]-low[valid])/1000.
                    f[valid,1]=(high[valid]-cross_origin[valid])/1000.
                    f[valid,2]=1.;valid_count+=int(valid.sum())
                arrays.append(f)
        output.append({**seq,'features':np.column_stack([base,*arrays]).astype(np.float32)})
    return output,{'feature_version':FEATURE_VERSION,'channels':102,'windows_mm':list(WINDOWS_MM),
        'planes':len(cache),'valid_scale_samples':valid_count,
        'scope':'Observed extents in each contact-plane cross direction. Includes disconnected coplanar surfaces; not a claimed plate segmentation or exact customer process measurement.'}
