"""Exact range-query acceleration of frame_context_features.py."""
import numpy as np
from weld_pipeline.indexed_surface_features import PlaneIndex
from weld_pipeline.frame_context_features import FEATURE_VERSION,WINDOWS_MM
from weld_pipeline.range_extrema import RangeExtrema


def append_frame_context_fast(xyz,sequences):
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
            ranges=RangeExtrema(along,across)
            for radius in (*WINDOWS_MM,None):
                f=np.zeros((len(q),3),np.float32)
                if len(p):
                    if radius is None:
                        low=np.full(len(q),across.min());high=np.full(len(q),across.max());count=np.full(len(q),len(p))
                    else:low,high,count=ranges.query(query,radius)
                    valid=count>0
                    f[valid,0]=(cross_origin[valid]-low[valid])/1000.
                    f[valid,1]=(high[valid]-cross_origin[valid])/1000.
                    f[valid,2]=1.;valid_count+=int(valid.sum())
                arrays.append(f)
        output.append({**seq,'features':np.column_stack([base,*arrays]).astype(np.float32)})
    return output,{'feature_version':FEATURE_VERSION,'channels':102,'windows_mm':list(WINDOWS_MM),
        'planes':len(cache),'valid_scale_samples':valid_count,'algorithm':'exact_block_range_extrema',
        'scope':'Observed coplanar extents; not exact process dimensions.'}
