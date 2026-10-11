"""Additional vertical contact candidates derived from point-only planes."""
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.geometry_candidates import horizontal_candidates


def vertical_ridges(xyz):
    """Candidate ridges from repeated vertical CAD columns, never labels.

    Surface interiors remain hard negatives for learning; this is a proposal
    stage, not a rule declaring every vertical column to be a weld.
    """
    key=np.round(xyz[:,:2]/.02).astype(np.int64)
    keys,first,inverse,counts=np.unique(key,axis=0,return_index=True,return_inverse=True,return_counts=True)
    low=np.full(len(keys),np.inf);high=np.full(len(keys),-np.inf)
    np.minimum.at(low,inverse,xyz[:,2]);np.maximum.at(high,inverse,xyz[:,2])
    keep=np.flatnonzero((counts>=4)&(high-low>=20))
    centers=xyz[first[keep],:2]
    if len(centers)<3:return []
    tree=cKDTree(centers)
    distances,neighbors=tree.query(centers,k=min(12,len(centers)),workers=2)
    output=[]
    for ci,index in enumerate(keep):
        nearby=centers[neighbors[ci][distances[ci]<=20.]]
        if len(nearby)<3:continue
        eig=np.linalg.eigvalsh(np.cov(nearby.T))
        if eig[0]/max(eig[1],1e-9)<.025:continue
        z=np.unique(np.round(xyz[inverse==index,2],3))
        if len(z)<4:continue
        output.append({'id':f'ridge_candidate_{len(output)+1:04d}','type':'vertical',
            'segments':[{'kind':'line','points':[[*centers[ci],float(z[0])],[*centers[ci],float(z[-1])]]}],
            'point_support':len(z),'proposal_source':'repeated_vertical_columns_with_noncollinear_local_support'})
    return output


def contact_candidates(xyz,include_ridges=False):
    horizontal,meta=horizontal_candidates(xyz)
    planes=meta['plane_lines'];tree=cKDTree(xyz[:,:2]);seen=set();vertical=[]
    for i,a in enumerate(planes):
        na=np.asarray(a['normal_xy']);da=np.asarray(a['direction_xy'])
        for b in planes[i+1:]:
            nb=np.asarray(b['normal_xy']);db=np.asarray(b['direction_xy'])
            matrix=np.stack([na,nb])
            if abs(np.linalg.det(matrix))<.5:continue
            xy=np.linalg.solve(matrix,[a['offset_mm'],b['offset_mm']])
            ta,tb=xy@da,xy@db
            if not(a['range_mm'][0]-5<=ta<=a['range_mm'][1]+5 and b['range_mm'][0]-5<=tb<=b['range_mm'][1]+5):continue
            key=tuple(np.round(xy/.05).astype(int))
            if key in seen:continue
            seen.add(key)
            indices=tree.query_ball_point(xy,.15)
            if len(indices)<4:continue
            z=np.unique(np.round(xyz[indices,2],3));delta=np.diff(z)
            if len(z)<4 or np.ptp(z)<20:continue
            typical=np.median(delta[delta>1]) if np.any(delta>1) else 8.
            breaks=np.r_[0,np.flatnonzero(delta>max(20.,typical*2.2))+1,len(z)]
            for start,stop in zip(breaks[:-1],breaks[1:]):
                if stop-start<4 or z[stop-1]-z[start]<20:continue
                vertical.append({'id':f'vertical_candidate_{len(vertical)+1:04d}','type':'vertical',
                    'segments':[{'kind':'line','points':[[*xy,float(z[start])],[*xy,float(z[stop-1])]]}],
                    'cross_section_basis':[na.tolist(),nb.tolist()],'point_support':int(stop-start)})
    if include_ridges:
        old_xy=np.array([c['segments'][0]['points'][0][:2] for c in vertical])
        for c in vertical_ridges(xyz):
            xy=np.array(c['segments'][0]['points'][0][:2])
            if not len(old_xy) or np.linalg.norm(old_xy-xy,axis=1).min()>.15:vertical.append(c)
    return horizontal+vertical,{**meta,'vertical_candidate_count':len(vertical)}
