"""Point-only contact-line experiment, v0. Not a complete weld predictor.

Current scope: horizontal base plane with approximately vertical planar webs.
It deliberately emits candidates without any weld labels or learned selection.
"""
import numpy as np
from scipy.spatial import cKDTree


def support_lines(xy,min_support=6,tolerance=.15):
    origin=xy.mean(0)
    xy=np.unique(np.round(xy-origin,3),axis=0)
    if len(xy)<min_support:return []
    tree=cKDTree(xy)
    _,ix=tree.query(xy,k=min(10,len(xy)),workers=2)
    a=np.repeat(np.arange(len(xy)),ix.shape[1]-1);b=ix[:,1:].reshape(-1)
    delta=xy[b]-xy[a];distance=np.linalg.norm(delta,axis=1)
    good=(distance>4)&(distance<60)&(a<b)
    a,b,delta=a[good],b[good],delta[good]
    angles=(np.arctan2(delta[:,1],delta[:,0])+np.pi/2)%np.pi
    normals=np.column_stack([np.cos(angles),np.sin(angles)])
    offset=(normals*xy[a]).sum(1)
    keys=np.column_stack([np.round(angles/(np.pi/360)),np.round(offset)])
    unique,inverse,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
    order=np.argsort(counts)[::-1]
    remaining=np.ones(len(xy),bool);lines=[]
    for group in order:
        pairs=inverse==group
        seed=np.unique(np.r_[a[pairs],b[pairs]])
        seed=seed[remaining[seed]]
        if len(seed)<min_support:continue
        points=xy[seed]
        center=points.mean(0)
        _,_,vt=np.linalg.svd(points-center,full_matrices=False)
        normal=vt[-1];offset=float(normal@center)
        inliers=(np.abs(xy@normal-offset)<=tolerance)&remaining
        if inliers.sum()<min_support:continue
        points=xy[inliers];center=points.mean(0)
        _,_,vt=np.linalg.svd(points-center,full_matrices=False)
        normal=vt[-1];offset=float(normal@center)
        inliers=(np.abs(xy@normal-offset)<=tolerance)&remaining
        if inliers.sum()<min_support:continue
        remaining[inliers]=False
        direction=np.array([-normal[1],normal[0]])
        t=(xy[inliers]+origin)@direction
        lines.append({'normal_xy':normal.tolist(),'offset_mm':float(offset+normal@origin),
            'direction_xy':direction.tolist(),'support_count':int(inliers.sum()),
            'range_mm':[float(t.min()),float(t.max())]})
    return lines


def horizontal_candidates(xyz,band_depth_mm=50.,surface_tolerance_mm=.15):
    # First experiment only: dominant axis-aligned z-plane, inferred from points.
    z,counts=np.unique(np.round(xyz[:,2],2),return_counts=True)
    base=float(z[np.argmax(counts)])
    # The slab's opposite face can outnumber the web samples. In this prototype
    # the side with the larger physical depth is the web-bearing side.
    def band(sign):
        depth=(xyz[:,2]-base)*sign
        q=xyz[(depth>1)&(depth<band_depth_mm)]
        return q
    lower,upper=band(-1),band(1)
    sign=-1 if base-xyz[:,2].min()>=xyz[:,2].max()-base else 1
    q=lower if sign==-1 else upper
    lines=support_lines(q[:,:2],tolerance=surface_tolerance_mm)
    candidates=[]
    # Limit support to the immediate contact band; holes remain visible as gaps.
    depth=(xyz[:,2]-base)*sign
    # CAD face boundaries are sampled exactly on the contact plane. Interior
    # web samples may start tens of mm below it and bridge a hole incorrectly.
    near=xyz[np.abs(depth)<.015]
    for li,line in enumerate(lines):
        normal=np.asarray(line['normal_xy']);direction=np.asarray(line['direction_xy'])
        near_line=near[np.abs(near[:,:2]@normal-line['offset_mm'])<=surface_tolerance_mm]
        if len(near_line)<4:continue
        t=np.unique(np.round(near_line[:,:2]@direction,3))
        delta=np.diff(t)
        typical=np.median(delta[delta>1]) if np.any(delta>1) else 8.
        gap=max(15.,typical*1.8)
        breaks=np.r_[0,np.flatnonzero(delta>gap)+1,len(t)]
        for a,b in zip(breaks[:-1],breaks[1:]):
            if b-a<2 or t[b-1]-t[a]<5:continue
            xy0=normal*line['offset_mm']+direction*t[a]
            xy1=normal*line['offset_mm']+direction*t[b-1]
            candidates.append({'id':f'candidate_{len(candidates)+1:04d}','type':'horizontal',
                'segments':[{'kind':'line','points':[[*xy0,base],[*xy1,base]]}],
                'plane_id':li,'support_count':int(b-a),'gap_threshold_mm':float(gap)})
    return candidates,{'base_z_mm':base,'web_sign':sign,'band_points':len(q),'line_count':len(lines),
        'candidate_count':len(candidates),'plane_lines':lines,
        'scope':'Unselected geometric candidates; horizontal planar contact prototype only.'}
