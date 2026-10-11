"""CAD surface proposals with 3D support checks, from XYZ alone.

This experimental generator assumes the provided nominal Z/gravity frame and
an orthogonal dominant plate frame. It never reads customer reference curves.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def dominant_xy_frame(xyz):
    z,counts=np.unique(np.round(xyz[:,2],2),return_counts=True)
    major=z[np.argsort(counts)[-2:]]
    q=xyz[~np.isin(np.round(xyz[:,2],2),major),:2]
    if len(q)<20:q=xyz[:,:2]
    q=np.unique(np.round(q,3),axis=0)
    if len(q)>50000:q=q[np.random.default_rng(17).choice(len(q),50000,replace=False)]
    distances,ids=cKDTree(q).query(q,k=min(4,len(q)),workers=2)
    delta=q[ids[:,1:]]-q[:,None]
    length=distances[:,1:]
    good=(length>2)&(length<50)
    angle=np.arctan2(delta[:,:,1][good],delta[:,:,0][good])%(np.pi/2)
    bins=1800
    hist=np.bincount(np.minimum((angle/(np.pi/2)*bins).astype(int),bins-1),minlength=bins)
    peak=np.argmax(gaussian_filter1d(hist.astype(float),1,mode='wrap'))
    center=(peak+.5)/bins*(np.pi/2)
    residual=(angle-center+np.pi/4)%(np.pi/2)-np.pi/4
    selected=angle[np.abs(residual)<np.deg2rad(.2)]
    theta=np.angle(np.mean(np.exp(4j*selected)))/4
    return np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]],float)


def vertical_surface_planes(xyz):
    frame=dominant_xy_frame(xyz);origin=xyz[:,:2].mean(0)
    z,counts=np.unique(np.round(xyz[:,2],2),return_counts=True)
    major=z[np.argsort(counts)[-2:]]
    nonbase=~np.isin(np.round(xyz[:,2],2),major)
    planes=[];rejected=[]
    for initial in frame:
        value=(xyz[:,0]-origin[0])*initial[0]+(xyz[:,1]-origin[1])*initial[1]
        bin_width=.5;minimum=np.floor(value.min()/bin_width)*bin_width
        bins=np.round((value-minimum)/bin_width).astype(int)
        histogram=np.bincount(bins[nonbase],minlength=int(bins.max())+1)
        smooth=gaussian_filter1d(histogram.astype(float),.7)
        peaks,_=find_peaks(np.r_[0,smooth,0],height=8,prominence=4,distance=3);peaks=peaks-1
        for peak in peaks[np.argsort(smooth[peaks])[::-1]]:
            estimate=minimum+peak*bin_width
            preliminary=np.abs(value-estimate)<1.0
            if preliminary.sum()<12:continue
            q=xyz[preliminary]
            xy=q[:,:2];center=xy.mean(0)
            _,_,vt=np.linalg.svd(xy-center,full_matrices=False)
            normal=vt[-1]
            if abs(normal@initial)<.995:continue
            distance=(xyz[:,0]-center[0])*normal[0]+(xyz[:,1]-center[1])*normal[1]
            ix=np.flatnonzero(np.abs(distance)<.12)
            if len(ix)<12:continue
            # Refit exact CAD plane after excluding histogram-band outliers.
            xy=xyz[ix,:2];center=xy.mean(0)
            _,_,vt=np.linalg.svd(xy-center,full_matrices=False);normal=vt[-1]
            if normal@initial<0:normal=-normal
            offset=float(normal@center);direction=np.array([-normal[1],normal[0]])
            if direction[np.argmax(np.abs(direction))]<0:direction=-direction
            distance=xyz[:,0]*normal[0]+xyz[:,1]*normal[1]-offset
            ix=np.flatnonzero(np.abs(distance)<.12);points=xyz[ix]
            zz,nz=np.unique(np.round(points[:,2],2),return_counts=True)
            discard=zz[np.argsort(nz)[-2:]]
            evidence=points[~np.isin(np.round(points[:,2],2),discard)]
            if len(evidence)<9 or np.ptp(points[:,2])<10:continue
            t=evidence[:,:2]@direction
            span=np.ptp(t);height=np.ptp(evidence[:,2])
            if span<3 or height<10:continue
            # A face occupies area in (t,z). A horizontal mesh-grid row plus a
            # few crossing edges fails this coverage test after dense Z levels
            # are removed; this suppresses the previous projected-grid artifact.
            cells=np.column_stack([np.floor((t-t.min())/30),np.floor((evidence[:,2]-evidence[:,2].min())/30)]).astype(int)
            occupied=len(np.unique(cells,axis=0));total=(int(span/30)+1)*(int(height/30)+1)
            fill=occupied/max(total,1)
            if fill<.10:
                rejected.append({'normal':normal.tolist(),'offset':offset,'fill':fill,'count':len(points)})
                continue
            if any(abs(normal@np.array(p['normal_xy']))>.99999 and abs(offset-p['offset_mm'])<.3 for p in planes):continue
            full_t=points[:,:2]@direction
            planes.append({'normal_xy':normal.tolist(),'direction_xy':direction.tolist(),'offset_mm':offset,
                'range_mm':[float(full_t.min()),float(full_t.max())],
                'z_range_mm':[float(points[:,2].min()),float(points[:,2].max())],
                'support_count':len(points),'vertical_area_occupancy':float(fill)})
    return planes,{'xy_frame':frame.tolist(),'rejected_grid_planes':len(rejected),'rejected_examples':rejected[:20]}


def surface_candidates(xyz):
    planes,meta=vertical_surface_planes(xyz)
    z,counts=np.unique(np.round(xyz[:,2],2),return_counts=True)
    levels=z[counts>=max(50,len(xyz)*.004)]
    candidates=[];line_records=[]
    for base in levels:
        near=xyz[np.abs(xyz[:,2]-base)<.015]
        for pi,plane in enumerate(planes):
            # No extension through empty space beyond the observed plane extent.
            if not plane['z_range_mm'][0]-.05<=base<=plane['z_range_mm'][1]+.05:continue
            normal=np.array(plane['normal_xy']);direction=np.array(plane['direction_xy'])
            line=near[np.abs(near[:,0]*normal[0]+near[:,1]*normal[1]-plane['offset_mm'])<.12]
            if len(line)<3:continue
            t=np.unique(np.round(line[:,:2]@direction,3));delta=np.diff(t)
            step=np.median(delta[delta>1]) if np.any(delta>1) else 8.
            breaks=np.r_[0,np.flatnonzero(delta>max(15,1.8*step))+1,len(t)]
            for a,b in zip(breaks[:-1],breaks[1:]):
                if b-a<3 or t[b-1]-t[a]<5:continue
                xy0=normal*plane['offset_mm']+direction*t[a];xy1=normal*plane['offset_mm']+direction*t[b-1]
                candidates.append({'id':f'surface_h_{len(candidates)+1:04d}','type':'horizontal',
                    'segments':[{'kind':'line','points':[[*xy0,float(base)],[*xy1,float(base)]]}],
                    'plane_id':pi,'surface_z_mm':float(base)})
                line_records.append({'plane_id':pi,'z_mm':float(base),'range_mm':[float(t[a]),float(t[b-1])]})
    tree=cKDTree(xyz[:,:2]);seen=set()
    for i,a in enumerate(planes):
        for b in planes[i+1:]:
            matrix=np.array([a['normal_xy'],b['normal_xy']])
            if abs(np.linalg.det(matrix))<.5:continue
            xy=np.linalg.solve(matrix,[a['offset_mm'],b['offset_mm']])
            if not all(p['range_mm'][0]-2<=xy@np.array(p['direction_xy'])<=p['range_mm'][1]+2 for p in [a,b]):continue
            key=tuple(np.round(xy/.1).astype(int))
            if key in seen:continue
            seen.add(key);ix=tree.query_ball_point(xy,.2)
            if len(ix)<4:continue
            zz=np.unique(np.round(xyz[ix,2],3))
            if len(zz)<4 or np.ptp(zz)<15:continue
            lo=max(a['z_range_mm'][0],b['z_range_mm'][0],zz.min())
            hi=min(a['z_range_mm'][1],b['z_range_mm'][1],zz.max())
            if hi-lo<15:continue
            candidates.append({'id':f'surface_v_{len(candidates)+1:04d}','type':'vertical',
                'segments':[{'kind':'line','points':[[*xy,lo],[*xy,hi]]}],
                'plane_ids':[i,planes.index(b)]})
    return candidates,{**meta,'vertical_planes':planes,'horizontal_levels':levels.tolist(),
        'candidate_count':len(candidates),'horizontal_lines':line_records}
