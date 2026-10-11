"""Equivalent plane proposals with eight-thread nearest-neighbor queries.

Input is raw XYZ in mm; no reference files or learned labels are accessed here.
The provided nominal gravity frame is retained. Dominant plate directions are
estimated from point samples rather than stored per workpiece.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from weld_pipeline.point_frame import dominant_xy_frame
from weld_pipeline.geometry_candidates import support_lines


def normal_directions(xyz,frame):
    """Recover secondary diagonal plate directions from exact CAD sample pairs."""
    z,count=np.unique(xyz[:,2].round(2),return_counts=True)
    main=z[np.argsort(count)[-2:]]
    q=np.unique(np.round(xyz[~np.isin(xyz[:,2].round(2),main),:2],3),axis=0)
    if len(q)<4:return frame
    if len(q)>200000:q=q[np.random.default_rng(29).choice(len(q),200000,False)]
    distance,ids=cKDTree(q).query(q,k=min(6,len(q)),workers=8)
    delta=q[ids[:,1:]]-q[:,None];good=(distance[:,1:]>4)&(distance[:,1:]<60)
    angles=(np.arctan2(delta[:,:,1][good],delta[:,:,0][good])+np.pi/2)%np.pi
    bins=1800;hist=np.bincount(np.minimum((angles/np.pi*bins).astype(int),bins-1),minlength=bins)
    smooth=gaussian_filter1d(hist.astype(float),1,mode='wrap')
    padded=np.r_[smooth[-10:],smooth,smooth[:10]]
    peaks,_=find_peaks(padded,height=max(10,float(smooth.max())*.0003),prominence=8,distance=4)
    peaks=peaks[(peaks>=10)&(peaks<bins+10)]-10
    normals=[n for n in frame]
    for peak in peaks[np.argsort(smooth[peaks])[::-1]]:
        center=(peak+.5)/bins*np.pi
        residual=(angles-center+np.pi/2)%np.pi-np.pi/2
        subset=angles[np.abs(residual)<np.deg2rad(.2)]
        if not len(subset):continue
        angle=np.angle(np.exp(2j*subset).mean())/2
        normal=np.array([np.cos(angle),np.sin(angle)])
        if any(abs(normal@n)>.99995 for n in normals):continue
        normals.append(normal)
        if len(normals)>=24:break
    return np.array(normals)


def vertical_edge_anchors(xyz):
    """XY columns with >=4 DISTINCT heights, not repeated face vertices."""
    xy=np.round(xyz[:,:2]/.01).astype(np.int64)
    xy-=xy.min(0)
    key=xy[:,0]*(int(xy[:,1].max())+1)+xy[:,1]
    z=np.round(xyz[:,2]/.01).astype(np.int64)
    order=np.lexsort((z,key));key=key[order];zz=z[order]
    starts=np.r_[0,np.flatnonzero(np.diff(key)!=0)+1]
    new_z=np.r_[True,(np.diff(zz)!=0)|(np.diff(key)!=0)]
    counts=np.add.reduceat(new_z.astype(int),starts)
    ends=np.r_[starts[1:]-1,len(order)-1]
    spans=(zz[ends]-zz[starts])*.01
    gaps=np.r_[0,np.maximum(0,np.diff(zz))].astype(float)*.01
    gaps[starts]=0
    largest_gap=np.maximum.reduceat(gaps,starts)
    keep=(counts>=6)&(spans>=50)&(largest_gap<=spans*.25)
    # Sample inside the column, away from horizontal slab corners.
    middles=(starts+ends)//2
    return xyz[order[middles[keep]]]


def vertical_surface_planes(xyz, max_samples=160000):
    frame=dominant_xy_frame(xyz)
    directions=frame
    rng=np.random.default_rng(173)
    ids=rng.choice(len(xyz),min(max_samples,len(xyz)),replace=False)
    selected=xyz[ids]
    tree=cKDTree(xyz)
    valid=np.zeros((len(selected),len(directions)),bool)
    secondary=np.zeros(len(selected),bool)
    for begin in range(0,len(selected),10000):
        q=selected[begin:begin+10000]
        _,neighbors=tree.query(q,k=min(64,len(xyz)),workers=8)
        delta=xyz[neighbors]-q[:,None]
        centered=delta-delta.mean(1,keepdims=True)
        covariance=np.einsum('nki,nkj->nij',centered,centered)/centered.shape[1]
        eigenvalues,eigenvectors=np.linalg.eigh(covariance)
        local_normal=eigenvectors[:,:,0]
        planar=eigenvalues[:,0]<.35*np.maximum(eigenvalues[:,1],1e-8)
        secondary[begin:begin+len(q)]=planar&(np.abs(local_normal[:,2])<.15)&(np.abs(local_normal[:,:2]@frame.T).max(1)<.996)
        for axis,normal in enumerate(directions):
            tangent=np.array([-normal[1],normal[0]])
            nn=delta[:,:,:2]@normal
            tt=delta[:,:,:2]@tangent
            zz=delta[:,:,2]
            coplanar=np.abs(nn)<.12
            off_row=coplanar&(np.abs(zz)>3.)
            count=off_row.sum(1)
            spread=np.max(np.where(off_row,tt,-1e6),axis=1)-np.min(np.where(off_row,tt,1e6),axis=1)
            z_span=np.max(np.where(coplanar,zz,-1e6),axis=1)-np.min(np.where(coplanar,zz,1e6),axis=1)
            aligned=np.abs(local_normal[:,:2]@normal)>.90
            valid[begin:begin+len(q),axis]=(count>=4)&(spread>=6)&(z_span>=10)&aligned&planar
    anchors=vertical_edge_anchors(xyz)
    anchor_valid=np.zeros((len(anchors),len(directions)),bool)
    for begin in range(0,len(anchors),10000):
        q=anchors[begin:begin+10000]
        _,neighbors=tree.query(q,k=min(32,len(xyz)),workers=8)
        relative=xyz[neighbors]-q[:,None];delta=relative-relative.mean(1,keepdims=True)
        ev,vec=np.linalg.eigh(np.einsum('nki,nkj->nij',delta,delta)/delta.shape[1])
        vertical_edge=(ev[:,1]<.12*np.maximum(ev[:,2],1e-8))&(np.abs(vec[:,2,2])>.90)
        for axis,n in enumerate(directions):
            side=relative[:,:,:2]@n
            boundary=((side.min(1)>-.12)|(side.max(1)<.12))&(np.abs(vec[:,2,0])<.25)
            anchor_valid[begin:begin+len(q),axis]=(np.abs(vec[:,:2,0]@n)>.75)|vertical_edge|boundary
    planes=[]
    origin=xyz[:,:2].mean(0)
    for axis,normal0 in enumerate(directions):
        q=np.concatenate([selected[valid[:,axis]],anchors[anchor_valid[:,axis]]])
        if len(q)<2:continue
        values=(q[:,:2]-origin)@normal0
        # Retain all supported coarse planes; the fine pass only adds missing faces.
        for scale,window,height,prominence,separation in [(2.,.7,1.5,1.,3),(5.,.25,1.,.5,2)]:
            minimum=np.floor(values.min()*scale)/scale
            bins=np.round((values-minimum)*scale).astype(int)
            hist=np.bincount(bins)
            smooth=gaussian_filter1d(hist.astype(float),.5)
            peaks,_=find_peaks(np.r_[0,smooth,0],height=height,prominence=prominence,distance=separation)
            peaks=peaks-1
            for peak in peaks[np.argsort(smooth[peaks])[::-1]]:
                evidence=q[np.abs(values-(minimum+peak/scale))<window]
                if len(evidence)<2:continue
                center=evidence[:,:2].mean(0)
                _,_,vt=np.linalg.svd(evidence[:,:2]-center,full_matrices=False)
                normal=vt[-1]
                if normal@normal0<0:normal=-normal
                if normal@normal0<.99999:continue
                if np.ptp(evidence[:,:2]@np.array([-normal[1],normal[0]]))<6:continue
                residual=(evidence[:,:2]-center)@normal
                if np.quantile(np.abs(residual),.9)>.10:continue
                offset=float(center@normal)
                direction=np.array([-normal[1],normal[0]])
                if direction[np.argmax(np.abs(direction))]<0:direction=-direction
                if any(abs(normal@np.array(p['normal_xy']))>.99999 and abs(offset-p['offset_mm'])<.3 for p in planes):continue
                distance=xyz[:,0]*normal[0]+xyz[:,1]*normal[1]-offset
                pts=xyz[np.abs(distance)<.12]
                t=pts[:,:2]@direction
                if len(pts)<10 or np.ptp(t)<6 or np.ptp(pts[:,2])<10:continue
                planes.append({'normal_xy':normal.tolist(),'direction_xy':direction.tolist(),
                    'offset_mm':offset,'range_mm':[float(t.min()),float(t.max())],
                    'z_range_mm':[float(pts[:,2].min()),float(pts[:,2].max())],
                    'support_count':len(pts),'local_3d_evidence_count':len(evidence)})
    # Only points whose local 3D normals indicate a secondary vertical face
    # enter the more costly line vote. Horizontal mesh rows are excluded first.
    diagonal_xy=np.unique(np.round(selected[secondary,:2],3),axis=0)
    if len(diagonal_xy)>20000:diagonal_xy=diagonal_xy[rng.choice(len(diagonal_xy),20000,False)]
    diagonal_lines=support_lines(diagonal_xy,min_support=6,tolerance=.12) if len(diagonal_xy)>=6 else []
    for line in diagonal_lines:
        n=np.array(line['normal_xy']);offset=line['offset_mm']
        if n[np.argmax(np.abs(n))]<0:n=-n;offset=-offset
        if np.abs(frame@n).max()>.99995:continue
        if any(abs(n@np.array(p['normal_xy']))>.99999 and abs(offset-p['offset_mm'])<.3 for p in planes):continue
        pts=xyz[np.abs(xyz[:,:2]@n-offset)<.12]
        direction=np.array([-n[1],n[0]])
        if direction[np.argmax(np.abs(direction))]<0:direction=-direction
        t=pts[:,:2]@direction
        if len(pts)<12 or np.ptp(t)<20 or np.ptp(pts[:,2])<10:continue
        planes.append({'normal_xy':n.tolist(),'direction_xy':direction.tolist(),'offset_mm':offset,
            'range_mm':[float(t.min()),float(t.max())],
            'z_range_mm':[float(pts[:,2].min()),float(pts[:,2].max())],
            'support_count':len(pts),'source':'secondary_3d_normal_hough'})
    return planes,{'xy_frame':frame.tolist(),'local_samples':len(selected),
        'plane_normal_directions':directions.tolist(),
        'local_surface_counts':valid.sum(0).tolist(),'vertical_edge_anchors':len(anchors),
        'oriented_edge_anchors':anchor_valid.sum(0).tolist(),'secondary_direction_points':len(diagonal_xy)}


def surface_candidates(xyz):
    planes,meta=vertical_surface_planes(xyz)
    z,counts=np.unique(np.round(xyz[:,2],2),return_counts=True)
    levels=z[counts>=max(50,len(xyz)*.004)]
    candidates=[];line_records=[]
    for base in levels:
        near=xyz[np.abs(xyz[:,2]-base)<.015]
        for pi,plane in enumerate(planes):
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
                candidates.append({'id':f'local_h_{len(candidates)+1:04d}','type':'horizontal',
                    'segments':[{'kind':'line','points':[[*xy0,float(base)],[*xy1,float(base)]]}],
                    'plane_id':pi,'surface_z_mm':float(base)})
                line_records.append({'plane_id':pi,'z_mm':float(base),'range_mm':[float(t[a]),float(t[b-1])]})
    tree=cKDTree(xyz[:,:2]);seen=set()
    for i,a in enumerate(planes):
        for j in range(i+1,len(planes)):
            b=planes[j];matrix=np.array([a['normal_xy'],b['normal_xy']])
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
            candidates.append({'id':f'local_v_{len(candidates)+1:04d}','type':'vertical',
                'segments':[{'kind':'line','points':[[*xy,lo],[*xy,hi]]}],
                'plane_ids':[i,j]})
    return candidates,{**meta,'vertical_planes':planes,'horizontal_levels':levels.tolist(),
        'candidate_count':len(candidates),'horizontal_lines':line_records}
