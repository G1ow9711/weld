"""Unclassified 3D circle arcs fitted to observed finite-plane mesh boundaries."""
import numpy as np
from weld_pipeline.contract_curve_geometry import CircularArc


def boundary_chains(triangles):
    triangles=np.asarray(triangles,np.int64)
    edges=np.sort(np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]]),axis=1)
    edges,count=np.unique(edges,axis=0,return_counts=True);boundary=edges[count==1]
    adjacent={}
    for a,b in boundary:
        if a==b:continue
        adjacent.setdefault(int(a),[]).append(int(b));adjacent.setdefault(int(b),[]).append(int(a))
    used=set();chains=[]
    def walk(a,b):
        path=[a,b];used.add(tuple(sorted((a,b))))
        while len(adjacent[b])==2 and b!=path[0]:
            candidates=[c for c in adjacent[b] if tuple(sorted((b,c))) not in used]
            if not candidates:break
            c=candidates[0];used.add(tuple(sorted((b,c))));path.append(c);a,b=b,c
        return np.asarray(path,np.int64)
    for a in sorted(adjacent):
        if len(adjacent[a])==2:continue
        for b in sorted(adjacent[a]):
            if tuple(sorted((a,b))) not in used:chains.append(walk(a,b))
    for a,b in boundary:
        a,b=int(a),int(b)
        if a!=b and tuple(sorted((a,b))) not in used:chains.append(walk(a,b))
    assert len(used)==len(boundary)
    return chains,dict(boundary_edges=len(boundary),chains=len(chains),branch_vertices=sum(len(v)>2 for v in adjacent.values()))


def fit_circle(points):
    p=np.asarray(points,float);origin=p.mean(0);scale=float(np.linalg.norm(np.ptp(p,axis=0)))
    if scale<=1e-10:return None
    q=(p-origin)/scale;matrix=np.c_[2*q,np.ones(len(q))]
    value,_,rank,_=np.linalg.lstsq(matrix,(q*q).sum(1),rcond=None)
    if rank<3:return None
    radius2=float(value[2]+value[:2]@value[:2])
    if radius2<=0:return None
    center=origin+scale*value[:2];radius=scale*np.sqrt(radius2)
    error=abs(np.linalg.norm(p-center,axis=1)-radius)
    return center,float(radius),error


def arc_geometry(points,center,radius,tolerance):
    if len(np.unique(points,axis=0))<5:return None
    angles=np.unwrap(np.arctan2(points[:,1]-center[1],points[:,0]-center[0]));delta=np.diff(angles)
    sign=np.sign(np.median(delta[np.abs(delta)>1e-12])) if np.any(abs(delta)>1e-12) else 0
    if sign==0 or np.any(sign*delta < -tolerance/max(radius,1e-12)):return None
    sweep=float(angles[-1]-angles[0])
    if abs(sweep)>2*np.pi+1e-8:return None
    sagitta=radius*(1-np.cos(min(np.pi,abs(sweep))/2))
    if sagitta<3*tolerance:return None
    closed=bool(np.linalg.norm(points[0]-points[-1])<=1e-9 and abs(abs(sweep)-2*np.pi)<1e-8)
    if closed:sweep=float(sign*2*np.pi)
    elif abs(sweep)>=2*np.pi:return None
    return sweep,closed


def fit_chain_arcs(uv,tolerance):
    """Multiscale local seeds; accepted runs overlap by two boundary vertices."""
    uv=np.asarray(uv,float);result=[];start=0
    while start+5<=len(uv):
        fit=None
        for width in sorted(set(min(len(uv)-start,w) for w in (5,9,17,33,65,129))):
            stop=start+width;candidate=fit_circle(uv[start:stop])
            if candidate is not None and candidate[2].max()<=tolerance and arc_geometry(uv[start:stop],candidate[0],candidate[1],tolerance) is not None:
                fit=candidate;break
        if fit is None:
            start+=1;continue
        center,radius,_=fit
        while stop<len(uv) and abs(np.linalg.norm(uv[stop]-center)-radius)<=tolerance:stop+=1
        while stop>=start+5:
            fit=fit_circle(uv[start:stop]);geometry=None if fit is None else arc_geometry(uv[start:stop],fit[0],fit[1],tolerance)
            if fit is not None and fit[2].max()<=tolerance and geometry is not None:break
            stop-=1
        if stop<start+5:start+=1;continue
        center,radius,error=fit;sweep,closed=geometry
        result.append(dict(start=start,stop=stop,center=center,radius=radius,sweep=sweep,closed=closed,max_error=float(error.max())))
        start=stop-2
    return result


def face_boundary_arcs(xyz,face,point_ids,triangles):
    ids=np.asarray(point_ids,np.int64);points=np.asarray(xyz)[ids];basis=np.asarray(face['basis']);origin=np.asarray(face['center'])
    assert basis.shape==(3,2) and np.allclose(basis.T@basis,np.eye(2),atol=1e-7)
    normal=np.cross(basis[:,0],basis[:,1]);normal/=np.linalg.norm(normal)
    uv=(points-origin)@basis;chains,stats=boundary_chains(triangles);arcs=[];gap_splits=0
    split_chains=[]
    for chain in chains:
        distances=np.linalg.norm(np.diff(uv[chain],axis=0),axis=1);positive=distances[distances>1e-9]
        breaks=np.flatnonzero(distances>3*np.median(positive))+1 if len(positive) else []
        gap_splits+=len(breaks);split_chains.extend(np.split(chain,breaks))
    for ci,chain in enumerate(split_chains):
        if len(chain)<5:continue
        sample=uv[chain];dist=np.linalg.norm(np.diff(sample,axis=0),axis=1);positive=dist[dist>1e-9]
        if not len(positive):continue
        tolerance=float(np.clip(.025*np.median(positive),.02,.2))
        for fitted in fit_chain_arcs(sample,tolerance):
            start=fitted['start'];stop=fitted['stop'];center=fitted['center']
            radial=sample[start]-center;radial/=np.linalg.norm(radial)
            arc=CircularArc(origin+basis@center,fitted['radius'],normal,basis@radial,fitted['sweep'],fitted['closed'])
            arcs.append(dict(kind='arc',arc_parameters=arc.to_parameters(),support_point_ids=ids[chain[start:stop]].tolist(),
                chain=ci,start_vertex=start,stop_vertex_exclusive=stop,radial_max_error_mm=fitted['max_error'],
                fit_tolerance_mm=tolerance,length_mm=float(arc.length_mm),material_pair=None,weld_type=None,
                status='UNCLASSIFIED_FINITE_PLANE_BOUNDARY_NOT_WELD',observed_boundary_chain_only=True,
                disconnected_chains_not_joined=True,subsampling_gaps_unresolved=True))
    return arcs,dict(stats,arc_candidates=len(arcs),all_boundary_edges_visited=True,large_sampling_gap_splits=gap_splits)
