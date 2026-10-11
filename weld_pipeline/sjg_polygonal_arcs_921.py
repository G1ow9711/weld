"""Recover analytic arcs from tessellated finite boundaries, using XYZ only."""
import numpy as np
from weld_pipeline.sjg_planar_boundary_curves_894 import boundary_chains,fit_circle,arc_geometry
from weld_pipeline.contract_curve_geometry import CircularArc

def simplify_indices(points,tolerance=.05):
    points=np.asarray(points,float);keep=np.zeros(len(points),bool)
    if len(points)<3:return np.arange(len(points),dtype=np.int64)
    keep[[0,-1]]=True
    if np.linalg.norm(points[0]-points[-1])<1e-10:
        far=int(np.argmax(np.linalg.norm(points-points[0],axis=1)));keep[far]=True;pending=[(0,far),(far,len(points)-1)]
    else:pending=[(0,len(points)-1)]
    while pending:
        lo,hi=pending.pop()
        if hi-lo<2:continue
        a,b=points[lo],points[hi];delta=b-a;squared=float(delta@delta);p=points[lo+1:hi]
        if squared<1e-20:d=np.linalg.norm(p-a,axis=1)
        else:
            t=np.clip((p-a)@delta/squared,0,1);d=np.linalg.norm(p-(a+t[:,None]*delta),axis=1)
        at=int(d.argmax())
        if d[at]>tolerance:
            k=lo+1+at;keep[k]=True;pending.extend([(lo,k),(k,hi)])
    return np.flatnonzero(keep).astype(np.int64)

def fitted_runs(uv,vertex_tolerance=.2):
    result=[];start=0
    while start+4<=len(uv):
        fitted=None
        for width in sorted(set(min(len(uv)-start,w) for w in (4,5,9,17,33,65,129))):
            stop=start+width;candidate=fit_circle(uv[start:stop])
            if candidate is None:continue
            # arc_geometry requires five samples; a midpoint is an actual
            # geometric test point only for angular monotonicity, not fitting.
            check=uv[start:stop]
            if len(check)==4:check=np.insert(check,2,(check[1]+check[2])/2,axis=0)
            geometry=arc_geometry(check,candidate[0],candidate[1],vertex_tolerance)
            if candidate[2].max()<=vertex_tolerance and geometry is not None:fitted=candidate;break
        if fitted is None:start+=1;continue
        center,radius,_=fitted
        while stop<len(uv) and abs(np.linalg.norm(uv[stop]-center)-radius)<=vertex_tolerance:stop+=1
        while stop>=start+4:
            fitted=fit_circle(uv[start:stop]);check=uv[start:stop]
            if len(check)==4:check=np.insert(check,2,(check[1]+check[2])/2,axis=0)
            geometry=arc_geometry(check,fitted[0],fitted[1],vertex_tolerance) if fitted else None
            if fitted is not None and fitted[2].max()<=vertex_tolerance and geometry is not None:break
            stop-=1
        if stop<start+4:start+=1;continue
        result.append(dict(start=start,stop=stop,center=fitted[0],radius=fitted[1],sweep=geometry[0],closed=geometry[1],vertex_error=float(fitted[2].max())))
        start=stop-2
    return result

def fit_connected_polyline(sample,maximum_deviation_mm=3.):
    simplified=simplify_indices(sample);vertices=sample[simplified];arcs=[]
    for candidate in fitted_runs(vertices):
        lo=int(simplified[candidate['start']]);hi=int(simplified[candidate['stop']-1]);points=sample[lo:hi+1]
        center,radius=candidate['center'],candidate['radius']
        radial_error=abs(np.linalg.norm(points-center,axis=1)-radius)
        angle=np.unwrap(np.arctan2(points[:,1]-center[1],points[:,0]-center[0]));delta=np.diff(angle)
        sign=np.sign(candidate['sweep'])
        if np.any(sign*delta<-1e-8) or np.any(abs(delta)>np.pi):continue
        # Between consecutive observations, chord sagitta plus endpoint radial
        # error bounds both directions of the circle/polyline distance.
        bound=float(radial_error.max()+np.max(2*radius*np.sin(abs(delta)/4)**2,initial=0.))
        if bound>maximum_deviation_mm:continue
        arcs.append(dict(**candidate,raw_start=lo,raw_stop=hi+1,polyline_bidirectional_bound_mm=bound))
    return arcs,dict(raw_vertices=len(sample),simplified_vertices=len(simplified))

def face_arcs(xyz,face):
    ids=np.asarray(face['point_ids'],np.int64);points=xyz[ids];basis=np.asarray(face['basis']);origin=np.asarray(face['center'])
    uv=(points-origin)@basis;normal=np.cross(basis[:,0],basis[:,1]);normal/=np.linalg.norm(normal)
    chains,stats=boundary_chains(face['triangles']);arcs=[];raw_vertices=simplified_vertices=0
    for ci,chain in enumerate(chains):
        if len(chain)<4:continue
        fitted,count=fit_connected_polyline(uv[chain]);raw_vertices+=count['raw_vertices'];simplified_vertices+=count['simplified_vertices']
        for c in fitted:
            radial=uv[chain[c['raw_start']]]-c['center'];radial/=np.linalg.norm(radial)
            arc=CircularArc(origin+basis@c['center'],c['radius'],normal,basis@radial,c['sweep'],c['closed'])
            arcs.append(dict(representation=dict(kind='arc',arc_parameters=arc.to_parameters()),length_mm=arc.length_mm,
                support_normal=normal.tolist(),support_intervals_mm=[[0.,arc.length_mm]],unobserved_gaps_mm=[],
                material_pair=None,weld_type=None,second_contact_face_unknown=True,
                status='UNCLASSIFIED_ANALYTIC_ARC_FROM_POLYGONAL_BOUNDARY',
                polyline_bidirectional_bound_mm=c['polyline_bidirectional_bound_mm'],vertex_fit_error_mm=c['vertex_error'],
                raw_support_point_ids=ids[chain[c['raw_start']:c['raw_stop']]].tolist(),boundary_chain=ci,
                connected_mesh_boundary_preserved=True,disconnected_chains_not_joined=True))
    return arcs,dict(**stats,raw_vertices=raw_vertices,simplified_vertices=simplified_vertices,arcs=len(arcs))

def controls():
    theta=np.linspace(0,.7,5);vertices=400*np.c_[np.cos(theta),np.sin(theta)]
    dense=np.concatenate([np.linspace(a,b,501)[:-1] for a,b in zip(vertices[:-1],vertices[1:])]+[vertices[-1:]])
    arcs,stats=fit_connected_polyline(dense)
    assert len(arcs)==1 and abs(arcs[0]['radius']-400)<1e-6 and arcs[0]['polyline_bidirectional_bound_mm']<3
    corner=np.concatenate([np.c_[np.linspace(0,100,501),np.zeros(501)],np.c_[np.full(500,100),np.linspace(.2,100,500)]])
    assert not fit_connected_polyline(corner)[0]
    square=np.array([[100.,0.],[0.,100.],[-100.,0.],[0.,-100.],[100.,0.]])
    dense_square=np.concatenate([np.linspace(a,b,101)[:-1] for a,b in zip(square[:-1],square[1:])]+[square[-1:]])
    assert not fit_connected_polyline(dense_square)[0]
    return dict(dense_tessellated_arc_radius_recovered=True,raw_points=len(dense),simplified_points=stats['simplified_vertices'],
                error_bound_mm=arcs[0]['polyline_bidirectional_bound_mm'],large_right_corner_and_square_rejected=True)
