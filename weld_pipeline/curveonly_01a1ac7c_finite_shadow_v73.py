"""Finite material projection observations, never a weld or gap decision.

Projecting every actual triangle preserves finite holes and nonconvex outlines;
no material bounding rectangle is substituted for its occupied footprint.
Ray heights refer to mesh surface intersections, not a torch clearance model.
"""
import numpy as np
from . import curveonly_01a1ac7c as cad


def triangles(solid,deflection_mm):
    cad.BRepMesh_IncrementalMesh(solid,deflection_mm,False,.2,False)
    result=[]
    for item in cad.items(solid,cad.TopAbs_FACE):
        face=cad.TopoDS.Face_s(item);location=cad.TopLoc_Location()
        mesh=cad.BRep_Tool.Triangulation_s(face,location)
        if mesh is None:continue
        nodes=np.array([cad.vec(mesh.Node(i).Transformed(location.Transformation()))
                        for i in range(1,mesh.NbNodes()+1)])
        indices=np.array([mesh.Triangle(i).Get() for i in range(1,mesh.NbTriangles()+1)],int)-1
        result.extend(nodes[indices])
    if not result:raise ValueError('Material has no finite triangulation')
    return np.asarray(result)


def basis(base_axis):
    z=np.asarray(base_axis,float);z=z/np.linalg.norm(z)
    axis=np.eye(3)[int(np.argmin(abs(z)))];x=axis-z*(axis@z);x/=np.linalg.norm(x)
    return np.column_stack([x,np.cross(z,x),z])


def observe(points,mesh):
    """Exact projected mesh-triangle distances and vertical surface crossings."""
    points=np.asarray(points,float);mesh=np.asarray(mesh,float)
    a=mesh[:,0,:2];e0=mesh[:,1,:2]-a;e1=mesh[:,2,:2]-a
    det=e0[:,0]*e1[:,1]-e0[:,1]*e1[:,0]
    scale=np.maximum(np.linalg.norm(e0,axis=1)*np.linalg.norm(e1,axis=1),1.)
    regular=abs(det)>1e-12*scale
    denominator=np.where(regular,det,1.)
    low=mesh[:,:,:2].min(1);high=mesh[:,:,:2].max(1)
    distance=np.empty(len(points));zmin=np.full(len(points),np.nan);zmax=zmin.copy();hits=np.zeros(len(points),int)
    for i,point in enumerate(points):
        q=point[:2];delta=q-a
        u=(delta[:,0]*e1[:,1]-delta[:,1]*e1[:,0])/denominator
        v=(e0[:,0]*delta[:,1]-e0[:,1]*delta[:,0])/denominator
        inside=regular&(u>=-1e-10)&(v>=-1e-10)&(u+v<=1.+1e-10)
        if inside.any():
            height=mesh[:,0,2]+u*(mesh[:,1,2]-mesh[:,0,2])+v*(mesh[:,2,2]-mesh[:,0,2])
            height=height[inside]-point[2]
            distance[i]=0.;zmin[i]=height.min();zmax[i]=height.max();hits[i]=int(inside.sum())
            continue
        # AABB bounds only prune exact triangle-edge distance computations.
        lower=np.linalg.norm(np.maximum(np.maximum(low-q,q-high),0.),axis=1)
        nearest=int(np.argmin(lower));candidates=np.array([nearest]);best=np.inf
        for phase in range(2):
            for edge in range(3):
                start=mesh[candidates,edge,:2];end=mesh[candidates,(edge+1)%3,:2];vector=end-start
                t=np.clip(((q-start)*vector).sum(1)/np.maximum((vector*vector).sum(1),1e-24),0.,1.)
                best=min(best,float(np.linalg.norm(q-start-t[:,None]*vector,axis=1).min()))
            if phase==0:candidates=np.flatnonzero(lower<=best+1e-10)
        distance[i]=best
    return dict(distance_mm=distance,minimum_ray_height_mm=zmin,maximum_ray_height_mm=zmax,ray_hits=hits)


def owner_at(candidate_index,s,packet,stack=()):
    if candidate_index in stack:raise ValueError('Cyclic input source provenance')
    candidate=packet['candidates'][candidate_index];path=candidate.get('source_material_path')
    if not path:return int(candidate.get('material',-1))
    length=float(cad.arclength(candidate['points'])[-1])
    if np.linalg.norm(np.asarray(candidate['points'])[0]-candidate['points'][-1])<=1e-6:s=s%length
    for member in path:
        if member['start_mm']-1e-8<=s<=member['end_mm']+1e-8:
            local=s-member['start_mm']
            if member['reversed']:local=member['end_mm']-member['start_mm']-local
            return owner_at(int(member['curve']),local,packet,stack+(candidate_index,))
    return -1
