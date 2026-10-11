"""Finite support using the smallest neighborhood spanning two dimensions.

Neighbor analysis uses 1 mm representatives to avoid dense collinear export
strips hiding the next observed strip. Geometry uses retained original vertices.
"""
import numpy as np
from scipy.spatial import cKDTree,Delaunay,QhullError
from weld_pipeline.sjg_native_finite_geometry_982 import extract as original_extract


def two_dimensional_radius(uv):
    _,ids=np.unique(np.floor((uv-uv.min(0))/1.).astype(np.int64),axis=0,return_index=True)
    tree=cKDTree(uv[ids]);ks=sorted(set(min(k,len(ids)) for k in (24,48,96,192)))
    radius=np.zeros(len(uv));used=np.zeros(len(uv),np.int16)
    if len(ids)<3:return radius,used
    for start in range(0,len(uv),2048):
        q=uv[start:start+2048];distance,near=tree.query(q,k=ks[-1],workers=2)
        delta=uv[ids[near]]-q[:,None,:];r=radius[start:start+len(q)];selected=used[start:start+len(q)]
        for k in ks:
            v=delta[:,:k];mean=v.mean(1);cov=np.einsum('nki,nkj->nij',v,v)/k-mean[:,:,None]*mean[:,None,:]
            eigen=np.linalg.eigvalsh(cov);good=(selected==0)&(eigen[:,0]>=.05*eigen[:,1])&(eigen[:,1]>1e-6)
            r[good]=distance[good,k-1];selected[good]=k
    return radius,used


def local_support(xyz,plane):
    ids=plane['point_ids'];uv=(xyz[ids]-plane['center'])@plane['basis']
    _,unique=np.unique(np.round(uv,5),axis=0,return_index=True);ids=ids[unique];uv=uv[unique]
    if len(ids)<16:return None
    radius,used=two_dimensional_radius(uv)
    try:triangles=Delaunay(uv).simplices
    except QhullError:return None
    keep=np.all(used[triangles]>0,axis=1);q=uv[triangles]
    for i,j in ((0,1),(1,2),(2,0)):
        keep &= np.linalg.norm(q[:,i]-q[:,j],axis=1)<=np.maximum(2.,1.5*np.minimum(radius[triangles[:,i]],radius[triangles[:,j]]))
    triangles=triangles[keep]
    if not len(triangles):return None
    unique_k,count=np.unique(used,return_counts=True)
    return dict(**{k:v for k,v in plane.items() if k!='point_ids'},point_ids=ids,triangles=triangles,
        sample_spacing_mm=1.,support_rule='smallest 24/48/96/192 canonical-neighbor set with 2D covariance ratio >=0.05',
        neighborhood_counts={str(int(k)):int(n) for k,n in zip(unique_k,count)},
        support_radius_p50_p95_mm=np.quantile(radius[used>0],[.5,.95]).tolist(),
        unresolved_subsampling_holes=True,material_id=None,material_status='UNKNOWN_NOT_INFERRED',
        max_plane_residual_mm=float(np.abs(xyz[ids]@plane['normal']-plane['offset']).max()))


def extract(xyz,pred,instance,material,progress=None):
    return original_extract(xyz,pred,instance,material,progress,mesh_builder=local_support,preserve_cell_extrema=True,shared_membership=True)


def extract_boundary_vertices(xyz,pred,instance,material,progress=None):
    return original_extract(xyz,pred,instance,material,progress,mesh_builder=local_support,preserve_cell_extrema=True,shared_membership=True,preserve_predicted_boundary=True)


def boundary_vertex_controls():
    from weld_pipeline.sjg_native_finite_geometry_982 import mesh_indices
    result=controls()
    uv=np.array([[.05,.05],[.95,.05],[.05,.95],[.95,.95],[.5,.5]])
    boundary=np.array([0.,0.,0.,0.,.99]);cell=np.floor(uv).astype(int)
    assert 4 not in mesh_indices(uv,cell,boundary,True,False)
    assert 4 in mesh_indices(uv,cell,boundary,True,True)
    result['internal_cell_boundary_vertex_retained']=True
    return result


def controls():
    from weld_pipeline.sjg_xyz_batch_795 import intervals
    from weld_pipeline.sjg_anisotropic_mesh_986 import shared_controls
    result=shared_controls()
    for angle in (0.,.27):
        x,y=np.meshgrid(np.arange(-200.,201.,20.),np.arange(-30.,31.));p=np.c_[x.ravel(),y.ravel()]
        r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        cloud=np.c_[p@r.T,np.zeros(len(p))]
        plane=dict(normal=np.array([0.,0.,1.]),center=np.zeros(3),offset=0.,basis=np.eye(3)[:,:2],point_ids=np.arange(len(cloud)))
        face=local_support(cloud,plane);assert face is not None
        span=intervals(face,cloud,np.zeros(3),np.r_[r[:,0],0.]);assert len(span)==1 and np.allclose(span[0],[-200,200])
        cut=cloud[~((np.abs(p[:,0])<40)&(p[:,1]<-5))]
        face=local_support(cut,dict(plane,point_ids=np.arange(len(cut))));assert face is not None
        parts=intervals(face,cut,np.r_[r@np.array([0.,-15.]),0.],np.r_[r[:,0],0.])
        assert len(parts)==2 and not any(lo<=0<=hi for lo,hi in parts)
    result.update(rotated_local_strips_connected=True,multi_cell_notches_preserved=True,global_grid_orientation_not_required=True)
    return result
