"""Finite support across sampling-density transitions; no target access.

Use the symmetric union of local sampling neighborhoods for undirected edges.
Mutual-neighborhood intersection deletes edges when only one endpoint lies in a
dense boundary cluster. Both endpoints must still have measured 2D support.
Sub-sampling holes remain unresolved, as in the preceding estimator.
"""
import numpy as np
from scipy.spatial import Delaunay,QhullError
from weld_pipeline.sjg_local_sampling_mesh_991 import two_dimensional_radius
from weld_pipeline.sjg_native_finite_geometry_982 import extract as original_extract


def projected_unique(xyz,plane):
    ids=plane['point_ids'];points=xyz[ids]
    uv=(points-plane['center'])@plane['basis']
    residual=np.abs(points@plane['normal']-plane['offset'])
    # Equivalent UV vertices retain the observation closest to this surface.
    order=np.argsort(residual,kind='stable')
    _,unique=np.unique(np.round(uv[order],5),axis=0,return_index=True)
    selected=order[unique]
    return ids[selected],uv[selected]


def local_support(xyz,plane):
    ids,uv=projected_unique(xyz,plane)
    if len(ids)<16:return None
    radius,used=two_dimensional_radius(uv)
    try:triangles=Delaunay(uv).simplices
    except QhullError:return None
    keep=np.all(used[triangles]>0,axis=1);q=uv[triangles]
    for i,j in ((0,1),(1,2),(2,0)):
        keep &= np.linalg.norm(q[:,i]-q[:,j],axis=1)<=np.maximum(2.,1.5*np.maximum(radius[triangles[:,i]],radius[triangles[:,j]]))
    triangles=triangles[keep]
    if not len(triangles):return None
    unique_k,count=np.unique(used,return_counts=True)
    return dict(**{k:v for k,v in plane.items() if k!='point_ids'},point_ids=ids,triangles=triangles,
        sample_spacing_mm=1.,support_rule='symmetric union of measured local 2D sampling neighborhoods; both vertices rank-valid',
        neighborhood_counts={str(int(k)):int(n) for k,n in zip(unique_k,count)},
        support_radius_p50_p95_mm=np.quantile(radius[used>0],[.5,.95]).tolist(),
        unresolved_subsampling_holes=True,material_id=None,material_status='UNKNOWN_NOT_INFERRED',
        max_plane_residual_mm=float(np.abs(xyz[ids]@plane['normal']-plane['offset']).max()))


def extract(xyz,pred,instance,material,progress=None):
    return original_extract(xyz,pred,instance,material,progress,mesh_builder=local_support,
        preserve_cell_extrema=True,shared_membership=True,preserve_predicted_boundary=True)


def controls():
    from weld_pipeline import sjg_local_sampling_mesh_991 as previous
    from weld_pipeline.sjg_xyz_batch_795 import intervals
    old=previous.local_support
    try:
        previous.local_support=local_support
        result=previous.boundary_vertex_controls()
    finally:previous.local_support=old
    plane=dict(normal=np.array([0.,0.,1.]),center=np.zeros(3),offset=0.,basis=np.eye(3)[:,:2])
    xyz=np.array([[0.,0.,.17],[0.,0.,0.],[1.,1.,0.]])
    ids,uv=projected_unique(xyz,dict(plane,point_ids=np.arange(3)))
    assert 1 in ids and 0 not in ids
    # A dense border joins a sparse interior, without closing a wide real gap.
    x,y=np.meshgrid(np.arange(-200.,201.,20.),np.arange(-30.,31.))
    dense_x,dense_y=np.meshgrid(np.arange(180.,201.),np.arange(-30.,31.))
    cloud=np.unique(np.r_[np.c_[x.ravel(),y.ravel(),np.zeros(x.size)],np.c_[dense_x.ravel(),dense_y.ravel(),np.zeros(dense_x.size)]],axis=0)
    for angle in (0.,.37):
        r=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
        points=cloud@r.T
        f=local_support(points,dict(plane,point_ids=np.arange(len(points))))
        spans=intervals(f,points,np.zeros(3),r[:,0])
        assert len(spans)==1 and np.allclose(spans[0],[-200.,200.]),spans
        cut=points[~((np.abs(cloud[:,0])<40.)&(cloud[:,1]<-5.))]
        f=local_support(cut,dict(plane,point_ids=np.arange(len(cut))))
        spans=intervals(f,cut,r@np.array([0.,-15.,0.]),r[:,0])
        assert len(spans)==2 and not any(lo<=0<=hi for lo,hi in spans),spans
    result.update(projected_duplicate_prefers_closest_surface=True,density_transition_continuous=True,
        density_transition_notch_preserved=True,edge_support_union_not_threshold_tuning=True)
    return result
