"""Finite planar mesh scales from non-collinear spatial representatives."""
import numpy as np
from scipy.spatial import cKDTree,Delaunay


def adaptive_radii(uv,progress=None):
    keys=np.floor(uv-uv.min(0)).astype(np.int64)
    _,ids,inverse=np.unique(keys,axis=0,return_index=True,return_inverse=True)
    points=uv[ids];tree=cKDTree(points);radii=np.zeros(len(points));initial=np.zeros(len(points));pending=np.arange(len(points))
    if len(points)<3:
        return np.full(len(uv),2./1.5),dict(canonical_points=len(points),unresolved_rank_points=len(points),neighborhoods=[])
    counts=[]
    for requested in (24,48,96,192,384,768):
        if not len(pending):break
        k=min(requested,len(points));remaining=[]
        for begin in range(0,len(pending),2048):
            take=pending[begin:begin+2048];distance,near=tree.query(points[take],k=k,workers=2)
            cloud=points[near];delta=cloud-cloud.mean(1)[:,None]
            eigen=np.linalg.eigvalsh(np.einsum('nki,nkj->nij',delta,delta)/k)
            valid=eigen[:,0]/np.maximum(eigen[:,1],1e-12)>=1e-4
            if requested==24:initial[take]=distance[:,-1]
            radii[take[valid]]=distance[valid,-1];remaining.extend(take[~valid].tolist())
        pending=np.asarray(remaining,np.int64);counts.append(dict(k=k,remaining_collinear=len(pending)))
        if progress:progress('MESH_RANK_SCALE',k,'UNRESOLVED',len(pending),'/',len(points))
        if k==len(points):break
    # A genuinely thin/unresolved support must not gain a large artificial
    # radius merely because no two-dimensional neighborhood was found.
    radii[pending]=initial[pending]
    return radii[inverse],dict(canonical_points=len(points),unresolved_rank_points=len(pending),neighborhoods=counts)


def remesh(uv,progress=None):
    if len(uv)<3:return np.empty((0,3),np.int64),dict(points=len(uv))
    radius,stats=adaptive_radii(uv,progress);triangles=Delaunay(uv).simplices
    q=uv[triangles];keep=np.ones(len(triangles),bool)
    for i,j in ((0,1),(1,2),(2,0)):
        keep &= np.linalg.norm(q[:,i]-q[:,j],axis=1)<=np.maximum(2.,1.5*np.minimum(radius[triangles[:,i]],radius[triangles[:,j]]))
    return triangles[keep],dict(stats,all_triangles=len(triangles),retained_triangles=int(keep.sum()))


def controls():
    x,y=np.meshgrid(np.arange(101.),np.arange(0,41,20.));uv=np.c_[x.ravel(),y.ravel()]
    tri,stats=remesh(uv)
    areas=abs(np.cross(uv[tri[:,1]]-uv[tri[:,0]],uv[tri[:,2]]-uv[tri[:,0]]))/2
    assert abs(areas.sum()-4000)<1e-8
    x,y=np.meshgrid(np.arange(11.),np.arange(11.));a=np.c_[x.ravel(),y.ravel()];uv=np.r_[a,a+[100,0]]
    tri,_=remesh(uv);assert not np.any((uv[tri][:,:,0].min(1)<20)&(uv[tri][:,:,0].max(1)>90))
    return dict(anisotropic_plane_sampling_surface_recovered=True,well_supported_disconnected_gap_not_bridged=True)
