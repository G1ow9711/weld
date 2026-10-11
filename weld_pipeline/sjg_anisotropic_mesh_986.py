"""Observed sampling-pitch normalization for striped ideal-CAD point supports."""
import numpy as np
from scipy.spatial import Delaunay,QhullError,cKDTree
from weld_pipeline.sjg_learned_surfaces_810 import finite_support
from weld_pipeline.sjg_native_finite_geometry_982 import extract as original_extract


def sampling_pitch(values,peaked=False):
    bins=np.floor(values-values.min()).astype(np.int64);counts=np.bincount(bins);positive=counts[counts>0]
    # Reject low-density contour spill while keeping regular interior stripes.
    cutoff=np.median(positive)/4.
    if peaked and positive.max()>=3*np.median(positive):cutoff=2*np.median(positive)
    active=np.flatnonzero(counts>=max(1.,cutoff))
    groups=np.split(active,np.flatnonzero(np.diff(active)>1)+1)
    centers=np.array([np.average(g,weights=counts[g]) for g in groups if len(g)])
    gaps=np.diff(centers);eligible=gaps[gaps>=2]
    if len(eligible)<8:return 1.,dict(periodic=False,groups=len(groups))
    rounded=np.maximum(2,np.round(eligible).astype(int));hist=np.bincount(rounded);mode=int(hist.argmax())
    near=eligible[np.abs(eligible-mode)<=max(1.,mode*.15)];pitch=float(np.median(near))
    agreement=float((np.abs(eligible/pitch-np.round(eligible/pitch))<=.15).mean())
    direct=float((np.abs(eligible-pitch)<=max(1.,pitch*.15)).mean())
    accepted=agreement>=.8 and direct>=.5 and pitch>=3
    return (pitch if accepted else 1.),dict(periodic=accepted,groups=len(groups),inferred_pitch_mm=pitch,
        lattice_multiple_agreement=agreement,adjacent_gap_fraction=direct)


def sampling_frame(uv):
    ids=np.linspace(0,len(uv)-1,min(4096,len(uv)),dtype=np.int64)
    distance,near=cKDTree(uv).query(uv[ids],k=min(8,len(uv)),workers=2)
    valid=distance>1e-5;available=valid.any(1);column=valid.argmax(1)
    delta=uv[near[np.arange(len(ids))[available],column[available]]]-uv[ids[available]]
    angle=np.mod(np.arctan2(delta[:,1],delta[:,0]),np.pi)
    bins=np.floor(angle/np.pi*180).astype(int).clip(0,179);hist=np.bincount(bins,minlength=180)
    mode=int(hist.argmax());center=(mode+.5)*np.pi/180
    deviation=np.abs(np.angle(np.exp(2j*(angle-center))))/2;selected=deviation<np.deg2rad(2.)
    fraction=float(selected.mean())
    if fraction<.3:return np.eye(2),dict(estimated=False,dominant_neighbor_fraction=fraction)
    theta=float(np.angle(np.exp(2j*angle[selected]).mean())/2)
    axis=np.array([np.cos(theta),np.sin(theta)]);rotation=np.column_stack([axis,[-axis[1],axis[0]]])
    return rotation,dict(estimated=True,dominant_neighbor_fraction=fraction,angle_radians=theta)


def anisotropic_support(xyz,plane,align_sampling=False):
    ids=plane['point_ids'];uv=(xyz[ids]-plane['center'])@plane['basis']
    _,unique=np.unique(np.round(uv,5),axis=0,return_index=True);ids=ids[unique];uv=uv[unique]
    if len(ids)<16:return None
    rotation,orientation=sampling_frame(uv) if align_sampling else (np.eye(2),dict(estimated=False))
    aligned=uv@rotation
    pitch_info=[sampling_pitch(aligned[:,axis],peaked=align_sampling) for axis in range(2)];pitch=np.array([x[0] for x in pitch_info])
    if pitch.max()<3:return finite_support(xyz,plane)
    metric=aligned/pitch
    try:triangles=Delaunay(metric).simplices
    except QhullError:return None
    # Adjacent observed sampling cells may connect. A gap of two or more
    # sampling cells stays unresolved instead of being filled as a convex hull.
    keep=np.ones(len(triangles),bool);q=metric[triangles]
    for i,j in ((0,1),(1,2),(2,0)):keep &= np.linalg.norm(q[:,i]-q[:,j],axis=1)<=1.8
    triangles=triangles[keep]
    if not len(triangles):return None
    return dict(**{k:v for k,v in plane.items() if k!='point_ids'},point_ids=ids,triangles=triangles,
        sample_spacing_mm=float(pitch.min()),sampling_pitch_mm=pitch.tolist(),sampling_pitch_diagnosis=[x[1] for x in pitch_info],
        sampling_orientation=orientation,sampling_rotation=rotation.tolist(),
        support_rule='observed periodic sampling metric; triangle edges <=1.8 cells; no convex fill',
        unresolved_subsampling_holes=True,material_id=None,material_status='UNKNOWN_NOT_INFERRED',
        max_plane_residual_mm=float(np.abs(xyz[ids]@plane['normal']-plane['offset']).max()))


def extract(xyz,pred,instance,material,progress=None):
    return original_extract(xyz,pred,instance,material,progress,mesh_builder=anisotropic_support)


def extract_preserve_boundary(xyz,pred,instance,material,progress=None):
    return original_extract(xyz,pred,instance,material,progress,mesh_builder=anisotropic_support,preserve_cell_extrema=True)


def extract_shared_boundary(xyz,pred,instance,material,progress=None):
    return original_extract(xyz,pred,instance,material,progress,mesh_builder=anisotropic_support,preserve_cell_extrema=True,shared_membership=True)


def oriented_support(xyz,plane):return anisotropic_support(xyz,plane,align_sampling=True)


def extract_oriented_sampling(xyz,pred,instance,material,progress=None):
    return original_extract(xyz,pred,instance,material,progress,mesh_builder=oriented_support,preserve_cell_extrema=True,shared_membership=True)


def oriented_controls():
    from weld_pipeline.sjg_xyz_batch_795 import intervals
    result=shared_controls()
    x,y=np.meshgrid(np.arange(-200.,201.,20.),np.arange(-30.,31.));p=np.c_[x.ravel(),y.ravel()]
    angle=.27;r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    cloud=np.c_[p@r.T,np.zeros(len(p))]
    plane=dict(normal=np.array([0.,0.,1.]),center=np.zeros(3),offset=0.,basis=np.eye(3)[:,:2],point_ids=np.arange(len(cloud)))
    face=oriented_support(cloud,plane);assert face is not None
    span=intervals(face,cloud,np.zeros(3),np.r_[r[:,0],0.])
    assert len(span)==1 and np.allclose(span[0],[-200,200])
    result['rotated_sampling_grid_recovered']=True
    return result


def boundary_controls():
    from weld_pipeline.sjg_native_finite_geometry_982 import cell_extrema
    result=controls()
    uv=np.array([[.05,.05],[.95,.05],[.05,.95],[.95,.95],[.5,.5]])
    selected=cell_extrema(uv,np.floor(uv).astype(int))
    assert set(range(4))<=set(selected)
    result['all_four_cell_corners_preserved']=True
    return result


def shared_controls():
    result=boundary_controls()
    u,v=np.meshgrid(np.arange(21.),np.arange(21.));h=np.c_[u.ravel(),v.ravel(),np.zeros(u.size)]
    wall=np.c_[u.ravel(),np.zeros(u.size),v.ravel()];xyz=np.r_[h,wall]
    labels=np.r_[np.zeros(len(h),int),np.ones(len(wall),int)]
    normal=np.r_[np.tile([0.,0.,1.],(len(h),1)),np.tile([0.,1.,0.],(len(wall),1))]
    edge=((np.abs(xyz[:,1])<1e-8)&(np.abs(xyz[:,2])<1e-8)).astype(float)
    faces,full,left,stats=extract_shared_boundary(xyz,dict(normal=normal,boundary=edge),labels,labels)
    assert len(faces)==2 and len(left)==0
    common=np.intersect1d(full['original_point_ids_0'],full['original_point_ids_1'])
    assert len(common)==42 and np.all(edge[common]==1)
    result['observed_shared_edge_supports_both_faces']=True
    return result


def controls():
    from weld_pipeline.sjg_xyz_batch_795 import intervals
    x,y=np.meshgrid(np.arange(-200.,201.,20.),np.arange(-30.,31.))
    cloud=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
    plane=dict(normal=np.array([0.,0.,1.]),center=np.zeros(3),offset=0.,basis=np.eye(3)[:,:2],point_ids=np.arange(len(cloud)))
    face=anisotropic_support(cloud,plane);assert face is not None
    span=intervals(face,cloud,np.zeros(3),np.array([1.,0.,0.]))
    assert len(span)==1 and np.allclose(span[0],[-200,200])
    cut=cloud[~((np.abs(cloud[:,0])<40)&(cloud[:,1]<-5))]
    face=anisotropic_support(cut,dict(plane,point_ids=np.arange(len(cut))))
    parts=intervals(face,cut,np.array([0.,-15.,0.]),np.array([1.,0.,0.]))
    assert len(parts)==2 and not any(lo<=0<=hi for lo,hi in parts)
    return dict(striped_plane_continuous=True,pitch_inferred_without_fixed_20mm=True,
        multi_cell_notch_preserved=True,sub_sampling_holes_not_certified=True)
