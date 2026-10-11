"""Refit predicted fragments to finite planar supports using original XYZ.

Coplanar support unions are geometric carriers, not inferred material mergers.
All original fitted memberships remain available alongside the mesh samples.
"""
import numpy as np
from weld_pipeline.sjg_learned_surfaces_810 import split_component,finite_support


def merge_supports(xyz,planes):
    groups=[]
    for plane in sorted(planes,key=lambda p:-len(p['point_ids'])):
        for group in groups:
            reference=group['reference']
            if abs(plane['normal']@reference['normal'])<np.cos(np.deg2rad(1.)):continue
            if abs((plane['center']-reference['center'])@reference['normal'])>.2:continue
            if abs((reference['center']-plane['center'])@plane['normal'])>.2:continue
            residual=np.abs(xyz[plane['point_ids']]@reference['normal']-reference['offset'])
            if residual.max()>.2:continue
            group['chunks'].append(plane['point_ids']);group['components'].extend(plane['learned_component_ids']);break
        else:
            groups.append(dict(reference=plane,chunks=[plane['point_ids']],components=list(plane['learned_component_ids'])))
    merged=[]
    for group in groups:
        original=group['reference'];ids=np.unique(np.concatenate(group['chunks']))
        # Keep the largest fitted patch's plane unless a refit preserves every
        # support point within the unchanged 0.2 mm tolerance.
        points=xyz[ids];center=points.mean(0);delta=points-center
        eigen,vectors=np.linalg.eigh(delta.T@delta/max(1,len(delta)))
        normal=vectors[:,0];offset=float(center@normal)
        if np.abs(points@normal-offset).max()<=.2:
            if normal[np.argmax(abs(normal))]<0:normal=-normal;offset=-offset
            basis=vectors[:,1:]
        else:normal=original['normal'];offset=original['offset'];center=original['center'];basis=original['basis']
        merged.append(dict(normal=normal,offset=offset,center=center,basis=basis,point_ids=ids,
            learned_component_ids=sorted(set(group['components'])),
            coplanar_union_is_not_material_identity=True))
    return merged


def cell_extrema(uv,cell):
    """Keep extrema in eight directions per cell, preserving boundary samples."""
    _,inverse=np.unique(cell,axis=0,return_inverse=True);selected=[]
    for direction in ((1.,0.),(0.,1.),(1.,1.),(1.,-1.)):
        coordinate=uv@np.asarray(direction);order=np.lexsort((coordinate,inverse))
        cuts=np.r_[0,np.flatnonzero(np.diff(inverse[order]))+1,len(order)]
        selected.extend([order[cuts[:-1]],order[cuts[1:]-1]])
    return np.unique(np.concatenate(selected))


def mesh_indices(uv,cell,boundary,preserve_extrema=False,preserve_boundary=False):
    _,selected=np.unique(cell,axis=0,return_index=True)
    if preserve_extrema:selected=cell_extrema(uv,cell)
    if preserve_boundary:selected=np.r_[selected,np.flatnonzero(boundary>=.5)]
    return np.unique(np.r_[selected,np.argmin(uv,axis=0),np.argmax(uv,axis=0)])


def extract(xyz,pred,instance,material,progress=None,mesh_builder=finite_support,preserve_cell_extrema=False,shared_membership=False,preserve_predicted_boundary=False,support_refiner=None):
    assert len(xyz)==len(instance)==len(material)==len(pred['normal'])
    order=np.argsort(instance,kind='stable');bounds=np.r_[0,np.flatnonzero(np.diff(instance[order]))+1,len(order)]
    rng=np.random.default_rng(982);planes=[];remaining=[];exhausted=0
    for i,(lo,hi) in enumerate(zip(bounds[:-1],bounds[1:])):
        ids=order[lo:hi]
        if len(ids)<16:remaining.append(ids);continue
        local,left,trials=split_component(xyz,ids,pred['normal'],pred['boundary'],rng)
        if len(left):remaining.append(left)
        exhausted+=int(trials==64 and len(left)>=16)
        for plane in local:plane['learned_component_ids']=[int(instance[ids[0]])]
        planes.extend(local)
        if progress and i%20==0:progress('FIT_INSTANCE',i+1,len(bounds)-1)
    raw=len(planes);planes=merge_supports(xyz,planes);faces=[];full={};failed=[]
    for i,plane in enumerate(planes):
        seed_support=len(plane['point_ids'])
        if shared_membership:
            additional=[]
            for begin in range(0,len(xyz),65536):
                points=xyz[begin:begin+65536];near=np.flatnonzero(np.abs(points@plane['normal']-plane['offset'])<=.2)
                candidates=begin+near
                evidence=(np.abs(pred['normal'][candidates]@plane['normal'])>=np.cos(np.deg2rad(20.)))|(pred['boundary'][candidates]>=.5)
                additional.append(candidates[evidence])
            plane['point_ids']=np.unique(np.concatenate([plane['point_ids']]+additional))
        if support_refiner is not None:
            before=plane['point_ids'];plane=support_refiner(xyz,plane)
            removed=np.setdiff1d(before,plane['point_ids'],assume_unique=True)
            if len(removed):remaining.append(removed)
        ids=plane['point_ids'];uv=(xyz[ids]-plane['center'])@plane['basis']
        # Sample only the mesh; preserve complete original support separately.
        cell=np.floor((uv-uv.min(0))/1.).astype(np.int64)
        selected=mesh_indices(uv,cell,pred['boundary'][ids],preserve_cell_extrema,preserve_predicted_boundary)
        mesh_plane=dict(plane,point_ids=ids[selected]);face=mesh_builder(xyz,mesh_plane)
        if face is None:failed.append(ids);continue
        fid=len(faces);full[f'original_point_ids_{fid}']=ids
        owners,counts=np.unique(material[ids],return_counts=True)
        face.update(full_support_key=f'original_point_ids_{fid}',full_support_count=len(ids),
            mesh_sample_grid_mm=1.,mesh_approximation_not_acceptance_certified=True,
            mesh_cell_extrema_preserved=preserve_cell_extrema,
            mesh_predicted_boundary_points_preserved=preserve_predicted_boundary,
            shared_membership_from_normal_or_boundary=shared_membership,seed_support_count=seed_support,
            full_support_max_plane_residual_mm=float(np.abs(xyz[ids]@face['normal']-face['offset']).max()),
            predicted_material_distribution=[dict(instance=int(o),points=int(c),fraction=float(c/len(ids))) for o,c in zip(owners,counts)],
            material_id=None,material_status='PREDICTED_DISTRIBUTION_NOT_CERTIFIED_ID',proposal_branch='learned_native_finite_geometry')
        faces.append(face)
        if progress:progress('FINITE_MESH',i+1,len(planes),'ORIGINAL',len(ids),'MESH',len(face['point_ids']))
    unassigned=np.unique(np.concatenate(remaining+failed)) if remaining or failed else np.empty(0,np.int64)
    seen=np.zeros(len(xyz),np.uint8)
    for ids in full.values():seen[ids]=1
    assert np.all((seen>0)|np.isin(np.arange(len(xyz)),unassigned))
    unassigned=np.flatnonzero(seen==0)
    return faces,full,unassigned,dict(original_points=len(xyz),input_instances=len(bounds)-1,
        raw_fitted_planes=raw,merged_support_planes=len(planes),finite_surfaces=len(faces),
        original_points_with_finite_support=int(seen.sum()),unresolved_original_points=len(unassigned),
        fit_budget_exhausted_components=exhausted,curved_surface_fitting_pending=True,
        predicted_shared_face_memberships=shared_membership,
        unresolved_subsampling_holes=True,material_partition_not_replaced_by_coplanarity=True)


def controls():
    from weld_pipeline.sjg_xyz_batch_795 import intervals
    x,y=np.meshgrid(np.arange(-30.,31.),np.arange(-30.,31.));xyz=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
    xyz=xyz[~((np.abs(xyz[:,0])<5)&(xyz[:,1]<-5))]
    plane=dict(normal=np.array([0.,0.,1.]),center=np.zeros(3),offset=0.,basis=np.eye(3)[:,:2],point_ids=np.arange(len(xyz)))
    face=finite_support(xyz,plane);segments=intervals(face,xyz,np.array([0.,-15.,0.]),np.array([1.,0.,0.]))
    assert len(segments)==2 and not any(lo<=0<=hi for lo,hi in segments)
    # Two separated parallel surfaces cannot merge just because the normals match.
    a=dict(plane,point_ids=np.arange(20),learned_component_ids=[0]);b=dict(a,center=np.array([0.,0.,12.]),offset=12.,point_ids=np.arange(20,40),learned_component_ids=[1])
    points=np.r_[xyz[:20],xyz[:20]+np.array([0.,0.,12.])]
    assert len(merge_supports(points,[a,b]))==2
    # Fit fragmented perpendicular surfaces; meshing may sample their support,
    # but the union of fitted and unresolved original indices must stay exact.
    u,v=np.meshgrid(np.arange(21.),np.arange(21.));horizontal=np.c_[u.ravel(),v.ravel(),np.zeros(u.size)]
    vertical=np.c_[u.ravel(),np.zeros(u.size),v.ravel()];points=np.r_[horizontal,vertical]
    membership=np.r_[(horizontal[:,0]>=10).astype(int),2+(vertical[:,0]>=10).astype(int)]
    owner=np.r_[np.zeros(len(horizontal),int),np.ones(len(vertical),int)]
    pred=dict(normal=np.r_[np.tile([0.,0.,1.],(len(horizontal),1)),np.tile([0.,1.,0.],(len(vertical),1))],boundary=np.zeros(len(points)))
    faces,full,left,stats=extract(points,pred,membership,owner)
    assert len(faces)==2 and len(left)==0 and stats['original_points_with_finite_support']==len(points)
    assert np.array_equal(np.sort(np.concatenate(list(full.values()))),np.arange(len(points)))
    return dict(notch_stays_open=True,notch_intervals=segments,parallel_offset_surfaces_not_merged=True,
                original_index_support_preserved=True,fragmented_planes_reconstructed=True)
