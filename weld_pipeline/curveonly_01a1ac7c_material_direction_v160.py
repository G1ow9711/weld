"""Finite direction envelopes with periodic frames at duplicate closed stations."""
import time
import numpy as np
from .curveonly_01a1ac7c_material_direction_v157 import (
    closest,layers,NAMES,LAYER_NAMES,LAYER_TIE_MM,DEFLECTION_MM,original,indexed,shadow,reference)


def station_frames(points,closed,base_axis=(0.,0.,1.)):
    points=np.asarray(points,float);axis=np.asarray(base_axis,float)
    assert points.ndim==2 and points.shape[1]==3 and len(points)>=2
    axis=axis/np.linalg.norm(axis)
    if closed:
        assert np.linalg.norm(points[0]-points[-1])<=reference.PROJECTION_NUMERIC_MM
        unique=points[:-1]
        tangent=np.roll(unique,-1,axis=0)-np.roll(unique,1,axis=0)
        tangent=np.concatenate([tangent,tangent[:1]])
    else:tangent=np.gradient(points,axis=0)
    norm=np.linalg.norm(tangent,axis=1)
    tangent=tangent/np.maximum(norm[:,None],1e-12)
    base=np.broadcast_to(axis,points.shape)
    radial=np.cross(tangent,base);radial_norm=np.linalg.norm(radial,axis=1)
    valid=(norm>1e-10)&(radial_norm>1e-10)
    radial/=np.maximum(radial_norm[:,None],1e-12)
    return np.stack([tangent,base,radial],axis=1),valid


def build(packet,solids):
    start=time.perf_counter();values=np.zeros((packet['offsets'][-1],len(NAMES)),np.float32)
    pose=packet.get('pose',{});T=np.asarray(pose.get('transform_STEP_to_XYZ',[]))
    if not(solids and pose.get('passed')and T.shape==(4,4)):
        return values,dict(available=False,reason='NO_QUALIFIED_FINITE_STEP',seconds=time.perf_counter()-start,
            names=list(NAMES),geometry_and_GT_retained=True)
    B=shadow.basis(packet['base_axis']);origin=T[:3,3];meshes=[];boxes=[]
    for solid in solids:
        mesh=shadow.triangles(solid,DEFLECTION_MM);mesh=((mesh@T[:3,:3].T+T[:3,3])-origin)@B
        meshes.append(mesh);boxes.append(np.r_[mesh.min(axis=(0,1)),mesh.max(axis=(0,1))])
    boxes=np.asarray(boxes);owner_at=indexed.owner_function(packet);cache={};details=[];queries=0
    for ci,candidate in enumerate(packet['candidates']):
        lo,hi=packet['offsets'][ci:ci+2];points=(np.asarray(candidate['stations'])-origin)@B
        native,reference_receipt=reference.parameters(candidate)
        owners=np.asarray([owner_at(ci,float(s),packet)for s in native]);known=(owners>=0)&(owners<len(solids))
        closed=np.linalg.norm(np.asarray(candidate['points'])[0]-np.asarray(candidate['points'])[-1])<=1e-6
        frames,frame=station_frames(points,closed)
        lower=np.linalg.norm(np.maximum(np.maximum(boxes[:,:3][None]-points[:,None],points[:,None]-boxes[:,3:][None]),0),axis=2)
        distance=np.full((len(points),len(solids)),np.inf);direction_available=np.zeros(distance.shape,bool)
        low=np.zeros((*distance.shape,3));high=np.zeros_like(low);second=np.full(len(points),np.inf)
        for mi in np.argsort(np.median(lower,axis=0),kind='stable'):
            mi=int(mi);take=np.flatnonzero(known&(owners!=mi)&(lower[:,mi]<=second+indexed.PRUNE_GUARD_MM))
            if not len(take):continue
            part=closest(points[take],meshes[mi],frames[take],cache);queries+=len(take)
            distance[take,mi]=part['distance_mm'];direction_available[take,mi]=part['direction_available']
            low[take,mi]=part['direction_min'];high[take,mi]=part['direction_max']
            first=distance.min(1);second=np.where(distance>first[:,None]+LAYER_TIE_MM,distance,np.inf).min(1)
        values[lo:hi,:3]=np.column_stack([np.ones(len(points)),known,frame])
        values[lo:hi,3:]=layers(distance,direction_available,low,high,frame)
        details.append(dict(candidate=ci,rows=hi-lo,known_owner_rows=int(known.sum()),
            native_reference=reference_receipt,geometry_retained=True))
    assert np.isfinite(values).all()
    return values,dict(available=True,names=list(NAMES),materials=len(solids),mesh_triangles=sum(len(q)for q in meshes),
        mesh_deflection_mm=DEFLECTION_MM,exact_triangle_queries=queries,seconds=time.perf_counter()-start,candidates=details,
        all_nearest_triangle_and_material_ties_enveloped=True,closest_direction_is_not_surface_normal=True,
        no_inside_outside_solid_grids=True,no_labels_models_or_process_decision=True,unknown_owner_retains_geometry=True)
