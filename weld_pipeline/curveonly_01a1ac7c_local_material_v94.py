"""Local finite-material evidence; never itself a should-weld/clearance rule."""
import time
import numpy as np
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_finite_shadow_v73 as shadow

NAMES=('pair_available','pair_surface_distance_log_mm',
       'smaller_material_below_signed_log_mm','smaller_material_above_signed_log_mm',
       'larger_material_below_signed_log_mm','larger_material_above_signed_log_mm',
       'third_available','third_surface_distance_log_mm','third_closest_height_signed_log_mm',
       'third_projected_boundary_distance_log_mm','third_projected_boundary_min_height_signed_log_mm',
       'third_projected_boundary_max_height_signed_log_mm')
DEFLECTION_MM=.025


def slog(x):
    x=np.asarray(x);return np.sign(x)*np.log1p(abs(x))


def closest(points,mesh,projected=False):
    """Exact continuous triangle/edge queries on a finite input mesh.

    Surface mode returns a closest 3D point. Projection mode returns distance
    to triangle edges in the base plane and the height range at all tied edges.
    Projected edge distance is not distance to the occupied footprint.
    """
    q=np.asarray(points,float);tri=np.asarray(mesh,float)
    result=np.empty(len(q));point=np.empty((len(q),3));zlo=np.empty(len(q));zhi=np.empty(len(q))
    batch=max(1,min(64,100000//max(1,len(tri))))
    a=tri[:,0];v=tri[:,1]-a;w=tri[:,2]-a
    normal=np.cross(v,w);nn=(normal*normal).sum(1)
    vv=(v*v).sum(1);vw=(v*w).sum(1);ww=(w*w).sum(1);det=vv*ww-vw*vw
    for lo in range(0,len(q),batch):
        hi=min(len(q),lo+batch);x=q[lo:hi];best=np.full((len(x),len(tri)),np.inf);near=np.zeros((len(x),len(tri),3))
        if not projected:
            d=x[:,None]-a;signed=(d*normal).sum(2)/np.maximum(nn,1e-30)
            plane=x[:,None]-signed[:,:,None]*normal
            av=((plane-a)*v).sum(2);aw=((plane-a)*w).sum(2)
            u=(av*ww-aw*vw)/np.maximum(det,1e-30);t=(aw*vv-av*vw)/np.maximum(det,1e-30)
            inside=(nn>1e-24)&(u>=0)&(t>=0)&(u+t<=1)
            square=((x[:,None]-plane)**2).sum(2);best=np.where(inside,square,best);near=np.where(inside[:,:,None],plane,near)
        edge_results=[]
        dimensions=2 if projected else 3
        for edge in range(3):
            start=tri[:,edge];vector=tri[:,(edge+1)%3]-start
            delta=x[:,None,:dimensions]-start[:,:dimensions]
            t=np.clip((delta*vector[:,:dimensions]).sum(2)/np.maximum((vector[:,:dimensions]**2).sum(1),1e-30),0,1)
            candidate=start+t[:,:,None]*vector
            square=((x[:,None,:dimensions]-candidate[:,:,:dimensions])**2).sum(2)
            take=square<best;best=np.where(take,square,best);near=np.where(take[:,:,None],candidate,near)
            if projected:edge_results.append((square,candidate[:,:,2]))
        indices=best.argmin(1);value=best[np.arange(len(x)),indices]
        result[lo:hi]=np.sqrt(value);point[lo:hi]=near[np.arange(len(x)),indices]
        if projected:
            lows=np.full(len(x),np.inf);highs=np.full(len(x),-np.inf)
            # Fixed geometric tie tolerance, unrelated to weld/process thresholds.
            for square,height in edge_results:
                tied=np.sqrt(square)<=np.sqrt(value)[:,None]+1e-8
                lows=np.minimum(lows,np.where(tied,height,np.inf).min(1))
                highs=np.maximum(highs,np.where(tied,height,-np.inf).max(1))
            zlo[lo:hi]=lows-x[:,2];zhi[lo:hi]=highs-x[:,2]
        else:
            zlo[lo:hi]=point[lo:hi,2]-x[:,2];zhi[lo:hi]=zlo[lo:hi]
    assert np.isfinite(result).all() and np.isfinite(point).all()
    return dict(distance_mm=result,closest_point=point,minimum_height_mm=zlo,maximum_height_mm=zhi)


def build(packet,solids):
    start=time.perf_counter();n=packet['offsets'][-1];values=np.zeros((n,len(NAMES)),np.float32)
    pose=packet.get('pose',{});T=np.asarray(pose.get('transform_STEP_to_XYZ',[]))
    if not(solids and pose.get('passed')and T.shape==(4,4)):
        return values,dict(available=False,reason='NO_QUALIFIED_FINITE_STEP',queries=0,seconds=time.perf_counter()-start)
    B=shadow.basis(packet['base_axis']);origin=T[:3,3];meshes=[];height=[];boxes=[]
    for solid in solids:
        cad.BRepTools.Clean_s(solid)
        tri=shadow.triangles(solid,DEFLECTION_MM);tri=((tri@T[:3,:3].T+T[:3,3])-origin)@B
        meshes.append(tri);boxes.append(np.r_[tri.min(axis=(0,1)),tri.max(axis=(0,1))]);height.append([tri[:,:,2].min(),tri[:,:,2].max()])
    boxes=np.asarray(boxes);height=np.asarray(height);details=[];queries=0
    for ci,candidate in enumerate(packet['candidates']):
        lo,hi=packet['offsets'][ci:ci+2]
        if not(candidate.get('STEP')and candidate.get('pose')):
            details.append(dict(candidate=ci,available=False));continue
        points=(np.asarray(candidate['stations'])-origin)@B
        ss=np.asarray(candidate['station_s_mm']);owners=np.asarray([shadow.owner_at(ci,float(s),packet)for s in ss])
        if np.any(owners<0)or np.any(owners>=len(solids)):raise ValueError('Qualified source has unresolved owner')
        # Only material AABBs prune exact triangle distances; no footprint fill.
        lower=np.linalg.norm(np.maximum(np.maximum(boxes[:,:3][None]-points[:,None],points[:,None]-boxes[:,3:][None]),0),axis=2)
        distances=np.full((len(points),len(solids)),np.inf);relative=np.zeros_like(distances)
        second=np.full(len(points),np.inf)
        for mi in np.argsort(np.median(lower,axis=0),kind='stable'):
            mi=int(mi);indices=np.flatnonzero((owners!=mi)&(lower[:,mi]<=second+1e-7))
            if not len(indices):continue
            q=closest(points[indices],meshes[mi]);queries+=len(indices)
            distances[indices,mi]=q['distance_mm'];relative[indices,mi]=q['minimum_height_mm']
            if len(solids)>2:second=np.partition(distances,1,axis=1)[:,1]
        available=np.isfinite(distances)
        order=[]
        # Geometric descriptors resolve distance ties without material IDs.
        spans=height[:,1]-height[:,0]
        for i,row in enumerate(distances):
            ids=np.flatnonzero(available[i]);ids=sorted(ids,key=lambda mi:(round(float(row[mi]),7),
                round(float(spans[mi]),7),round(float(height[mi,0]-points[i,2]),7),round(float(relative[i,mi]),7)))
            order.append(ids[:2])
        for i,ids in enumerate(order):
            if not ids:continue
            mate=ids[0];owner=int(owners[i]);z=points[i,2]
            pair=sorted([owner,mate],key=lambda mi:(round(float(spans[mi]),7),round(float(height[mi,0]-z),7)))
            values[lo+i,:6]=[1.,np.log1p(distances[i,mate]),*slog([z-height[pair[0],0],height[pair[0],1]-z,z-height[pair[1],0],height[pair[1],1]-z])]
            if len(ids)>1:
                third=ids[1];values[lo+i,6:9]=[1.,np.log1p(distances[i,third]),slog(relative[i,third])]
        for mi in range(len(solids)):
            indices=np.asarray([i for i,ids in enumerate(order)if len(ids)>1 and ids[1]==mi],int)
            if not len(indices):continue
            q=closest(points[indices],meshes[mi],projected=True);queries+=len(indices)
            values[lo+indices,9]=np.log1p(q['distance_mm']);values[lo+indices,10]=slog(q['minimum_height_mm']);values[lo+indices,11]=slog(q['maximum_height_mm'])
        details.append(dict(candidate=ci,available=True,rows=hi-lo,
            pair_available_rows=int((values[lo:hi,0]>0).sum()),third_available_rows=int((values[lo:hi,6]>0).sum())))
    assert np.isfinite(values).all()
    return values,dict(available=True,names=list(NAMES),mesh_deflection_mm=DEFLECTION_MM,materials=len(solids),
        mesh_triangles=sum(len(q)for q in meshes),queries=queries,candidates=details,seconds=time.perf_counter()-start,
        whole_material_extrema_not_local_board_attributes=True,third_projection_is_boundary_not_filled_footprint=True,
        nearest_materials_not_confirmed_joint=True,no_weld_or_clearance_decision=True,no_labels_or_models_read=True)
