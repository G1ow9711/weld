"""Finite closest-material direction envelopes; no welding-permission rule."""
import time
import numpy as np
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_local_material_v94 as original
from . import curveonly_01a1ac7c_exact_mesh_index_v110 as indexed
from . import curveonly_01a1ac7c_finite_shadow_v73 as shadow
from . import curveonly_01a1ac7c_native_reference_v154 as reference

TIE_MM=1e-8
LAYER_TIE_MM=1e-7
DIRECTION_EPS_MM=1e-8
DEFLECTION_MM=original.DEFLECTION_MM
LAYER_NAMES=('available','material_count_log','distance_log_mm','direction_available',
             'abs_tangent_min','abs_tangent_max','abs_base_min','abs_base_max',
             'abs_radial_min','abs_radial_max')
NAMES=('geometry_available','owner_known','frame_available')+tuple(
    f'nearest_layer{layer}_{name}'for layer in (1,2)for name in LAYER_NAMES)
assert len(NAMES)==23


def closest_envelope(points,mesh,frames):
    """V94 continuous triangle distances, with all tied closest directions.

    A direction is query-to-closest-point, not an oriented surface normal.
    Finite edges, holes, medial-axis ties and zero-distance UNKNOWN remain.
    """
    q=np.asarray(points,float);tri=np.asarray(mesh,float);frames=np.asarray(frames,float)
    assert frames.shape==(len(q),3,3)
    result=dict(distance_mm=np.empty(len(q)),direction_available=np.empty(len(q),bool),
                direction_min=np.zeros((len(q),3)),direction_max=np.zeros((len(q),3)))
    a=tri[:,0];v=tri[:,1]-a;w=tri[:,2]-a;normal=np.cross(v,w);nn=(normal*normal).sum(1)
    vv=(v*v).sum(1);vw=(v*w).sum(1);ww=(w*w).sum(1);det=vv*ww-vw*vw
    batch=max(1,min(64,100000//max(1,len(tri))))
    for lo in range(0,len(q),batch):
        hi=min(len(q),lo+batch);x=q[lo:hi]
        best=np.full((len(x),len(tri)),np.inf);near=np.zeros((len(x),len(tri),3))
        delta=x[:,None]-a;signed=(delta*normal).sum(2)/np.maximum(nn,1e-30)
        plane=x[:,None]-signed[:,:,None]*normal
        av=((plane-a)*v).sum(2);aw=((plane-a)*w).sum(2)
        u=(av*ww-aw*vw)/np.maximum(det,1e-30);t=(aw*vv-av*vw)/np.maximum(det,1e-30)
        inside=(nn>1e-24)&(u>=0)&(t>=0)&(u+t<=1)
        square=((x[:,None]-plane)**2).sum(2);best=np.where(inside,square,best);near=np.where(inside[:,:,None],plane,near)
        for edge in range(3):
            start=tri[:,edge];vector=tri[:,(edge+1)%3]-start
            delta=x[:,None]-start
            t=np.clip((delta*vector).sum(2)/np.maximum((vector*vector).sum(1),1e-30),0,1)
            candidate=start+t[:,:,None]*vector;square=((x[:,None]-candidate)**2).sum(2)
            take=square<best;best=np.where(take,square,best);near=np.where(take[:,:,None],candidate,near)
        indices=best.argmin(1);value=best[np.arange(len(x)),indices];distance=np.sqrt(value)
        result['distance_mm'][lo:hi]=distance
        each=np.sqrt(best);tied=each<=distance[:,None]+TIE_MM
        valid=tied&(each>DIRECTION_EPS_MM);available=valid.any(1)
        result['direction_available'][lo:hi]=available
        delta=near-x[:,None]
        for axis in range(3):
            align=np.clip(np.abs((delta*frames[lo:hi,axis,None,:]).sum(2))/np.maximum(each,1e-30),0,1)
            low=np.where(valid,align,np.inf).min(1);high=np.where(valid,align,-np.inf).max(1)
            result['direction_min'][lo:hi,axis]=np.where(available,low,0.)
            result['direction_max'][lo:hi,axis]=np.where(available,high,0.)
    assert np.isfinite(result['distance_mm']).all()and np.isfinite(result['direction_min']).all()and np.isfinite(result['direction_max']).all()
    return result


def closest(points,mesh,frames,cache=None):
    points=np.asarray(points,float);mesh=np.asarray(mesh,float);frames=np.asarray(frames,float);cache={}if cache is None else cache
    key=id(mesh);entry=cache.get(key)
    if entry is None or entry['mesh']is not mesh:entry=indexed.index(mesh,False);cache[key]=entry
    _,seed=entry['tree'].query(points,workers=1);seed=np.asarray(seed).reshape(-1);upper=np.empty(len(points))
    for ti in np.unique(seed):
        rows=np.flatnonzero(seed==ti);upper[rows]=original.closest(points[rows],mesh[ti:ti+1])['distance_mm']
    selected=[{int(ti)}for ti in seed]
    for group in entry['groups']:
        near=group['tree'].query_ball_point(points,upper+group['radius']+indexed.PRUNE_GUARD_MM,workers=1)
        for values,local in zip(selected,near):values.update(map(int,group['ids'][local]))
    batches={}
    for qi,ids in enumerate(selected):batches.setdefault(tuple(sorted(ids)),[]).append(qi)
    out=dict(distance_mm=np.empty(len(points)),direction_available=np.empty(len(points),bool),
             direction_min=np.zeros((len(points),3)),direction_max=np.zeros((len(points),3)))
    for ids,rows in batches.items():
        rows=np.asarray(rows,int);part=closest_envelope(points[rows],mesh[np.asarray(ids,int)],frames[rows])
        for name in out:out[name][rows]=part[name]
    return out


def layers(distance,direction_available,direction_min,direction_max,frame_available):
    """Permutation-invariant nearest two distinct distance layers."""
    d=np.asarray(distance,float);available=np.isfinite(d);n=len(d)
    out=np.zeros((n,2*len(LAYER_NAMES)),np.float32)
    first=d.min(1);second=np.where(d>first[:,None]+LAYER_TIE_MM,d,np.inf).min(1)
    for layer,target in enumerate((first,second)):
        finite=np.isfinite(target);difference=np.full(d.shape,np.inf)
        np.subtract(d,target[:,None],out=difference,where=available&finite[:,None])
        tied=available&(np.abs(difference)<=LAYER_TIE_MM)
        valid=tied&direction_available&np.asarray(frame_available)[:,None]
        has=valid.any(1);offset=layer*len(LAYER_NAMES)
        out[:,offset]=finite;out[:,offset+1]=np.log1p(tied.sum(1))
        out[:,offset+2]=np.log1p(np.where(finite,target,0.));out[:,offset+3]=has
        for axis in range(3):
            low=np.where(valid,direction_min[:,:,axis],np.inf).min(1)
            high=np.where(valid,direction_max[:,:,axis],-np.inf).max(1)
            out[:,offset+4+axis*2]=np.where(has,low,0.)
            out[:,offset+5+axis*2]=np.where(has,high,0.)
    return out


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
        tangent=np.gradient(points,axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-12)
        base=np.broadcast_to([0.,0.,1.],points.shape);radial=np.cross(tangent,base);norm=np.linalg.norm(radial,axis=1)
        frame=norm>1e-10;radial/=np.maximum(norm[:,None],1e-12);frames=np.stack([tangent,base,radial],axis=1)
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
