"""Input-only cylinders and finite plane intersections from stable 2D neighborhoods."""
import numpy as np
from scipy.spatial import cKDTree,Delaunay,QhullError
from weld_pipeline.sjg_planar_boundary_curves_894 import fit_circle
from weld_pipeline.contract_curve_geometry import CircularArc

def canonical_normals(xyz,progress=None):
    grid=np.floor((xyz-xyz.min(0))/1.).astype(np.int32)
    _,ids,inverse=np.unique(grid,axis=0,return_index=True,return_inverse=True)
    points=xyz[ids];tree=cKDTree(points);normal=np.empty_like(points);eigen=np.empty_like(points)
    for begin in range(0,len(points),4096):
        end=min(begin+4096,len(points));near=tree.query(points[begin:end],k=min(32,len(points)),workers=2)[1]
        cloud=points[near];delta=cloud-cloud.mean(1)[:,None]
        values,vectors=np.linalg.eigh(np.einsum('nki,nkj->nij',delta,delta)/cloud.shape[1])
        normal[begin:end]=vectors[:,:,0];eigen[begin:end]=values
        if progress and begin%65536==0:progress('CANONICAL_NORMALS',end,len(points))
    normal*=np.sign(normal[np.arange(len(normal)),np.argmax(abs(normal),axis=1)])[:,None]
    rank2=eigen[:,1]/eigen[:,2].clip(1e-12)>=.05
    planar=rank2&(eigen[:,0]/eigen[:,1].clip(1e-12)<=.02)
    return dict(original_ids=ids.astype(np.int64),inverse=inverse.astype(np.int64),xyz=points,
                normal=normal,eigen=eigen,rank2=rank2,planar=planar)

def dominant_axes(data,maximum=8):
    usable=data['normal'][data['planar']]
    keys,inverse,counts=np.unique(np.round(usable/.02).astype(np.int32),axis=0,return_inverse=True,return_counts=True)
    result=[]
    for k in np.argsort(-counts):
        if counts[k]<32:continue
        n=np.median(usable[inverse==k],axis=0);n/=np.linalg.norm(n)
        if any(abs(n@old)>np.cos(np.deg2rad(2.)) for old in result):continue
        result.append(n)
        if len(result)>=maximum:break
    return result

def basis_for(axis):
    seed=np.eye(3)[np.argmin(abs(axis))];u=np.cross(axis,seed);u/=np.linalg.norm(u)
    return np.stack([u,np.cross(axis,u)],axis=-1)

def columns(xyz,axis):
    origin=xyz.mean(0);basis=basis_for(axis);p=xyz-origin;uv=p@basis;height=p@axis
    keys=np.c_[np.round(uv/.1).astype(np.int32),np.round(height/.1).astype(np.int32)]
    keys,representative=np.unique(keys,axis=0,return_index=True)
    _,inverse,count=np.unique(keys[:,:2],axis=0,return_inverse=True,return_counts=True)
    low=np.full(len(count),np.inf);high=np.full(len(count),-np.inf)
    np.minimum.at(low,inverse,height[representative]);np.maximum.at(high,inverse,height[representative])
    means=np.stack([np.bincount(inverse,weights=uv[representative,i],minlength=len(count))/count for i in range(2)],axis=1)
    valid=(count>=3)&(high-low>=1.)
    return dict(origin=origin,basis=basis,axis=axis,uv=means[valid],low=low[valid],high=high[valid],distinct_heights=count[valid])

def longest_angular_run(points,center,radius):
    angle=np.mod(np.arctan2(points[:,1]-center[1],points[:,0]-center[0]),2*np.pi)
    order=np.argsort(angle);sorted_angle=angle[order]
    gaps=np.diff(np.r_[sorted_angle,sorted_angle[0]+2*np.pi]);cut=int(np.argmax(gaps))
    order=np.r_[order[cut+1:],order[:cut+1]];unwrapped=np.unwrap(angle[order])
    dif=np.diff(unwrapped);positive=dif[dif>1e-10]
    if not len(positive):return np.empty(0,np.int64),0.
    split=np.flatnonzero(dif>max(5*np.median(positive),2./radius))+1
    groups=np.split(np.arange(len(order)),split);best=max(groups,key=len)
    span=float(unwrapped[best[-1]]-unwrapped[best[0]])
    return order[best],span

def fit_column_circles(col,seed=919,maximum_trials=4096,progress=None):
    uv=col['uv'];rng=np.random.default_rng(seed);circles=[]
    if len(uv)<32:return circles
    tree=cKDTree(uv);anchors=rng.choice(len(uv),min(maximum_trials//2,len(uv)),replace=False)
    nearest=tree.query(uv[anchors],k=min(32,len(uv)))[1]
    for trial in range(len(anchors)*2):
        if progress and trial%128==0:progress('CYLINDER_LOCAL_TRIAL',trial+1,len(anchors)*2,'HYPOTHESES',len(circles))
        anchor=trial//2;near=nearest[anchor]
        picks=np.r_[near[0],rng.choice(near[1:],2,replace=False)]
        sample=uv[picks];centered=sample-sample.mean(0);s=np.linalg.svd(centered,compute_uv=False)
        if s[0]<=1e-8 or s[1]/s[0]<1e-6:continue
        fitted=fit_circle(sample)
        if fitted is None:continue
        center,radius,_=fitted
        # A good existing primitive already explains this triple.
        if any(np.linalg.norm(center-c['center'])<=.2 and abs(radius-c['radius'])<=.2 for c in circles):continue
        support=None;span=0.
        for _ in range(3):
            ids=np.flatnonzero(abs(np.linalg.norm(uv-center,axis=1)-radius)<=.2)
            if len(ids)<16:break
            run,span=longest_angular_run(uv[ids],center,radius);ids=ids[run]
            if len(ids)<16 or radius*(1-np.cos(min(span,np.pi)/2))<.6:break
            refit=fit_circle(uv[ids])
            if refit is None:break
            center,radius,_=refit;support=ids
        if support is None:continue
        error=abs(np.linalg.norm(uv[support]-center,axis=1)-radius)
        support=support[error<=.2]
        if len(support)<16:continue
        run,span=longest_angular_run(uv[support],center,radius);support=support[run]
        if len(support)<16 or radius*(1-np.cos(min(span,np.pi)/2))<.6:continue
        value=dict(center=center,radius=float(radius),column_ids=support,angular_span=float(span))
        duplicates=[i for i,c in enumerate(circles) if np.linalg.norm(center-c['center'])<=.5 and abs(radius-c['radius'])<=.5]
        if duplicates:
            i=duplicates[0]
            if len(support)>len(circles[i]['column_ids']):circles[i]=value
        else:circles.append(value)
        if progress and len(circles)%10==0:progress('COLUMN_CIRCLE_HYPOTHESES',trial+1,len(anchors)*2,len(circles))
    return circles

def plane_levels(data,col):
    dot=abs(data['normal']@col['axis']);select=data['planar']&(dot>=np.cos(np.deg2rad(2.)))
    ids=np.flatnonzero(select);height=(data['xyz'][ids]-col['origin'])@col['axis']
    bins,counts=np.unique(np.round(height/.2).astype(np.int64),return_counts=True)
    lookup={int(b):int(c) for b,c in zip(bins,counts)};levels=[]
    for b,count in zip(bins,counts):
        if count<32 or count<lookup.get(int(b)-1,0) or count<lookup.get(int(b)+1,0):continue
        close=abs(height-b*.2)<=.3
        level=float(np.median(height[close]))
        if any(abs(level-old)<.3 for old in levels):continue
        levels.append(level)
    return levels,ids,height

def finite_plane(data,col,level,eligible,height):
    ids=eligible[abs(height-level)<=.2];uv=(data['xyz'][ids]-col['origin'])@col['basis']
    _,unique=np.unique(np.round(uv,5),axis=0,return_index=True);uv=uv[unique];ids=ids[unique]
    if len(ids)<16:return None
    try:tri=Delaunay(uv)
    except QhullError:return None
    distance=cKDTree(uv).query(uv,k=min(24,len(uv)),workers=2)[0];radius=distance[:,-1]
    keep=np.ones(len(tri.simplices),bool);q=uv[tri.simplices]
    for i,j in ((0,1),(1,2),(2,0)):
        keep &= np.linalg.norm(q[:,i]-q[:,j],axis=1)<=np.maximum(2.,1.5*np.minimum(radius[tri.simplices[:,i]],radius[tri.simplices[:,j]]))
    if not keep.any():return None
    center=col['origin']+level*col['axis']
    descriptor=dict(normal=col['axis'],center=center,offset=float(center@col['axis']),basis=col['basis'],
        point_ids=data['original_ids'][ids],triangles=tri.simplices[keep],proposal_branch='canonical_geometry',
        primitive_kind='plane',material_id=None,unresolved_subsampling_holes=True)
    return descriptor,tri,keep

def supported_intervals(q,mask):
    runs=[];start=None
    for i,on in enumerate(mask):
        if on and start is None:start=i
        if start is not None and (not on or i==len(mask)-1):
            end=i if on else i-1
            if end>start:runs.append([float(q[start]),float(q[end])])
            start=None
    return runs

def extract(xyz,progress=None):
    data=canonical_normals(xyz,progress);axes=dominant_axes(data);surfaces=[];profiles=[];cylinders=[];stats=[]
    for ai,axis in enumerate(axes):
        col=columns(xyz,axis)
        if progress:progress('CYLINDER_AXIS_COLUMNS',ai+1,len(axes),len(col['uv']))
        circles=fit_column_circles(col,919+ai,progress=progress)
        levels,eligible,height=plane_levels(data,col)
        row=dict(axis=axis.tolist(),columns=len(col['uv']),circle_hypotheses=len(circles),plane_levels=levels);stats.append(row)
        if progress:progress('OBSERVED_CYLINDERS',ai+1,'CIRCLES',len(circles),'LEVELS',len(levels))
        for ci,circle in enumerate(circles):
            cylinder=dict(axis=axis.tolist(),axis_origin_mm=(col['origin']+col['basis']@circle['center']).tolist(),
                radius_mm=circle['radius'],support_columns=len(circle['column_ids']),axis_family=ai,
                axial_min_mm=float(col['low'][circle['column_ids']].min()),axial_max_mm=float(col['high'][circle['column_ids']].max()),
                status='OBSERVED_CYLINDER_HYPOTHESIS_MATERIAL_UNKNOWN')
            cylinders.append(cylinder);circle['cylinder_id']=len(cylinders)-1
        for level in levels:
            mesh=finite_plane(data,col,level,eligible,height)
            if mesh is None:continue
            face,tri,keep=mesh;face_index=len(surfaces);surfaces.append(face)
            for circle in circles:
                columns_ids=circle['column_ids']
                ids=columns_ids[(col['low'][columns_ids]-.2<=level)&(col['high'][columns_ids]+.2>=level)]
                if len(ids)<16:continue
                run,span=longest_angular_run(col['uv'][ids],circle['center'],circle['radius']);ids=ids[run]
                if len(ids)<16 or span<=0:continue
                angles=np.mod(np.arctan2(col['uv'][ids,1]-circle['center'][1],col['uv'][ids,0]-circle['center'][0]),2*np.pi)
                gaps=np.diff(np.r_[np.sort(angles),np.sort(angles)[0]+2*np.pi])
                positive=gaps[gaps>1e-10]
                closed=bool(span>1.9*np.pi and len(positive) and gaps.max()<=3*np.median(positive))
                start_direction=col['basis']@np.array([np.cos(angles[0]),np.sin(angles[0])])
                arc=CircularArc(col['origin']+col['basis']@circle['center']+level*axis,circle['radius'],axis,start_direction,2*np.pi if closed else span,closed)
                length=arc.length_mm;q=np.linspace(0,length,int(np.ceil(length))+1)
                uv=(arc.point_at(q)-col['origin'])@col['basis'];which=tri.find_simplex(uv)
                finite=(which>=0)&keep[np.maximum(which,0)]
                distance=cKDTree(col['uv'][ids]).query(uv)[0]
                # Nearby columns establish observed cylindrical support. Larger
                # gaps remain explicit; the full analytic carrier is not a weld.
                step=cKDTree(col['uv'][ids]).query(col['uv'][ids],k=2)[0][:,1]
                near=distance<=max(2.,3.*float(np.median(step[step>1e-9])))
                intervals=supported_intervals(q,finite&near)
                if not intervals:continue
                gaps=[[a[1],b[0]] for a,b in zip(intervals[:-1],intervals[1:])]
                profiles.append(dict(representation=dict(kind='arc',arc_parameters=arc.to_parameters()),length_mm=length,
                    face_indices=[face_index],support_normal=axis.tolist(),support_intervals_mm=intervals,unobserved_gaps_mm=gaps,
                    cylinder_index=circle['cylinder_id'],cylinder_hypothesis=cylinders[circle['cylinder_id']],
                    second_contact_face_unknown=False,second_contact_primitive='cylinder',material_pair=None,weld_type=None,
                    proposal_branch='canonical_cylinder_plane',status='UNCLASSIFIED_FINITE_PLANE_CYLINDER_INTERSECTION',
                    full_carrier_closure_not_weld_closure=True,axial_sampling_gaps_unresolved=True))
            if progress:progress('FINITE_PLANE_CYLINDER_LEVEL',ai+1,level,'ARC_CARRIERS',len(profiles))
    return surfaces,profiles,cylinders,dict(original_points=len(xyz),canonical_points=len(data['xyz']),
        canonical_rank2_points=int(data['rank2'].sum()),canonical_planar_points=int(data['planar'].sum()),axis_search_budget=8,axes=stats)
