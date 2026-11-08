"""Input-only, bounded projected-hull pose rescue after the old pose fails."""
import math,time,types
import numpy as np
from scipy.spatial import cKDTree,ConvexHull
from scipy.spatial.transform import Rotation
from . import curveonly_01a1ac7c_registration_v2 as previous
cad=previous.cad


def plane_basis(normal):
    n=np.asarray(normal,float);n=n/np.linalg.norm(n);seed=np.eye(3)[np.argmin(abs(n))]
    u=seed-n*(seed@n);u/=np.linalg.norm(u)
    return np.column_stack([u,np.cross(n,u),n])


def rectangles(points,normal):
    basis=plane_basis(normal);p=np.asarray(points)@basis
    # Bounded hull population plus exact16direction support witnesses.
    q=p[::max(1,int(math.ceil(len(p)/100000))),:2]
    directions=np.column_stack([np.cos(np.arange(16)*math.pi/8),np.sin(np.arange(16)*math.pi/8)])
    witnesses=[int(np.argmax(p[:,:2]@u))for u in directions]
    hull=np.vstack([q,p[witnesses,:2]]);hull=hull[ConvexHull(hull).vertices]
    edges=np.roll(hull,-1,axis=0)-hull
    angles=np.unique(np.round(np.mod(np.arctan2(edges[:,1],edges[:,0]),math.pi/2),10))
    choices=[]
    for theta in angles:
        u=np.array([math.cos(theta),math.sin(theta)]);v=np.array([-u[1],u[0]])
        a=hull@u;b=hull@v;extent=np.array([np.ptp(a),np.ptp(b)])
        center2=((a.max()+a.min())/2)*u+((b.max()+b.min())/2)*v
        frame=basis@np.array([[u[0],v[0],0],[u[1],v[1],0],[0,0,1.]])
        center=np.r_[center2,(p[:,2].max()+p[:,2].min())/2]@basis.T
        choices.append(dict(theta=float(theta),area=float(extent.prod()),center=center,frame=frame))
    chosen=[]
    for c in sorted(choices,key=lambda c:c['area']):
        if any(abs(math.atan2(math.sin(4*(c['theta']-o['theta'])),math.cos(4*(c['theta']-o['theta']))))/4<math.pi/180 for o in chosen):continue
        chosen.append(c)
        if len(chosen)==3:break
    return chosen


def material_vertices(solids):
    out=[]
    for solid in solids:
        for fs in cad.items(solid,cad.TopAbs_FACE):
            face=cad.TopoDS.Face_s(fs);location=cad.TopLoc_Location();tri=cad.BRep_Tool.Triangulation_s(face,location)
            if tri is None:continue
            out.extend(cad.vec(tri.Node(i).Transformed(location.Transformation()))for i in range(1,tri.NbNodes()+1))
    return np.asarray(out)


def raw_normals(xyz):
    tree=cKDTree(xyz,balanced_tree=False);rng=np.random.default_rng(20261106)
    ids=np.sort(rng.choice(len(xyz),min(10000,len(xyz)),replace=False));raw=xyz[ids]
    _,near=tree.query(raw,k=16,workers=1);neighbor=xyz[near];d=neighbor-neighbor.mean(1)[:,None,:]
    values,vectors=np.linalg.eigh(np.einsum('nki,nkj->nij',d,d)/16)
    take=values[:,0]<np.maximum(values[:,1],1e-10)*.05
    return raw,vectors[take,:,0],tree


def refine(T,raw,samples,normals,native_axes):
    material_tree=cKDTree(samples);U=np.linalg.inv(T);source=raw[::2]
    for _ in range(16):
        p=source@U[:3,:3].T+U[:3,3];dist,j=material_tree.query(p,workers=1);n=normals[j]
        take=dist<=max(np.quantile(dist,.95),5.);center=p[take].mean(0)
        A=np.c_[np.cross(p[take]-center,n[take]),n[take]];rhs=((samples[j[take]]-p[take])*n[take]).sum(1)
        dominant=np.argmax(abs(n[take]@native_axes),axis=1);counts=np.bincount(dominant,minlength=3)
        w=1/np.sqrt(np.maximum(counts[dominant],1));solution=np.linalg.lstsq(A*w[:,None],rhs*w,rcond=None)[0]
        angle=np.linalg.norm(solution[:3])
        if angle>.05:solution[:3]*=.05/angle
        D=np.eye(4);D[:3,:3]=Rotation.from_rotvec(solution[:3]).as_matrix();D[:3,3]=center+solution[3:]-center@D[:3,:3].T;U=D@U
        if np.linalg.norm(solution)<1e-5:break
    return np.linalg.inv(U)


def rescue(solids,xyz,samples,normals):
    start=time.perf_counter();raw,rn,tree=raw_normals(xyz);na=previous.axes(normals);ra=previous.axes(rn)
    vertices=material_vertices(solids);source_frames=rectangles(vertices,na[:,0]);target_frames=rectangles(xyz,ra[:,0]);proposals=[]
    for si,s in enumerate(source_frames):
        for ti,t in enumerate(target_frames):
            for flip in (False,True):
                for quarter in range(4):
                    local=Rotation.from_rotvec(np.array([0.,0.,quarter*math.pi/2])).as_matrix()
                    if flip:local=local@np.diag([1.,-1.,-1.])
                    R=t['frame']@local@s['frame'].T;T=np.eye(4);T[:3,:3]=R;T[:3,3]=t['center']-R@s['center']
                    moved=samples[::16]@R.T+T[:3,3];d=tree.query(moved,workers=1,distance_upper_bound=300.)[0]
                    rank=float(np.quantile(np.minimum(d,300.),.8));proposals.append((rank,T,dict(native_rectangle=si,raw_rectangle=ti,flip=flip,quarter=quarter)))
    ordered=sorted(proposals,key=lambda p:p[0]);trials=[];boxes=np.array([cad.bbox(s)for s in solids])
    for rank,T,origin in ordered[:3]:
        for stage,U in [('HULL',T),('HULL_ICP',refine(T,raw,samples,normals,na))]:
            q,dist=cad.pose_quality(solids,boxes,xyz,U);trials.append((q,U,dist,stage,rank,origin))
    q,T,dist,stage,rank,origin=min(trials,key=lambda x:(x[0],x[4]))
    return T,dict(sample_p95_mm=q,passed=q<=3.,sample_distances_mm=dist,transform_STEP_to_XYZ=T.tolist(),
        input_only=True,globally_unique_pose_proved=False,method=stage,pose_threshold_unchanged=True,
        candidate_count=len(proposals),maximum_candidates=72,maximum_refined=3,ICP_iterations_limit=16,
        native_hull_frames=len(source_frames),raw_hull_frames=len(target_frames),candidate_rank=rank,origin=origin,
        tested=[dict(p95_mm=v[0],method=v[3],rank=v[4],origin=v[5],transform_STEP_to_XYZ=v[1].tolist())for v in trials],
        seconds=time.perf_counter()-start)


def choose_pose(solids,xyz,provided=None):
    start=time.perf_counter();cached=[]
    def material_samples(*args,**kwargs):
        if not cached:cached.append(previous.material_samples(*args,**kwargs))
        return cached[0]
    function=types.FunctionType(previous.choose_pose.__code__,dict(previous.__dict__,material_samples=material_samples))
    T,audit=function(solids,xyz,provided)
    if audit.get('passed'):return T,audit
    if not cached:cached.append(previous.material_samples(solids))
    U,extra=rescue(solids,xyz,*cached[0])
    if extra['sample_p95_mm']<audit['sample_p95_mm']:
        extra.update(previous_failed_pose=audit,total_pose_seconds=time.perf_counter()-start)
        return U,extra
    audit.update(hull_rescue_rejected=extra,total_pose_seconds=time.perf_counter()-start)
    return T,audit
