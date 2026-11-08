"""Input-only normal directions and 1D plane-offset correlation registration.

Native NURBS control boxes can greatly overestimate occupied material extent.
This registration uses actual finite triangulations and local XYZ plane normals.
No weld/reference curves or fitted models participate.
"""
import itertools,time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.signal import correlate
from . import curveonly_01a1ac7c as cad

def material_samples(solids,count=20000):
    triangles=[];normals=[]
    for solid in solids:
        cad.BRepMesh_IncrementalMesh(solid,.3,False,.2,False)
        for fs in cad.items(solid,cad.TopAbs_FACE):
            face=cad.TopoDS.Face_s(fs);loc=cad.TopLoc_Location();tri=cad.BRep_Tool.Triangulation_s(face,loc)
            if tri is None:continue
            nodes=np.array([cad.vec(tri.Node(i).Transformed(loc.Transformation()))for i in range(1,tri.NbNodes()+1)])
            indices=np.array([tri.Triangle(i).Get()for i in range(1,tri.NbTriangles()+1)],int)-1
            vertices=nodes[indices];cross=np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]);area=np.linalg.norm(cross,axis=1);keep=area>1e-10
            triangles.extend(vertices[keep]);normals.extend(cross[keep]/area[keep,None])
    triangles=np.array(triangles);normals=np.array(normals);area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
    rng=np.random.default_rng(20261106);ids=rng.choice(len(triangles),count,p=area/area.sum());uv=rng.random((count,2));uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1]
    p=triangles[ids,0]+uv[:,0,None]*(triangles[ids,1]-triangles[ids,0])+uv[:,1,None]*(triangles[ids,2]-triangles[ids,0])
    return p,normals[ids]

def axes(normals):
    n=np.array(normals,copy=True);main=np.argmax(abs(n),axis=1);n*=np.where(n[np.arange(len(n)),main]<0,-1,1)[:,None]
    rounded=np.round(n*40).astype(int);unique,counts=np.unique(rounded,return_counts=True,axis=0);directions=[]
    for idx in np.argsort(counts)[::-1]:
        u=unique[idx].astype(float);u/=np.linalg.norm(u)
        if directions and abs(u@directions[0])>.15:continue
        take=abs(n@u)>.999;v=n[take].mean(0);v/=np.linalg.norm(v)
        directions.append(v)
        if len(directions)==2:break
    if len(directions)<2:raise ValueError('Two independent observed plane-normal families unavailable')
    a=directions[0];b=directions[1]-a*(directions[1]@a);b/=np.linalg.norm(b);c=np.cross(a,b)
    return np.column_stack([a,b,c])

def offset_histogram(source,source_normals,target,target_normals,basis):
    shift=np.zeros(3);evidence=[]
    for axis in range(3):
        u=basis[:,axis];a=source[abs(source_normals@u)>.995]@u;b=target[abs(target_normals@u)>.995]@u
        if min(len(a),len(b))<12:evidence.append(dict(axis=axis,available=False));continue
        lo=min(a.min(),b.min())-10;hi=max(a.max(),b.max())+10;step=1.;n=int(np.ceil((hi-lo)/step));edges=lo+np.arange(n+1)*step
        sa=np.sqrt(np.histogram(a,edges)[0]);sb=np.sqrt(np.histogram(b,edges)[0]);cross=correlate(sb,sa,mode='full',method='fft');lags=np.arange(len(cross))-(len(sa)-1)
        allowed=abs(lags)<=max(100.,.15*(hi-lo));cross[~allowed]=-1
        best=int(np.argmax(cross));delta=float(lags[best]*step);shift+=delta*u;evidence.append(dict(axis=axis,available=True,offset_mm=delta))
    return shift,evidence

def choose_pose(solids,xyz,provided=None):
    start=time.perf_counter();boxes=np.array([cad.bbox(s)for s in solids]);trials=[]
    initial=[np.array(provided,float)]if provided is not None else []
    initial.append(np.eye(4))
    for T in initial:
        q,dist=cad.pose_quality(solids,boxes,xyz,T);trials.append((q,T,dist,'DECLARED_OR_IDENTITY'))
        if q<=.1:return T,dict(sample_p95_mm=q,passed=True,sample_distances_mm=dist,transform_STEP_to_XYZ=T.tolist(),input_only=True,globally_unique_pose_proved=False,method='DECLARED_OR_IDENTITY',seconds=time.perf_counter()-start)
    samples,ns=material_samples(solids);tree=cKDTree(xyz,balanced_tree=False);rng=np.random.default_rng(20261106);ids=np.sort(rng.choice(len(xyz),min(10000,len(xyz)),replace=False));raw=xyz[ids]
    _,near=tree.query(raw,k=16,workers=1);neighbor=xyz[near];delta=neighbor-neighbor.mean(1)[:,None,:];cov=np.einsum('nki,nkj->nij',delta,delta)/16
    eigen,vectors=np.linalg.eigh(cov);reliable=eigen[:,0]<np.maximum(eigen[:,1],1e-10)*.05;rn=vectors[reliable,:,0];observed=raw[reliable]
    native_axes=axes(ns);raw_axes=axes(rn);nc=samples.mean(0);rc=raw.mean(0);centered=raw-rc;material_tree=cKDTree(samples);proposals=[]
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1.,1.),repeat=3):
            R=raw_axes[:,perm]@np.diag(signs)@native_axes.T
            if np.linalg.det(R)<0:continue
            source=(samples-nc)@R.T;norm=ns@R.T;shift,evidence=offset_histogram(source,norm,observed-rc,rn,raw_axes)
            T=np.eye(4);T[:3,:3]=R;T[:3,3]=rc+shift-nc@R.T
            moved=samples[::16]@R.T+T[:3,3];d=tree.query(moved,workers=1,distance_upper_bound=300.)[0];score=float(np.quantile(np.minimum(d,300.),.8));proposals.append((score,T,evidence))
    ordered=sorted(proposals,key=lambda x:x[0])
    for rank,(score,T,evidence) in enumerate(ordered[:3]):
        U=np.linalg.inv(T);source=raw[::2]
        for iteration in range(16):
            p=source@U[:3,:3].T+U[:3,3];dist,j=material_tree.query(p,workers=1);n=ns[j]
            take=dist<=max(np.quantile(dist,.95),5.)
            center=p[take].mean(0);A=np.c_[np.cross(p[take]-center,n[take]),n[take]];rhs=((samples[j[take]]-p[take])*n[take]).sum(1)
            # Normal-family balancing retains the informative upright planes.
            dominant=np.argmax(abs(n[take]@native_axes),axis=1);counts=np.bincount(dominant,minlength=3);w=1/np.sqrt(np.maximum(counts[dominant],1))
            solution=np.linalg.lstsq(A*w[:,None],rhs*w,rcond=None)[0];angle=np.linalg.norm(solution[:3])
            if angle>.05:solution[:3]*=.05/angle
            D=np.eye(4);D[:3,:3]=Rotation.from_rotvec(solution[:3]).as_matrix();D[:3,3]=center+solution[3:]-center@D[:3,:3].T;U=D@U
            if np.linalg.norm(solution)<1e-5:break
        fitted=np.linalg.inv(U);q,dist=cad.pose_quality(solids,boxes,xyz,fitted);trials.append((q,fitted,dist,'NORMAL_HISTOGRAM_CENTERED_ICP'))
        if rank==0 and q<=1. and ordered[1][0]-score>=10.:
            return fitted,dict(sample_p95_mm=q,passed=True,sample_distances_mm=dist,transform_STEP_to_XYZ=fitted.tolist(),input_only=True,globally_unique_pose_proved=False,method='NORMAL_HISTOGRAM_CENTERED_ICP',normal_families_from_XYZ=len(observed),trial_p95=[v[0]for v in trials],histogram_ranks=[v[0]for v in ordered],early_stop='Independent finite-distance p95<=1mm and runner-up input mesh rank separated by>=10mm; frozen TRAIN-supported cost control.',seconds=time.perf_counter()-start)
        # Verify the exact histogram alignment too; mesh NN is not a truth.
        qq,dd=cad.pose_quality(solids,boxes,xyz,T);trials.append((qq,T,dd,'NORMAL_HISTOGRAM'))
    q,T,dist,origin=min(trials,key=lambda x:x[0])
    return T,dict(sample_p95_mm=q,passed=q<=3.,sample_distances_mm=dist,transform_STEP_to_XYZ=T.tolist(),input_only=True,globally_unique_pose_proved=False,method=origin,normal_families_from_XYZ=len(observed),trial_p95=[v[0]for v in trials],histogram_ranks=[v[0]for v in sorted(proposals,key=lambda x:x[0])],seconds=time.perf_counter()-start)
