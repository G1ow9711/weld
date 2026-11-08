"""Independent compact curve extractor. Inference accepts only material STEP/XYZ.

No project model, profile bank, historical prediction, name or label is imported.
Native material instances survive STEP transfer; free/helper wires are excluded.
"""
from __future__ import annotations
import math, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

RUNTIME=Path(__file__).resolve().parents[1]/'server_rebuild/cad_ocp_runtime_20261019_v1'
sys.path.insert(0,str(RUNTIME))
from OCP.STEPControl import STEPControl_Reader
from OCP.Interface import Interface_Static
from OCP.IFSelect import IFSelect_RetDone
from OCP.TopAbs import TopAbs_SOLID,TopAbs_EDGE,TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve,BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.BRepTools import BRepTools
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.GCPnts import GCPnts_QuasiUniformDeflection
from OCP.GeomAbs import GeomAbs_Line,GeomAbs_Plane,GeomAbs_Circle
from OCP.gp import gp_Pnt
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopLoc import TopLoc_Location
from scipy.spatial.transform import Rotation

NAMES=('log_length','log_radius','total_turn','circular','closed','plane_base_alignment',
       'tangent_base_alignment','height_fraction','height_above_base_log','station_fraction','end_distance_log',
       'curvature','other_material_distance_start','other_material_distance_mid','other_material_distance_end',
       'contact_observed','own_material_extent0','own_material_extent1','own_material_extent2',
       'nearest_raw_mm','raw_8th_mm','probe_n_plus2','probe_n_minus2','probe_b_plus2','probe_b_minus2',
       'probe_n_plus5','probe_n_minus5','probe_b_plus5','probe_b_minus5',
       'raw_before3','raw_after3','STEP_available','local_pose_observed','relative_curve_extent')

def vec(v): return np.array([v.X(),v.Y(),v.Z()],float)

def items(shape,kind):
    e=TopExp_Explorer(shape,kind)
    while e.More(): yield e.Current(); e.Next()

def bbox(shape):
    b=Bnd_Box();BRepBndLib.Add_s(shape,b,False);return np.array(b.Get())

def arclength(p): return np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]

def resample(p,step=1.):
    p=np.asarray(p,float);s=arclength(p);take=np.r_[True,np.diff(s)>1e-10];p=p[take];s=s[take]
    if len(p)<2:return p
    grid=np.linspace(0,s[-1],max(2,int(np.ceil(s[-1]/step))+1))
    return np.column_stack([np.interp(grid,s,p[:,i]) for i in range(3)])

def read_xyz(path):
    path=Path(path)
    if path.suffix=='.npy':return np.asarray(np.load(path),float)
    with path.open('rb') as f:
        first=f.readline().strip()
        if first!=b'ply':raise ValueError('Pure XYZ PLY is required')
        while True:
            line=f.readline()
            if not line:raise ValueError('Missing PLY end_header')
            if line.strip()==b'end_header':break
        xyz=np.loadtxt(f,usecols=(0,1,2),ndmin=2)
    if not np.isfinite(xyz).all():raise ValueError('Nonfinite original XYZ; source indices must be retained before cleaning')
    return xyz

def circle_fit(p):
    p=np.asarray(p,float)
    if len(p)<3:return None
    center=p.mean(0);_,_,v=np.linalg.svd(p-center,full_matrices=False)
    q=(p-center)@v[:2].T
    c=np.linalg.lstsq(np.c_[2*q,np.ones(len(q))],(q*q).sum(1),rcond=None)[0]
    r2=c[2]+c[:2]@c[:2]
    if r2<=0:return None
    cc=center+c[:2]@v[:2];r=float(np.sqrt(r2));n=v[2];u=(p[0]-cc)/r
    planar=np.abs((p-cc)@n);rad=np.abs(np.linalg.norm(p-cc,axis=1)-r)
    if max(planar.max(),rad.max())>.05:return None
    u-=n*(u@n);u/=np.linalg.norm(u);vv=np.cross(n,u)
    angles=np.unwrap(np.arctan2((p-cc)@vv,(p-cc)@u));sweep=float(angles[-1]-angles[0])
    if abs(sweep)<1e-7:return None
    return dict(center=cc.tolist(),radius=r,normal=n.tolist(),start=u.tolist(),sweep=sweep,
                max_fit_error_mm=float(max(planar.max(),rad.max())))

def curve_points(edge):
    ad=BRepAdaptor_Curve(edge)
    if ad.GetType()==GeomAbs_Line:return None
    a,b=ad.FirstParameter(),ad.LastParameter()
    if not np.isfinite([a,b]).all():return None
    if ad.GetType()==GeomAbs_Circle:
        r=ad.Circle().Radius();n=max(5,int(np.ceil(abs(b-a)*r))+1,int(np.ceil(abs(b-a)/.02))+1)
        return np.array([vec(ad.Value(float(t)))for t in np.linspace(a,b,n)])
    points=GCPnts_QuasiUniformDeflection(ad,.025,a,b)
    if not points.IsDone() or points.NbPoints()<2:return None
    p=np.array([vec(points.Value(i))for i in range(1,points.NbPoints()+1)])
    if len(p)<3:return None
    delta=p[-1]-p[0];length=np.linalg.norm(delta)
    if length>1e-8 and np.linalg.norm(np.cross(p-p[0],delta/length),axis=1).max()<1e-5:return None
    return resample(p,1.)

def load_material(path):
    Interface_Static.SetCVal_s('xstep.cascade.unit','MM');reader=STEPControl_Reader()
    if reader.ReadFile(str(path))!=IFSelect_RetDone or not reader.TransferRoots():raise ValueError('Material STEP transfer failed')
    solids=list(items(reader.OneShape(),TopAbs_SOLID))
    if not solids:raise ValueError('No original material solids')
    return solids

def finite_distance(shape,p):
    v=BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float,p))).Vertex();q=BRepExtrema_DistShapeShape(v,shape)
    return float(q.Value()) if q.IsDone() else 1e4

def pose_quality(solids,boxes,xyz,T):
    ids=np.linspace(0,len(xyz)-1,min(64,len(xyz)),dtype=int);q=(xyz[ids]-T[:3,3])@T[:3,:3];dist=[]
    for p in q:
        lo=np.linalg.norm(np.maximum(np.maximum(boxes[:,:3]-p,p-boxes[:,3:]),0),axis=1);best=1e4
        for i in np.argsort(lo):
            if lo[i]>best:break
            best=min(best,finite_distance(solids[int(i)],p))
        dist.append(best)
    return float(np.quantile(dist,.95)),dist

def choose_pose(solids,xyz,provided=None):
    boxes=np.array([bbox(s)for s in solids]);candidates=[np.eye(4)]
    if provided is not None:
        T=np.array(provided,float);candidates=[T,np.linalg.inv(T)]+candidates
    trials=[]
    for T in candidates:
        q,ds=pose_quality(solids,boxes,xyz,T);trials.append((q,T,ds))
        if q<.1:break
    q,T,ds=min(trials,key=lambda x:x[0])
    if q>3:
        # Input-only principal-axis hypotheses; labels never resolve symmetry.
        import itertools
        points=np.concatenate([np.array([bbox(s)[:3],bbox(s)[3:]])for s in solids])
        native_center=(boxes[:,:3].min(0)+boxes[:,3:].max(0))/2
        center=(xyz.min(0)+xyz.max(0))/2;extent=np.ptp(xyz,axis=0)
        proposals=[]
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product((-1,1),repeat=3):
                R=np.eye(3)[:,perm]@np.diag(signs)
                if np.linalg.det(R)<0:continue
                tt=np.eye(4);tt[:3,:3]=R;tt[:3,3]=center-native_center@R.T
                error=np.linalg.norm(np.ptp(points@R.T,axis=0)-extent)
                proposals.append((error,tt))
        for _,tt in sorted(proposals,key=lambda x:x[0])[:4]:
            qq,dds=pose_quality(solids,boxes,xyz,tt);trials.append((qq,tt,dds))
        q,T,ds=min(trials,key=lambda x:x[0])
    if q>3:
        mesh=[];normals=[]
        for solid in solids:
            BRepMesh_IncrementalMesh(solid,.3,False,.2,False)
            for fs in items(solid,TopAbs_FACE):
                face=TopoDS.Face_s(fs);loc=TopLoc_Location();tri=BRep_Tool.Triangulation_s(face,loc)
                if tri is None:continue
                nodes=np.array([vec(tri.Node(i).Transformed(loc.Transformation()))for i in range(1,tri.NbNodes()+1)])
                indices=np.array([tri.Triangle(i).Get()for i in range(1,tri.NbTriangles()+1)],int)-1
                vertices=nodes[indices];cross=np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]);area=np.linalg.norm(cross,axis=1);take=area>1e-10
                mesh.extend(vertices[take]);normals.extend(cross[take]/area[take,None])
        if mesh:
            mesh=np.array(mesh);normals=np.array(normals);areas=np.linalg.norm(np.cross(mesh[:,1]-mesh[:,0],mesh[:,2]-mesh[:,0]),axis=1);rng=np.random.default_rng(20261106)
            selected=rng.choice(len(mesh),12000,p=areas/areas.sum());uv=rng.random((len(selected),2));uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1]
            samples=mesh[selected,0]+uv[:,0,None]*(mesh[selected,1]-mesh[selected,0])+uv[:,1,None]*(mesh[selected,2]-mesh[selected,0]);ns=normals[selected]
            raw=xyz[np.sort(rng.choice(len(xyz),min(12000,len(xyz)),replace=False))];nc=samples.mean(0);rc=raw.mean(0)
            _,nv=np.linalg.eigh(np.cov(samples.T));_,rv=np.linalg.eigh(np.cov(raw.T));tree=cKDTree(samples);rawtree=cKDTree(raw);proposed=[]
            import itertools
            for perm in itertools.permutations(range(3)):
                for signs in itertools.product((-1,1),repeat=3):
                    R=rv[:,perm]@np.diag(signs)@nv.T
                    if np.linalg.det(R)<0:continue
                    tt=np.eye(4);tt[:3,:3]=R;tt[:3,3]=rc-nc@R.T
                    d=rawtree.query(samples[::12]@R.T+tt[:3,3],workers=1)[0];proposed.append((np.quantile(d,.8),tt))
            for _,initial in sorted(proposed,key=lambda v:v[0])[:3]:
                U=np.linalg.inv(initial);source=raw[::3]
                for iteration in range(24):
                    p=source@U[:3,:3].T+U[:3,3];dist,j=tree.query(p,workers=1);n=ns[j];take=dist<=max(np.quantile(dist,.8),2.)
                    A=np.c_[np.cross(p[take],n[take]),n[take]];rhs=((samples[j[take]]-p[take])*n[take]).sum(1)
                    solution=np.linalg.lstsq(A,rhs,rcond=None)[0];angle=np.linalg.norm(solution[:3])
                    if angle>.2:solution[:3]*=.2/angle
                    D=np.eye(4);D[:3,:3]=Rotation.from_rotvec(solution[:3]).as_matrix();D[:3,3]=solution[3:];U=D@U
                    if np.linalg.norm(solution)<1e-5:break
                tt=np.linalg.inv(U);qq,dds=pose_quality(solids,boxes,xyz,tt);trials.append((qq,tt,dds))
            q,T,ds=min(trials,key=lambda v:v[0])
    return T,dict(sample_p95_mm=q,passed=q<=3.,sample_distances_mm=ds,
                  input_only=True,globally_unique_pose_proved=False,trial_p95=[v[0]for v in trials])

def point_curves(xyz):
    """Bounded point-only fallback: circle hypotheses on dense planar strata."""
    rng=np.random.default_rng(20261106);out=[];extent=np.ptp(xyz,axis=0);scale=np.linalg.norm(extent)
    for axis in range(3):
        others=[j for j in range(3)if j!=axis];bins=np.round(xyz[:,axis]*2).astype(np.int64)
        ids,counts=np.unique(bins,return_counts=True)
        for level in ids[np.argsort(counts)[-6:]]:
            p=xyz[bins==level];p=p[rng.choice(len(p),min(len(p),2000),replace=False)]
            if len(p)<24:continue
            q=p[:,others]
            best=None
            for _ in range(96):
                a,b,c=q[rng.choice(len(q),3,replace=False)];M=2*np.array([b-a,c-a])
                if abs(np.linalg.det(M))<1e-6:continue
                cc=np.linalg.solve(M,np.array([b@b-a@a,c@c-a@a]));radius=np.linalg.norm(a-cc)
                if not 5<radius<2*scale:continue
                residual=abs(np.linalg.norm(q-cc,axis=1)-radius);take=residual<.25
                score=int(take.sum())
                if score>=24 and (best is None or score>best[0]):best=(score,cc,radius,take)
            if best is None:continue
            _,cc,radius,take=best;angles=np.sort(np.arctan2(q[take,1]-cc[1],q[take,0]-cc[0]))
            cuts=np.flatnonzero(np.diff(angles)*radius>12.)+1
            for part in np.split(angles,cuts):
                if len(part)<6 or (part[-1]-part[0])*radius<5.:continue
                aa=np.linspace(part[0],part[-1],max(5,int(np.ceil((part[-1]-part[0])*radius))+1))
                pp=np.zeros((len(aa),3));pp[:,axis]=np.median(p[:,axis]);pp[:,others[0]]=cc[0]+radius*np.cos(aa);pp[:,others[1]]=cc[1]+radius*np.sin(aa)
                out.append(dict(points=pp.tolist(),material=-1,other_distance=[1e4]*3,STEP=False,pose=False))
    return out

def input_curves(input_record):
    start=time.perf_counter();xyz=read_xyz(input_record['raw']['path']);read_seconds=time.perf_counter()-start
    candidates=[];timing=dict(read_xyz=read_seconds);audit={};base=np.array([0.,0.,1.])
    step=input_record.get('STEP')
    if step:
        begin=time.perf_counter();solids=load_material(step['path']);boxes=np.array([bbox(s)for s in solids]);faces=[list(items(s,TopAbs_FACE))for s in solids];faceboxes=[np.array([bbox(f)for f in fs])for fs in faces]
        T,audit=choose_pose(solids,xyz,input_record.get('transform'));timing['STEP_and_pose']=time.perf_counter()-begin
        areas=[]
        for material in faces:
            for face in material:
                face=TopoDS.Face_s(face);ad=BRepAdaptor_Surface(face,True)
                if ad.GetType()!=GeomAbs_Plane:continue
                props=GProp_GProps();BRepGProp.SurfaceProperties_s(face,props)
                areas.append((abs(props.Mass()),vec(ad.Plane().Axis().Direction())@T[:3,:3].T))
        if areas:base=max(areas,key=lambda x:x[0])[1]
        if base@np.array([0.,0.,1.])<0:base=-base
        begin=time.perf_counter();seen={}
        for mi,solid in enumerate(solids):
            for rawedge in items(solid,TopAbs_EDGE):
                edge=TopoDS.Edge_s(rawedge);p=curve_points(edge)
                if p is None or len(p)<3 or arclength(p)[-1]<1e-4:continue
                world=p@T[:3,:3].T+T[:3,3]
                # Symmetric signatures preserve distinct offset curves/materials.
                samples=resample(world,max(arclength(world)[-1]/16.,.1))
                a=np.round(samples,3).tobytes();b=np.round(samples[::-1],3).tobytes();signature=min(a,b)
                if signature in seen:continue
                seen[signature]=len(candidates);other=[]
                for point in p[[0,len(p)//2,-1]]:
                    ds=1e4
                    for mj,fs in enumerate(faces):
                        if mj==mi:continue
                        box=boxes[mj];lower=np.linalg.norm(np.maximum(np.maximum(box[:3]-point,point-box[3:]),0))
                        if lower>min(ds,20.):continue
                        fb=faceboxes[mj];low=np.linalg.norm(np.maximum(np.maximum(fb[:,:3]-point,point-fb[:,3:]),0),axis=1)
                        for fi in np.argsort(low):
                            if low[fi]>min(ds,20.):break
                            ds=min(ds,finite_distance(fs[int(fi)],point))
                    other.append(ds)
                candidates.append(dict(points=world.tolist(),material=mi,other_distance=other,STEP=True,pose=audit['passed'],material_extent=np.sort(boxes[mi,3:]-boxes[mi,:3]).tolist()))
        timing['curve_candidates_and_contact']=time.perf_counter()-begin
    else:
        begin=time.perf_counter();candidates=point_curves(xyz);timing['point_curve_candidates']=time.perf_counter()-begin
    begin=time.perf_counter();tree=cKDTree(xyz,balanced_tree=False);rows=[];owners=[];offsets=[0];bounds=np.array([xyz.min(0),xyz.max(0)]);scale=max(np.linalg.norm(bounds[1]-bounds[0]),1.)
    heights=xyz@base;hmin,hmax=heights.min(),heights.max()
    for ci,c in enumerate(candidates):
        p=resample(c['points'],2.)
        if len(p)==2:p=resample(c['points'],arclength(np.array(c['points']))[-1]/2.01)
        s=arclength(p);length=s[-1];circle=circle_fit(p);c['circle']=circle;c['stations']=p.tolist();c['station_s_mm']=s.tolist()
        tangent=np.gradient(p,axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-12)
        if circle:
            n=np.broadcast_to(np.array(circle['normal']),p.shape);radius=circle['radius'];turn=abs(circle['sweep'])
        else:
            centered=p-p.mean(0);_,_,vh=np.linalg.svd(centered,full_matrices=False);n=np.broadcast_to(vh[-1],p.shape);radius=0.;turn=float(np.arccos(np.clip((tangent[1:]*tangent[:-1]).sum(1),-1,1)).sum())
        b=np.cross(tangent,n);b/=np.maximum(np.linalg.norm(b,axis=1)[:,None],1e-12)
        dist,idx=tree.query(p,k=8,workers=1,distance_upper_bound=30.);probes=[]
        for delta in (2.,5.):
            for direction in (n,-n,b,-b):probes.append(np.minimum(tree.query(p+delta*direction,workers=1,distance_upper_bound=30.)[0],30.))
        nearbefore=tree.query(p-3*tangent,workers=1,distance_upper_bound=30.)[0];nearafter=tree.query(p+3*tangent,workers=1,distance_upper_bound=30.)[0]
        curvature=np.linalg.norm(np.gradient(tangent,axis=0),axis=1)/np.maximum(np.gradient(s),1e-8)
        he=p@base;other=np.minimum(c['other_distance'],30.);extent=np.log1p(c.get('material_extent',[0,0,0]))
        x=np.column_stack([np.full(len(p),np.log1p(length)),np.full(len(p),np.log1p(radius)),np.full(len(p),turn),np.full(len(p),circle is not None),np.full(len(p),np.linalg.norm(p[0]-p[-1])<.1),
            abs(n@base),abs(tangent@base),(he-hmin)/max(hmax-hmin,1.),np.log1p(np.maximum(he-hmin,0)),s/max(length,1e-9),np.log1p(np.minimum(s,length-s)),curvature,
            *[np.full(len(p),v)for v in other],np.full(len(p),min(other)<3),*[np.full(len(p),v)for v in extent],np.minimum(dist[:,0],30.),np.minimum(dist[:,-1],30.),*probes,np.minimum(nearbefore,30.),np.minimum(nearafter,30.),
            np.full(len(p),c['STEP']),np.full(len(p),c['pose']),np.full(len(p),length/scale)])
        assert x.shape[1]==len(NAMES) and np.isfinite(x).all(),x.shape
        rows.append(x.astype(np.float32));owners.extend([ci]*len(p));offsets.append(offsets[-1]+len(p))
    features=np.concatenate(rows) if rows else np.empty((0,len(NAMES)),np.float32)
    timing['KD_tree_and_features']=time.perf_counter()-begin;timing['input_to_features']=time.perf_counter()-start
    return features,dict(candidates=candidates,offsets=offsets,base_axis=base.tolist(),input_points=len(xyz),timing=timing,pose=audit,source='original_material_curves_or_bounded_XYZ_circle_fallback',labels_read=False)

def slice_curve(p,a,b):
    p=np.asarray(p,float);s=arclength(p);inside=(s>a)&(s<b)
    point=lambda t:np.array([np.interp(t,s,p[:,i])for i in range(3)])
    return np.vstack([point(a),p[inside],point(b)])

def decode(features,packet,model,threshold=.5):
    prob=model.predict_proba(features) if len(features) else np.empty((0,3));classes=list(model.classes_);result=[]
    for i,c in enumerate(packet['candidates']):
        a,b=packet['offsets'][i:i+2];p=np.array(c['stations']);s=np.array(c['station_s_mm']);pp=prob[a:b]
        if not len(pp):continue
        labels=np.argmax(pp,axis=1);labels=np.array([int(classes[v])for v in labels]);labels[np.max(pp,axis=1)<threshold]=0
        start=0
        while start<len(labels):
            if labels[start]==0:start+=1;continue
            end=start+1
            while end<len(labels) and labels[end]==labels[start]:end+=1
            lo=0. if start==0 else (s[start-1]+s[start])/2;hi=s[-1] if end==len(labels) else (s[end-1]+s[end])/2
            if hi-lo<1e-6:start=end;continue
            points=slice_curve(c['points'],lo,hi);circle=circle_fit(points)
            result.append(dict(points=points.tolist(),weld_type='horizontal' if labels[start]==1 else 'vertical',candidate=i,
                confidence=float(pp[start:end,classes.index(int(labels[start]))].mean()),circle=circle,material=c['material']))
            start=end
    # Only truly equivalent complete geometry is suppressed; no close parallel
    # seams, positive gaps, different weld types or offset faces are merged.
    kept=[]
    for r in sorted(result,key=lambda r:-r['confidence']):
        p=resample(r['points'],1.);duplicate=False
        for k in kept:
            q=np.asarray(k['points'])
            if k['weld_type']!=r['weld_type'] or np.linalg.norm(p.mean(0)-q.mean(0))>5.:continue
            if max(cKDTree(p).query(q)[0].max(),cKDTree(q).query(p)[0].max())<.5:duplicate=True;break
        if not duplicate:kept.append(r)
    parents=[]
    for i,r in enumerate(kept):
        p=np.asarray(r['points']);length=arclength(p)[-1];circle=r['circle'];n=1 if not circle else max(1,int(math.ceil(abs(circle['sweep'])/(math.pi/6)-1e-10)))
        segments=[]
        for j in range(n):
            q=slice_curve(p,length*j/n,length*(j+1)/n)
            segments.append(dict(segment_index=j,points=q.tolist(),length_mm=float(arclength(q)[-1]),geometry_type='arc'if circle else 'bspline_polyline',reason='ARC_TURN_AT_MOST_30_DEGREES'if n>1 else 'NATIVE_CURVE_INTERVAL',status='NOMINAL_GEOMETRY_OBSERVATION_RECORDED'))
        parents.append(dict(parent_id=f'curve_{i:05d}',weld_type=r['weld_type'],weld_side=None,is_closed=bool(np.linalg.norm(p[0]-p[-1])<.1),segments=segments,points=p.tolist(),confidence=r['confidence'],source_candidate=r['candidate'],circle=circle,actual_geometry_status='NOT_INDEPENDENTLY_ACCEPTED'))
    return dict(parents=parents,geometry_units=len(parents),published_segments=sum(len(p['segments'])for p in parents),units='mm',feature_columns=len(NAMES),same_model_and_configuration=True)
