"""Offline TRAIN CAD registration from raw XYZ only; no weld targets or DEV.

A dominant observed plane and CAD volume inertia provide eight rigid hypotheses.
Robust point-to-plane ICP uses sampled CAD triangle normals, without a gravity
assumption. Reference registration is not an inference-time dependency.
"""
import argparse,ctypes,itertools,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
ctypes.CDLL(str(ROOT/'server_rebuild/cad_runtime_libs/extracted/usr/lib/x86_64-linux-gnu/libGLU.so.1'),mode=ctypes.RTLD_GLOBAL)
import gmsh
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=ROOT/'server_rebuild/step_contact_scope_audit_v1';DATA=ROOT.parent/'data/G_Data'
def dominant_basis(x,rng):
 sample=x[rng.choice(len(x),min(len(x),20000),replace=False)];best=None
 for _ in range(192):
  p=sample[rng.choice(len(sample),3,replace=False)];n=np.cross(p[1]-p[0],p[2]-p[0]);norm=np.linalg.norm(n)
  if norm<1e-8:continue
  n/=norm;mask=abs((sample-p[0])@n)<.2;count=int(mask.sum())
  if best is None or count>best[0]:best=(count,mask)
 plane=sample[best[1]];e,u=np.linalg.eigh(np.cov(plane.T));normal=u[:,0];long=u[:,-1];long-=normal*(long@normal);long/=np.linalg.norm(long);basis=np.column_stack([long,np.cross(normal,long),normal]);return basis,dict(plane_sample_inliers=len(plane),plane_sample_total=len(sample),eigenvalues=e.tolist(),no_fixed_gravity_axis=True)
def run(key):
 start=time.perf_counter();dest=BASE/('source_'+key)/'registration';dest.mkdir(parents=True,exist_ok=False);inp=ROOT/'server_rebuild/inputs_v1'/key;receipt=read(inp/'receipt.json');folder=(DATA/receipt['source_file']).parent;step=next(p for p in folder.iterdir() if p.suffix.lower() in ('.stp','.step'));assert digest(inp/'xyz_mm.npy')==receipt['xyz_sha256'];rng=np.random.default_rng(1031)
 protocol=dict(source=key,step_sha256=digest(step),input_sha256=receipt['xyz_sha256'],source_sha256=digest(__file__),gmsh_version=gmsh.__version__,seed=1031,mesh_min_mm=2.,mesh_max_mm=30.,CAD_surface_samples=70000,XYZ_fit_samples=60000,coarse_candidates=8,refined_candidates=4,max_ICP_iterations=80,ICP_distance_cutoff_mm=45.,Huber_scale_mm=1.,GT_weld_reads=0,DEV_reads=0,STEP_offline_only=True,full_frame_validation_pending=True)
 write(dest/'protocol.json',protocol);gmsh.initialize([],readConfigFiles=False);gmsh.option.setNumber('General.Terminal',0);gmsh.option.setString('Geometry.OCCTargetUnit','MM')
 try:
  gmsh.model.occ.importShapes(str(step));gmsh.model.occ.synchronize();volumes=gmsh.model.getEntities(3);largest=max(volumes,key=lambda v:gmsh.model.occ.getMass(*v));_,basis=np.linalg.eigh(np.asarray(gmsh.model.occ.getMatrixOfInertia(*largest)).reshape(3,3))
  if np.linalg.det(basis)<0:basis[:,0]*=-1
  gmsh.option.setNumber('Mesh.MeshSizeMin',2.);gmsh.option.setNumber('Mesh.MeshSizeMax',30.);gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',64);gmsh.option.setNumber('Mesh.MinimumCirclePoints',64);gmsh.option.setNumber('Mesh.Algorithm',6);gmsh.model.mesh.generate(2);node_ids,coords,_=gmsh.model.mesh.getNodes();vertices=np.asarray(coords).reshape(-1,3);order=np.argsort(node_ids);keys=np.asarray(node_ids)[order];triangles=[];owners=[]
  for _,face in gmsh.model.getEntities(2):
   types,_,blocks=gmsh.model.mesh.getElements(2,face)
   for kind,block in zip(types,blocks):
    _,dim,degree,count,_,primary=gmsh.model.mesh.getElementProperties(kind);assert dim==2 and primary==3 and degree==1
    tri=order[np.searchsorted(keys,np.asarray(block).reshape(-1,count)[:,:3])];triangles.extend(tri.tolist());owners.extend([face]*len(tri))
  triangles=np.asarray(triangles,np.int32);np.savez(dest/'cad_mesh.npz',vertices=vertices,triangles=triangles,face_ids=np.asarray(owners,np.int32),inertia_basis=basis)
 finally:gmsh.finalize()
 cadcenter=(vertices.min(0)+vertices.max(0))/2;v=vertices-cadcenter;tv=v[triangles];cross=np.cross(tv[:,1]-tv[:,0],tv[:,2]-tv[:,0]);area2=np.linalg.norm(cross,axis=1);usable=np.flatnonzero(area2>1e-10);draw=rng.choice(usable,70000,p=area2[usable]/area2[usable].sum());u=np.sqrt(rng.random(len(draw)));w=rng.random(len(draw));sample=tv[draw,0]*(1-u[:,None])+tv[draw,1]*(u*(1-w))[:,None]+tv[draw,2]*(u*w)[:,None];normals=cross[draw]/area2[draw,None];tree=cKDTree(sample)
 xyz=np.load(inp/'xyz_mm.npy',mmap_mode='r');nativecenter=(xyz.min(0)+xyz.max(0))/2;indices=np.sort(rng.choice(len(xyz),min(len(xyz),60000),replace=False));x=np.asarray(xyz[indices])-nativecenter;targetbasis,plane=dominant_basis(x,rng);np.savez(dest/'fit_samples.npz',CAD_points=sample,CAD_normals=normals,XYZ_indices=indices,CAD_center=cadcenter,native_center=nativecenter);write(dest/'dominant_plane.json',plane);candidates=[]
 for swap in [False,True]:
  permute=np.eye(3)[:,[1,0,2]] if swap else np.eye(3)
  for signs in itertools.product([-1.,1.],repeat=3):
   mid=permute@np.diag(signs)
   if np.linalg.det(mid)<0:continue
   R=targetbasis@mid@basis.T;rotated=v@R.T;t=(x.min(0)+x.max(0)-rotated.min(0)-rotated.max(0))/2;A=R.T;B=-A@t;dist=tree.query(x[::4]@A.T+B,workers=2)[0];candidates.append(dict(score=float(np.minimum(dist,100).mean()),A=A,B=B))
 candidates.sort(key=lambda c:c['score']);results=[];print('CAD_MESH_READY',key,len(vertices),len(triangles),'INITIAL',[round(c['score'],3) for c in candidates],flush=True)
 for ci,initial in enumerate(candidates[:4]):
  A=initial['A'].copy();B=initial['B'].copy();history=[]
  for iteration in range(80):
   moved=x@A.T+B;distance,nearest=tree.query(moved,workers=2);valid=distance<45.;assert valid.sum()>1000,'Insufficient ICP correspondences';p=moved[valid];q=sample[nearest[valid]];n=normals[nearest[valid]];residual=((p-q)*n).sum(1);j=np.column_stack([np.cross(p,n),n]);weight=np.minimum(1.,1./np.maximum(abs(residual),1e-8));root=np.sqrt(weight);delta=np.linalg.lstsq(j*root[:,None],-residual*root,rcond=None)[0];angle=np.linalg.norm(delta[:3]);delta[:3]*=min(1.,.05/max(angle,1e-15));shift=np.linalg.norm(delta[3:]);delta[3:]*=min(1.,10./max(shift,1e-15));Q=Rotation.from_rotvec(delta[:3]).as_matrix();A=Q@A;B=Q@B+delta[3:];history.append(dict(iteration=iteration,valid=int(valid.sum()),point_to_plane_median_mm=float(np.median(abs(residual))),point_to_plane_p95_mm=float(np.percentile(abs(residual),95)),delta=delta.tolist()))
   if iteration%20==0:print('ICP',key,ci,iteration,history[-1]['point_to_plane_p95_mm'],flush=True)
   if np.linalg.norm(delta[:3])<1e-10 and np.linalg.norm(delta[3:])<1e-6:break
  moved=x@A.T+B;distance,nearest=tree.query(moved,workers=2);residual=abs(((moved-sample[nearest])*normals[nearest]).sum(1));R=A.T;t=nativecenter-R@(cadcenter+B);transform=np.eye(4);transform[:3,:3]=R;transform[:3,3]=t;result=dict(initial_score=initial['score'],transform_STEP_to_native=transform.tolist(),sampled_CAD_NN_mean_mm=float(distance.mean()),sampled_CAD_NN_p95_mm=float(np.percentile(distance,95)),nearest_plane_residual_median_mm=float(np.median(residual)),nearest_plane_residual_p95_mm=float(np.percentile(residual,95)),nearest_plane_within_point2_rate=float((residual<=.2).mean()),history=history,exact_finite_STEP_validation_pending=True);results.append(result);print('ICP_RESULT',key,ci,{k:v for k,v in result.items() if k not in ['history','transform_STEP_to_native']},flush=True)
 xyz._mmap.close();results.sort(key=lambda r:(r['nearest_plane_residual_p95_mm'],r['sampled_CAD_NN_mean_mm']));write(dest/'registration_candidates.json',dict(source=key,candidates=results,ranking='Raw XYZ nearest CAD-plane p95 then sampled-CAD NN mean; no weld GT',not_accepted_without_exact_finite_validation=True,mesh_sha256=digest(dest/'cad_mesh.npz'),samples_sha256=digest(dest/'fit_samples.npz'),protocol_sha256=digest(dest/'protocol.json'),seconds=time.perf_counter()-start));print('REGISTRATION_CANDIDATES_READY',key,round(time.perf_counter()-start,2),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--key',required=True,choices=['10','25','36']);args=ap.parse_args()
 with threadpool_limits(limits=2):run(args.key)
