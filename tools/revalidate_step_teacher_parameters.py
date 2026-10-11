"""Offline raw-XYZ planar refinement and held-out exact STEP trim validation."""
import argparse,ctypes,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
ctypes.CDLL(str(ROOT/'server_rebuild/cad_runtime_libs/extracted/usr/lib/x86_64-linux-gnu/libGLU.so.1'),mode=ctypes.RTLD_GLOBAL)
import gmsh
from weld_pipeline.server_offline_step_queries import closest_with_verified_parameters
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=ROOT/'server_rebuild/step_contact_scope_audit_v1';DATA=ROOT.parent/'data/G_Data'
def run(key,validation_only=False):
 start=time.perf_counter();folder=BASE/('source_'+key);reg=folder/'registration';dest=reg/('finite_validation_teacher_uv' if validation_only else 'finite_refinement');dest.mkdir(exist_ok=False);atlas=read(folder/'step_planar_faces_world.json');prior=read(reg/'registration_candidates.json');inputfolder=ROOT/'server_rebuild/inputs_v1'/key;ir=read(inputfolder/'receipt.json');rawfolder=(DATA/ir['source_file']).parent;step=next(p for p in rawfolder.iterdir() if p.suffix.lower() in ('.stp','.step'));assert digest(step)==atlas['step_sha256'];assert digest(inputfolder/'xyz_mm.npy')==ir['xyz_sha256']
 with np.load(reg/'fit_samples.npz') as z:fit_ids=z['XYZ_indices'];nativecenter=z['native_center'];cadcenter=z['CAD_center']
 xyz=np.load(inputfolder/'xyz_mm.npy',mmap_mode='r');x=np.asarray(xyz[fit_ids])-nativecenter;T=np.asarray(read(reg/'finite_refinement/validation.json')['transform_STEP_to_native'] if validation_only else prior['candidates'][0]['transform_STEP_to_native']);A=T[:3,:3].T;B=A@(nativecenter-T[:3,3])-cadcenter;faces=atlas['faces'];normal=np.array([f['normal'] for f in faces]);centers=np.array([f['center'] for f in faces])-cadcenter;boxes=np.array([f['bbox'] for f in faces]);boxes[:,:3]-=cadcenter;boxes[:,3:]-=cadcenter;offset=(centers*normal).sum(1);history=[]
 write(dest/'protocol.json',dict(source=key,raw_XYZ_only_fit=True,GT_weld_reads=0,DEV_reads=0,STEP_offline_only=True,initial_registration_sha256=digest(reg/'registration_candidates.json'),atlas_sha256=digest(folder/'step_planar_faces_world.json'),source_sha256=digest(__file__),plane_inlier_mm=.2,Huber_mm=.05,bbox_margin_mm=.2,maximum_iterations=30,validation_sample_count=2048,validation_disjoint_from_fit=True,no_local_annotation_closure_claim=True,pose_refitted=not validation_only,query_adapter_sha256=digest(ROOT/'weld_pipeline/server_offline_step_queries.py'),frozen_pose_sha256=digest(reg/'finite_refinement/validation.json') if validation_only else None))
 for iteration in range(0 if validation_only else 30):
  moved=x@A.T+B;chosen=[];valid=[]
  for begin in range(0,len(moved),2048):
   p=moved[begin:begin+2048];distance=abs(p@normal.T-offset);inside=np.all(p[:,None,:]>=boxes[None,:,:3]-.2,axis=2)&np.all(p[:,None,:]<=boxes[None,:,3:]+.2,axis=2);distance[~inside]=np.inf;idx=np.argmin(distance,axis=1);keep=distance[np.arange(len(p)),idx]<.2;valid.extend((np.flatnonzero(keep)+begin).tolist());chosen.extend(idx[keep].tolist())
  assert len(valid)>1000;p=moved[valid];n=normal[chosen];residual=(p*n).sum(1)-offset[chosen];j=np.column_stack([np.cross(p,n),n]);weight=np.minimum(1.,.05/np.maximum(abs(residual),1e-10));sq=np.sqrt(weight);delta=np.linalg.lstsq(j*sq[:,None],-residual*sq,rcond=None)[0];Q=Rotation.from_rotvec(delta[:3]).as_matrix();A=Q@A;B=Q@B+delta[3:];entry=dict(iteration=iteration,inliers=len(valid),median_mm=float(np.median(abs(residual))),p95_mm=float(np.percentile(abs(residual),95)),delta=delta.tolist());history.append(entry)
  if iteration%5==0:print('PLANE_REFINE',key,entry,flush=True)
  if np.linalg.norm(delta[:3])<1e-12 and np.linalg.norm(delta[3:])<1e-8:break
 R=A.T;t=nativecenter-R@(cadcenter+B);transform=np.eye(4);transform[:3,:3]=R;transform[:3,3]=t;rng=np.random.default_rng(2031);fitset=set(map(int,fit_ids));sample_ids=[]
 while len(sample_ids)<min(2048,len(xyz)-len(fit_ids)):
  i=int(rng.integers(len(xyz)))
  if i not in fitset and i not in sample_ids:sample_ids.append(i)
 points=(np.asarray(xyz[sample_ids])-nativecenter)@A.T+B+cadcenter;gmsh.initialize([],readConfigFiles=False);gmsh.option.setNumber('General.Terminal',0);gmsh.option.setString('Geometry.OCCTargetUnit','MM');records=[]
 try:
  gmsh.model.occ.importShapes(str(step));gmsh.model.occ.synchronize();tags=[tag for _,tag in gmsh.model.getEntities(2)];bounds=np.array([gmsh.model.getBoundingBox(2,tag) for tag in tags]);center={tag:np.asarray(gmsh.model.occ.getCenterOfMass(2,tag)) for tag in tags}
  for ii,(index,point) in enumerate(zip(sample_ids,points)):
   candidates=np.flatnonzero(np.all(point>=bounds[:,:3]-.2,axis=1)&np.all(point<=bounds[:,3:]+.2,axis=1));matches=[];unverified=[]
   for j in candidates:
    tag=tags[j];rawq,_=gmsh.model.getClosestPoint(2,tag,point.tolist());distance=float(np.linalg.norm(np.asarray(rawq)-point))
    if distance>.2:continue
    try:q,uv,param_error=closest_with_verified_parameters(tag,point)
    except ValueError as exc:unverified.append(dict(face=tag,reason=str(exc)));continue
    exact=bool(gmsh.model.isInside(2,tag,list(uv),parametric=True));nudged=False
    if not exact:
     n=np.asarray(gmsh.model.getNormal(tag,list(uv)));n/=np.linalg.norm(n);delta=center[tag]-q;delta-=n*(delta@n);length=np.linalg.norm(delta)
     if length>1e-8:
      try:
       qq,u2,_=closest_with_verified_parameters(tag,q+.1*delta/length);nudged=bool(gmsh.model.isInside(2,tag,list(u2),parametric=True))
      except ValueError as exc:unverified.append(dict(face=tag,reason=str(exc),boundary_probe=True))
    if exact or nudged:matches.append(dict(face=tag,distance_mm=distance,exact_trim=exact,boundary_nudge=not exact))
   records.append(dict(point_index=index,matches=matches,unverified_parameter_queries=unverified))
   if ii%512==0:print('FINITE_VALIDATION',key,ii+1,flush=True)
 finally:gmsh.finalize();xyz._mmap.close()
 exact=sum(any(m['exact_trim'] for m in r['matches']) for r in records);supported=sum(bool(r['matches']) for r in records);errors=[min(m['distance_mm'] for m in r['matches']) for r in records if r['matches']];report=dict(source=key,transform_STEP_to_native=transform.tolist(),history=history,validation=records,validation_samples=len(records),exact_trim_supported=exact,with_boundary_nudge_supported=supported,with_boundary_nudge_rate=supported/len(records),supported_distance_max_mm=max(errors,default=None),supported_distance_p95_mm=float(np.percentile(errors,95)) if errors else None,unsupported_indices=[r['point_index'] for r in records if not r['matches']],sampled_registration_supported=supported==len(records),full_input_validation_not_performed=True,reference_material_identity_not_customer_truth=True,local_annotation_closure_proved=False,protocol_sha256=digest(dest/'protocol.json'),seconds=time.perf_counter()-start)
 write(dest/'validation.json',report);print({k:v for k,v in report.items() if k not in ['transform_STEP_to_native','history','validation','unsupported_indices']},flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--key',required=True,choices=['10','25','36']);ap.add_argument('--validation-only',action='store_true');args=ap.parse_args()
 with threadpool_limits(limits=2):run(args.key,args.validation_only)
