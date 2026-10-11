"""Offline continuous finite CAD line support on sampled side-specific faces.

Project the reference line onto the two CAD planes, record every displacement
and every missing interval. CAD support is not annotation completeness.
"""
import argparse,ctypes,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ctypes.CDLL(str(ROOT/'server_rebuild/cad_runtime_libs/extracted/usr/lib/x86_64-linux-gnu/libGLU.so.1'),mode=ctypes.RTLD_GLOBAL)
import gmsh
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.continuous_step_support_273 import union_and_missing
BASE=ROOT/'server_rebuild/step_contact_scope_audit_v1';DATA=ROOT.parent/'data/G_Data'
def run(key):
 start=time.perf_counter();folder=BASE/('source_'+key);pairs=read(folder/'sampled_contact_face_pairs.json');atlas=read(folder/'step_planar_faces_world.json');faces={f['face']:f for f in atlas['faces']};posepath=folder/'registration/finite_validation_fixed_uv/validation.json'
 if not posepath.exists():posepath=folder/'registration/finite_refinement/validation.json'
 pose=read(posepath);T=np.asarray(pose['transform_STEP_to_native']);R=T[:3,:3];t=T[:3,3];targetpath=ROOT/'server_rebuild/labels_v1'/key/'targets.json';targets=native_targets(read(targetpath));inp=read(ROOT/'server_rebuild/inputs_v1'/key/'receipt.json');rawfolder=(DATA/inp['source_file']).parent;step=next(p for p in rawfolder.iterdir() if p.suffix.lower() in ('.stp','.step'));assert digest(step)==atlas['step_sha256'];gmsh.initialize([],readConfigFiles=False);gmsh.option.setNumber('General.Terminal',0);gmsh.option.setString('Geometry.OCCTargetUnit','MM');records=[]
 try:
  gmsh.model.occ.importShapes(str(step));gmsh.model.occ.synchronize();original=set(gmsh.model.occ.getEntities());masses={v:gmsh.model.occ.getMass(3,v) for _,v in gmsh.model.getEntities(3)}
  for row in pairs['rows']:
   base=dict(parent=row['parent'],child=row['child'],continuous_annotation_completeness=False)
   if not row['unique_common_sampled_face_pair']:records.append(dict(base,status='AMBIGUOUS_FACE_PAIR'));continue
   tags=row['common_face_pairs'][0]
   if any(tag not in faces for tag in tags):records.append(dict(base,status='NONPLANAR_REFERENCE'));continue
   ref=reference_segment(targets['parents'][row['parent']]['children'][row['child']]);ends=(np.asarray(ref['points'])[[0,-1]]-t)@R;n=np.array([faces[tag]['normal'] for tag in tags]);centers=np.array([faces[tag]['center'] for tag in tags]);offset=(n*centers).sum(1);projected=ends+(offset[None,:]-ends@n.T)@np.linalg.pinv(n).T;displacement=np.linalg.norm(projected-ends,axis=1);delta=projected[1]-projected[0];length=np.linalg.norm(delta);support=[]
   for tag in tags:
    try:
     a,b=[gmsh.model.occ.addPoint(*p.tolist()) for p in projected];line=gmsh.model.occ.addLine(a,b);copied=gmsh.model.occ.copy([(2,tag)]);common,_=gmsh.model.occ.intersect([(1,line)],copied);gmsh.model.occ.synchronize();intervals=[];witness=[]
     for dim,curve in common:
      if dim!=1:continue
      lo,hi=gmsh.model.getParametrizationBounds(1,curve);e=np.asarray(gmsh.model.getValue(1,curve,[float(lo[0]),float(hi[0])])).reshape(2,3);mass=gmsh.model.occ.getMass(1,curve);axiserror=np.linalg.norm(np.cross(e-projected[0],delta/length),axis=1).max();assert axiserror<1e-4 and abs(mass-np.linalg.norm(e[1]-e[0]))<1e-4
      interval=sorted(((e-projected[0])@delta/(length*length)).tolist());intervals.append(interval);witness.append(dict(endpoints_CAD_mm=e.tolist(),length_mm=mass,interval=interval))
     covered,missing=union_and_missing(intervals);support.append(dict(face=tag,covered=covered,missing=missing,missing_total_mm=sum(hi-lo for lo,hi in missing)*length,strict_complete=not missing,witnesses=witness))
    finally:
     created=set(gmsh.model.occ.getEntities())-original
     for dim in [3,2,1,0]:
      batch=[(d,tag) for d,tag in created if d==dim and (d,tag) in set(gmsh.model.occ.getEntities())]
      if batch:gmsh.model.occ.remove(batch,recursive=False)
     gmsh.model.occ.synchronize()
   assert set(gmsh.model.occ.getEntities())==original
   records.append(dict(base,status='AUDITED',face_pair=tags,projection_displacement_mm=displacement.tolist(),projection_within_registration_tolerance=bool(max(displacement)<=.2),projected_endpoints_CAD_mm=projected.tolist(),supports=support,strict_continuous_CAD_support=bool(max(displacement)<=.2 and all(s['strict_complete'] for s in support))))
   if len(records)%16==0:print(key,len(records),flush=True)
  assert all(abs(gmsh.model.occ.getMass(3,v)-mass)<max(1e-4,mass*1e-10) for v,mass in masses.items())
 finally:gmsh.finalize()
 report=dict(source=key,records=records,total=len(records),strict_supported=sum(r.get('strict_continuous_CAD_support',False) for r in records),negative_labels_released=False,annotation_closure_proved=False,GT_used_only_after_XYZ_registration=True,STEP_offline_only=True,pose_sha256=digest(posepath),targets_sha256=digest(targetpath),step_sha256=digest(step),source_sha256=digest(__file__),seconds=time.perf_counter()-start)
 write(folder/'continuous_contact_scope.json',report);print({k:v for k,v in report.items() if k!='records'},flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--key',required=True,choices=['10','25','36']);run(ap.parse_args().key)
