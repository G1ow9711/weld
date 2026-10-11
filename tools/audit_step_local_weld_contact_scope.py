"""Offline TRAIN sampled CAD contact identity and endpoint continuation evidence.

This does not certify continuous ownership, process termination, or annotation
completeness, and never supplies a deployment blacklist.
"""
import argparse,ctypes,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ctypes.CDLL(str(ROOT/'server_rebuild/cad_runtime_libs/extracted/usr/lib/x86_64-linux-gnu/libGLU.so.1'),mode=ctypes.RTLD_GLOBAL)
import gmsh
from weld_pipeline.server_offline_step_queries import closest_with_verified_parameters
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.world_set_evaluation_285 import reference_segment
BASE=ROOT/'server_rebuild/step_contact_scope_audit_v1';DATA=ROOT.parent/'data/G_Data'
def run(key):
 f=BASE/('source_'+key);posepath=f/'registration/finite_validation_fixed_uv/validation.json';posepath=posepath if posepath.exists() else f/'registration/finite_refinement/validation.json';pose=read(posepath);T=np.asarray(pose['transform_STEP_to_native']);R=T[:3,:3];t=T[:3,3];inp=read(ROOT/'server_rebuild/inputs_v1'/key/'receipt.json');rawfolder=(DATA/inp['source_file']).parent;step=next(p for p in rawfolder.iterdir() if p.suffix.lower() in ('.stp','.step'));targetpath=ROOT/'server_rebuild/labels_v1'/key/'targets.json';tr=read(targetpath.parent/'receipt.json');assert tr['role']=='TRAIN' and tr['targets_sha256']==digest(targetpath);target=native_targets(read(targetpath));gmsh.initialize([],readConfigFiles=False);gmsh.option.setNumber('General.Terminal',0);gmsh.option.setString('Geometry.OCCTargetUnit','MM');records=[]
 try:
  gmsh.model.occ.importShapes(str(step));gmsh.model.occ.synchronize();owners={}
  for _,v in gmsh.model.getEntities(3):
   for dim,face in gmsh.model.getBoundary([(3,v)],combined=False,oriented=False):
    if dim==2:owners.setdefault(face,set()).add(v)
  tags=[tag for _,tag in gmsh.model.getEntities(2)];bounds=np.array([gmsh.model.getBoundingBox(2,tag) for tag in tags])
  def contacts(native):
   point=(np.asarray(native)-t)@R;candidates=np.flatnonzero(np.all(point>=bounds[:,:3]-.2,axis=1)&np.all(point<=bounds[:,3:]+.2,axis=1));found=[];unresolved=[]
   for j in candidates:
    tag=tags[j]
    try:q,uv,error=closest_with_verified_parameters(tag,point)
    except ValueError as exc:unresolved.append(dict(face=tag,reason=str(exc)));continue
    distance=float(np.linalg.norm(q-point))
    if distance>.2:continue
    if gmsh.model.isInside(2,tag,uv.tolist(),parametric=True):found.append(dict(face=tag,volumes=sorted(owners.get(tag,set())),distance_mm=distance,normal=gmsh.model.getNormal(tag,uv.tolist()).tolist()))
   volumes=sorted({v for match in found for v in match['volumes']});return dict(volumes=volumes,faces=found,unresolved_inverse_queries=unresolved,unique_two_volume_contact_at_sample=len(volumes)==2 and not unresolved)
  for pi,p in enumerate(target['parents']):
   for ci,c in enumerate(p['children']):
    ref=reference_segment(c)
    if ref['kind']!='line':records.append(dict(parent=pi,child=ci,kind=ref['kind'],status='CURVE_NOT_AUDITED'));continue
    ends=np.asarray(ref['points'])[[0,-1]];v=ends[1]-ends[0];length=np.linalg.norm(v);v/=length;inside=[]
    for fraction in [.1,.3,.5,.7,.9]:inside.append(dict(fraction=fraction,**contacts(ends[0]+fraction*length*v)))
    candidates=[tuple(x['volumes']) for x in inside if x['unique_two_volume_contact_at_sample']];unique=len(candidates)==len(inside) and len(set(candidates))==1;outside=[]
    for side,endpoint,direction in [('start',ends[0],-v),('end',ends[1],v)]:
     for distance in [1.,5.,20.]:outside.append(dict(endpoint=side,outward_distance_mm=distance,**contacts(endpoint+distance*direction)))
    records.append(dict(parent=pi,child=ci,kind='line',interior_samples=inside,exterior_probes=outside,same_unique_volume_pair_at_all_interior_samples=unique,interior_volume_pair=list(candidates[0]) if unique else None,continuous_contact_identity_proved=False,local_annotation_closure_proved=False,process_termination_from_geometry_not_assumed=True))
 finally:gmsh.finalize()
 report=dict(source=key,records=records,line_children=sum(r['kind']=='line' for r in records),consistent_two_volume_sampled_contacts=sum(r.get('same_unique_volume_pair_at_all_interior_samples',False) for r in records),target_sha256=digest(targetpath),pose_sha256=digest(posepath),step_sha256=digest(step),GT_training_only=True,STEP_offline_only=True,negative_supervision_not_released=True,source_sha256=digest(__file__))
 write(f/'sampled_local_contact_scope.json',report);print({k:v for k,v in report.items() if k!='records'},flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--key',required=True,choices=['10','25','36']);run(ap.parse_args().key)
