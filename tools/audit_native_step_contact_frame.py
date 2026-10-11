"""Offline TRAIN STEP atlas and native-export frame check; never inference input."""
import argparse,ctypes,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ctypes.CDLL(str(ROOT/'server_rebuild/cad_runtime_libs/extracted/usr/lib/x86_64-linux-gnu/libGLU.so.1'),mode=ctypes.RTLD_GLOBAL)
import gmsh
from weld_pipeline.server_offline_step_queries import closest_with_verified_parameters
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
DATA=ROOT.parent/'data/G_Data';BASE=ROOT/'server_rebuild/step_contact_scope_audit_v1'
def run(key):
 start=time.perf_counter();dest=BASE/('source_'+key);dest.mkdir(parents=True,exist_ok=False);inp=ROOT/'server_rebuild/inputs_v1'/key;ir=read(inp/'receipt.json');raw=DATA/ir['source_file'];folder=raw.parent;steps=[p for p in folder.iterdir() if p.suffix.lower() in ('.stp','.step')];exports=list(folder.glob('*_weld_welds.json'));assert len(steps)==len(exports)==1
 metadata=read(exports[0]);assert metadata['coordinateFrame']=='reference_workpiece' and metadata['unit']=='millimeter' and metadata['matrixConvention']=='row_major_column_vector';transform=np.asarray(metadata['exportToWorld']);R=transform[:3,:3];t=transform[:3,3];assert np.allclose(R.T@R,np.eye(3),atol=1e-9) and abs(np.linalg.det(R)-1)<1e-9;assert digest(inp/'xyz_mm.npy')==ir['xyz_sha256']
 write(dest/'protocol.json',dict(source=key,role='TRAIN',STEP_offline_only=True,reference_is_not_customer_approved_material_truth=True,source_sha256=digest(__file__),step_sha256=digest(steps[0]),metadata_sha256=digest(exports[0]),input_sha256=ir['xyz_sha256'],sample_count=1024,seed=1031,plane_and_bbox_tolerance_mm=.2,finite_trim_inward_nudge_mm=.1,no_new_negative_labels=True,no_ICP_or_GT_fitted_transform=True))
 gmsh.initialize([],readConfigFiles=False);gmsh.option.setNumber('General.Terminal',0);gmsh.option.setString('Geometry.OCCTargetUnit','MM')
 try:
  gmsh.model.occ.importShapes(str(steps[0]));gmsh.model.occ.synchronize();volumes=gmsh.model.getEntities(3);owners={}
  for _,v in volumes:
   for dim,tag in gmsh.model.getBoundary([(3,v)],combined=False,oriented=False):
    if dim==2:owners.setdefault(tag,[]).append(v)
  faces=[];nonplanar=[]
  for _,tag in gmsh.model.getEntities(2):
   kind=gmsh.model.getType(2,tag)
   lo,hi=gmsh.model.getParametrizationBounds(2,tag);lo=np.asarray(lo);hi=np.asarray(hi);fractions=np.array([[.3,.3],[.3,.7],[.5,.5],[.7,.3],[.7,.7]]);uv=lo+fractions*(hi-lo);normals=np.asarray(gmsh.model.getNormal(tag,uv.ravel().tolist())).reshape(-1,3);n=normals[2];n/=np.linalg.norm(n);center=np.asarray(gmsh.model.occ.getCenterOfMass(2,tag));sample=np.asarray(gmsh.model.getValue(2,tag,uv.ravel().tolist())).reshape(-1,3);spread=float(np.rad2deg(np.arccos(np.clip(normals@n,-1,1))).max());residual=float(np.max(abs((sample-center)@n)))
   if not np.isfinite(normals).all() or spread>.02 or residual>.01:nonplanar.append(dict(face=tag,type=kind,normal_spread_deg=spread,sampled_plane_residual_mm=residual));continue
   faces.append(dict(face=tag,volumes=owners.get(tag,[]),type=kind,normal=n.tolist(),center=center.tolist(),bbox=gmsh.model.getBoundingBox(2,tag),area_mm2=gmsh.model.occ.getMass(2,tag),planarity_method='Five parameter samples, original reference audit tolerances; finite containment queried on exact STEP face',normal_spread_deg=spread,sampled_plane_residual_mm=residual))
  write(dest/'step_planar_faces_world.json',dict(faces=faces,nonplanar=nonplanar,volumes=[v for _,v in volumes],exportToWorld=transform.tolist(),step_sha256=digest(steps[0]),gmsh_version=gmsh.__version__))
  xyz=np.load(inp/'xyz_mm.npy',mmap_mode='r');ids=np.sort(np.random.default_rng(1031).choice(len(xyz),min(1024,len(xyz)),replace=False));points=np.asarray(xyz[ids])@R.T+t;normal=np.array([f['normal'] for f in faces]);center=np.array([f['center'] for f in faces]);boxes=np.array([f['bbox'] for f in faces]);records=[]
  for ix,point in zip(ids,points):
   possible=np.flatnonzero((abs(((point-center)*normal).sum(1))<=.2)&np.all(point>=boxes[:,:3]-.2,axis=1)&np.all(point<=boxes[:,3:]+.2,axis=1));found=[]
   for j in possible:
    f=faces[j];q,unused=gmsh.model.getClosestPoint(2,f['face'],point.tolist());distance=float(np.linalg.norm(np.asarray(q)-point))
    if distance>.2:continue
    try:q,uv,_=closest_with_verified_parameters(f['face'],point)
    except ValueError:continue # Unverified inverse map is never admitted as finite support.
    inside=bool(gmsh.model.isInside(2,f['face'],list(uv),parametric=True));nudged=False
    if not inside:
     delta=np.asarray(f['center'])-q;delta-=np.asarray(f['normal'])*np.dot(delta,f['normal']);norm=np.linalg.norm(delta)
     if norm>1e-8:
      q2,u2=gmsh.model.getClosestPoint(2,f['face'],(np.asarray(q)+.1*delta/norm).tolist());u2=gmsh.model.getParametrization(2,f['face'],list(q2));nudged=bool(gmsh.model.isInside(2,f['face'],list(u2),parametric=True))
    if inside or nudged:found.append(dict(face=f['face'],volumes=f['volumes'],distance_mm=distance,exact_trim_inside=inside,boundary_nudge_used=not inside))
   records.append(dict(point_index=int(ix),finite_planar_contacts=found))
  xyz._mmap.close();exact=sum(any(c['exact_trim_inside'] for c in r['finite_planar_contacts']) for r in records);supported=sum(bool(r['finite_planar_contacts']) for r in records)
  report=dict(source=key,volumes=len(volumes),planar_faces=len(faces),nonplanar_faces=len(nonplanar),sampled=len(records),sampled_exact_finite_planar_support=exact,sampled_support_with_boundary_nudge=supported,sampled_support_fraction=supported/len(records),records=records,metadata_transform_applied_without_refitting=True,GT_weld_coordinates_used=False,local_annotation_completeness_proved=False,material_identity_assignment_proved=False,all_input_points_checked=False,seconds=time.perf_counter()-start,atlas_sha256=digest(dest/'step_planar_faces_world.json'),protocol_sha256=digest(dest/'protocol.json'))
  write(dest/'frame_audit.json',report);print({k:v for k,v in report.items() if k!='records'},flush=True)
 finally:gmsh.finalize()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--key',required=True,choices=['10','25','36']);run(ap.parse_args().key)
