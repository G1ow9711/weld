"""Offline finite STEP surface teacher pilot; no weld/background truth inferred."""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/step_contact_scope_audit_v1';OUT=ROOT/'server_rebuild/exact_step_surface_teacher_v1'
def main():
 OUT.mkdir(exist_ok=True);assert not (OUT/"receipt.json").exists();reports=[]
 for key in ['10','25','36']:
  folder=BASE/('source_'+key);vp=folder/'registration/finite_validation_fixed_uv/validation.json';validation=read(vp);ap=folder/'step_planar_faces_world.json';atlas=read(ap);faces={f['face']:f for f in atlas['faces']};T=np.asarray(validation['transform_STEP_to_native']);R=T[:3,:3];assert np.allclose(R.T@R,np.eye(3),atol=1e-8)
  inp=ROOT/'server_rebuild/inputs_v1'/key;ir=read(inp/'receipt.json');assert digest(inp/'xyz_mm.npy')==ir['xyz_sha256'];xyz=np.load(inp/'xyz_mm.npy',mmap_mode='r');ids=np.array([r['point_index'] for r in validation['validation']],np.int64);rows=[];normal=np.zeros((len(ids),3),np.float64);normal_known=np.zeros(len(ids),bool);face_id=np.full(len(ids),-1,np.int64);volume_id=np.full(len(ids),-1,np.int64)
  for i,row in enumerate(validation['validation']):
   exact=[m for m in row['matches'] if m['exact_trim'] and not m['boundary_nudge']];tags=sorted(set(m['face'] for m in exact));planar=all(tag in faces for tag in tags)
   # Unknown/failed candidates can hide an additional material: retain all ambiguity.
   reliable=bool(tags) and not row.get('unverified_parameter_queries',[{'reason':'Historical validation lacks explicit query-failure audit; conservatively unknown'}])
   if reliable and len(tags)==1:face_id[i]=tags[0]
   volumes=sorted({v for tag in tags if tag in faces for v in faces[tag]['volumes']})
   if reliable and planar and len(volumes)==1:volume_id[i]=volumes[0]
   if reliable and planar:
    normals=np.array([faces[tag]['normal'] for tag in tags]);normals/=np.linalg.norm(normals,axis=1,keepdims=True)
    if np.all(np.abs(normals@normals[0])>=1-1e-10):normal[i]=R@normals[0];normal_known[i]=True
   rows.append(dict(point_index=int(ids[i]),exact_face_candidates=tags,planar_candidates_complete=planar,volume_candidates=volumes,unverified_queries=row.get('unverified_parameter_queries',[{'reason':'Historical validation lacks explicit query-failure audit; conservatively unknown'}]),unique_face_known=bool(face_id[i]>=0),unique_material_known=bool(volume_id[i]>=0),unsigned_normal_known=bool(normal_known[i])))
  dest=OUT/key;dest.mkdir(exist_ok=True);np.savez(dest/'teacher.npz',point_indices=ids,xyz_mm=np.asarray(xyz[ids]),face_id=face_id,volume_id=volume_id,unsigned_normal_native=normal,unsigned_normal_known=normal_known);xyz._mmap.close();write(dest/'candidates.json',rows)
  report=dict(key=key,points=len(ids),unique_face_points=int((face_id>=0).sum()),unique_volume_points=int((volume_id>=0).sum()),unsigned_normal_points=int(normal_known.sum()),all_unknown_points=int(((face_id<0)&(volume_id<0)&~normal_known).sum()),teacher_sha256=digest(dest/'teacher.npz'),candidates_sha256=digest(dest/'candidates.json'),validation_sha256=digest(vp),atlas_sha256=digest(ap),XYZ_sha256=ir['xyz_sha256']);reports.append(report);print(report,flush=True)
 write(OUT/'receipt.json',dict(status='OFFLINE_SURFACE_TEACHER_PILOT_NOT_TRAINED',cases=reports,scope='Unique material/face and unsigned normal masks on exact finite registered CAD matches. Local CAD IDs are sample-specific instance labels, never global categorical IDs or inference inputs. Unknown or nudged matches are not negatives.',GT_weld_reads=0,DEV_reads=0,optimizer_updates=0,registration_validation_reuse='These points were held out of pose fitting. If later used in training they cease to be independent model evaluation points; frozen registration was not refit.',CAD_is_not_customer_material_truth=True,annotation_closure_proved=False,source_sha256=digest(__file__)))
if __name__=='__main__':main()
