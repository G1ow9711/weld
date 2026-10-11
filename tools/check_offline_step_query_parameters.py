"""Check actual finite-trim adapter on a hole fixture and imported STEP mesh."""
from pathlib import Path
import ctypes
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ctypes.CDLL(str(ROOT/'server_rebuild/cad_runtime_libs/extracted/usr/lib/x86_64-linux-gnu/libGLU.so.1'),mode=ctypes.RTLD_GLOBAL)
import gmsh
from weld_pipeline.server_offline_step_queries import closest_with_verified_parameters
from weld_pipeline.sjg_local_runtime_904 import write,digest

def main():
 gmsh.initialize([],readConfigFiles=False);gmsh.option.setNumber('General.Terminal',0);gmsh.option.setString('Geometry.OCCTargetUnit','MM');rows=[]
 try:
  plane=gmsh.model.occ.addRectangle(0,0,0,10,10);disk=gmsh.model.occ.addDisk(5,5,0,2,2);entities,_=gmsh.model.occ.cut([(2,plane)],[(2,disk)]);gmsh.model.occ.synchronize();face=entities[0][1]
  for point,wanted in [([1.,1.,0.],True),([5.,5.,0.],False),([12.,5.,0.],False)]:
   error=None;rejected=None
   try:
    q,uv,error=closest_with_verified_parameters(face,point);inside=bool(gmsh.model.isInside(2,face,uv.tolist(),parametric=True));accepted=inside and np.linalg.norm(q-point)<=.2
   except ValueError as exc:accepted=False;rejected=str(exc)
   assert accepted==wanted;rows.append(dict(kind='hole_fixture',point=point,expected=wanted,actual=bool(accepted),inverse_map_error_mm=error,unverified_parameter_rejected=rejected))
  gmsh.clear();step=next((ROOT.parent/'data/G_Data/标注数据20261004/真值/2-161-FR62C-0928').glob('*.stp'));gmsh.model.occ.importShapes(str(step));gmsh.model.occ.synchronize();f=ROOT/'server_rebuild/step_contact_scope_audit_v1/source_36/registration/cad_mesh.npz'
  with np.load(f) as z:v=z['vertices'];tri=z['triangles'];owners=z['face_ids']
  for i in np.linspace(0,len(tri)-1,12).astype(int):
   tag=int(owners[i]);point=v[tri[i]].mean(0);oldq,olduv=gmsh.model.getClosestPoint(2,tag,point.tolist());q,uv,error=closest_with_verified_parameters(tag,point);inside=bool(gmsh.model.isInside(2,tag,uv.tolist(),parametric=True));assert inside and np.linalg.norm(q-point)<.2;rows.append(dict(kind='imported_mesh_interior',face=tag,old_returned_parameters=list(olduv),recomputed_parameters=uv.tolist(),old_trim_result=bool(gmsh.model.isInside(2,tag,list(olduv),parametric=True)),verified_trim_result=inside,inverse_map_error_mm=error))
 finally:gmsh.finalize()
 out=ROOT/'server_rebuild/step_contact_scope_audit_v1/query_parameter_controls.json';write(out,dict(PASS=True,rows=rows,source_sha256=digest(__file__),adapter_sha256=digest(ROOT/'weld_pipeline/server_offline_step_queries.py'),scope='Finite trim parameter adapter correctness, not registration or annotation completeness'));print('STEP_QUERY_CONTROLS_PASS',flush=True)
if __name__=='__main__':main()
