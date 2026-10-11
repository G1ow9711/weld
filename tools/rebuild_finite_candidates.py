import argparse,hashlib,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.server_finite_candidates import extract,guard
ap=argparse.ArgumentParser();ap.add_argument('--key',required=True);ap.add_argument('--plane-method',choices=['rank2','local','global'],default='rank2');ap.add_argument('--version',default='v1');args=ap.parse_args()
out=ROOT/('server_rebuild/candidates_'+args.version)/args.key;out.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def serial(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 raise TypeError(type(v).__name__)
def write(name,value): (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=serial))
start=time.monotonic()
def progress(*v):
 
 if v[0] not in ('CONTACT_FACE','BOUNDARY_ARCS') or v[1]%25==0:print(*v,flush=True)
 with (out/'events.jsonl').open('a') as f:f.write(json.dumps(dict(seconds=time.monotonic()-start,event=v,resources=guard()))+'\n')
source=ROOT/'server_rebuild/inputs_v1'/args.key/'xyz_mm.npy'
write('protocol.json',dict(plane_method=args.plane_method,input_sha256=sha(source),input_only=True,GT_reads=0,STEP_reads=0,source_sha256={p.name:sha(p) for p in (ROOT/'weld_pipeline').glob('server_*.py')},speed_acceptance=False,new_reconstruction_not_run458_resume=True))
try:
 import shutil
 for p in (ROOT/'weld_pipeline').glob('server_*.py'):shutil.copyfile(p,out/(p.stem+'_snapshot.py'))
 xyz=np.load(source,mmap_mode='r')
 with threadpool_limits(limits=2):faces,candidates,stats=extract(xyz,progress,args.plane_method)
 arrays={};metadata=[]
 for i,face in enumerate(faces):
  arrays[f'ids_{i}']=face['point_ids'];arrays[f'tri_{i}']=face['triangles'];metadata.append({k:v for k,v in face.items() if k not in ('point_ids','triangles')})
 np.savez_compressed(out/'face_support.npz',**arrays);write('faces.json',metadata);write('candidates.json',candidates)
 write('receipt.json',dict(status='PURE_XYZ_CANDIDATES_COMPLETE',stats=stats,seconds=time.monotonic()-start,resources=guard(),files={n:sha(out/n) for n in ['face_support.npz','faces.json','candidates.json']},GT_reads=0,STEP_reads=0,accuracy_test=False))
 print('COMPLETE',args.key,len(faces),len(candidates),flush=True)
except Exception:
 import traceback
 write('failure.json',dict(error=traceback.format_exc(),seconds=time.monotonic()-start));raise
