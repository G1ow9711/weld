"""Recompute finite line contacts from verified pure-input face supports."""
import argparse,hashlib,json,pathlib,sys,time,shutil
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.server_finite_candidates import contacts,guard
ap=argparse.ArgumentParser();ap.add_argument('--key',required=True);a=ap.parse_args()
src=ROOT/'server_rebuild/candidates_v5'/a.key;dest=ROOT/'server_rebuild/candidates_v6'/a.key
r=json.loads((src/'receipt.json').read_text());assert r['status']=='PURE_XYZ_CANDIDATES_COMPLETE'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in r['files'].items():assert sha(src/n)==h
protocol=json.loads((src/'protocol.json').read_text());xfile=ROOT/'server_rebuild/inputs_v1'/a.key/'xyz_mm.npy';assert sha(xfile)==protocol['input_sha256']
x=np.load(xfile,mmap_mode='r');faces=json.loads((src/'faces.json').read_text())
with np.load(src/'face_support.npz') as z:
 for i,f in enumerate(faces):
  for k in ['normal','center','basis']:f[k]=np.asarray(f[k])
  f['point_ids']=z[f'ids_{i}'];f['triangles']=z[f'tri_{i}']
start=time.perf_counter()
with threadpool_limits(limits=2):lines=contacts(faces,x)
old=json.loads((src/'candidates.json').read_text());arcs=[c for c in old if c['representation']['kind']=='arc']
dest.mkdir(parents=True,exist_ok=False)
shutil.copyfile(ROOT/'weld_pipeline/server_finite_candidates.py',dest/'server_finite_candidates_snapshot.py')
(dest/'candidates.json').write_text(json.dumps(lines+arcs,ensure_ascii=False,indent=2))
receipt=dict(status='PURE_XYZ_CANDIDATES_COMPLETE',source_receipt_sha256=sha(src/'receipt.json'),source_face_files=r['files'],input_sha256=protocol['input_sha256'],lines=len(lines),arcs=len(arcs),seconds=time.perf_counter()-start,GT_reads=0,STEP_reads=0,accuracy_test=False,files={'candidates.json':sha(dest/'candidates.json')},intersection_tolerance='max(1e-6,sum of observed residual maxima of the two fitted planes)',resources=guard())
(dest/'receipt.json').write_text(json.dumps(receipt,indent=2));print(a.key,len(lines),len(arcs),flush=True)
