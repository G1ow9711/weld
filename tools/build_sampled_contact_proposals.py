import argparse,hashlib,json,pathlib,sys,time,shutil
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.server_sampled_contact_chains import proposals,controls
from weld_pipeline.server_finite_candidates import guard
ap=argparse.ArgumentParser();ap.add_argument('--key',required=True);a=ap.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
start=time.perf_counter();src=ROOT/'server_rebuild/candidates_v5'/a.key;prev=ROOT/'server_rebuild/candidates_v6'/a.key;dest=ROOT/'server_rebuild/candidates_v7'/a.key
r=json.loads((src/'receipt.json').read_text());r6=json.loads((prev/'receipt.json').read_text())
for n,h in r['files'].items():assert sha(src/n)==h
for n,h in r6['files'].items():assert sha(prev/n)==h
xfile=ROOT/'server_rebuild/inputs_v1'/a.key/'xyz_mm.npy';assert sha(xfile)==r6['input_sha256'];xyz=np.load(xfile,mmap_mode='r')
faces=json.loads((src/'faces.json').read_text())
with np.load(src/'face_support.npz') as z:
 for i,f in enumerate(faces):
  f['normal']=np.array(f['normal']);f['point_ids']=z[f'ids_{i}']
with threadpool_limits(limits=2):found=proposals(faces,xyz)
old=json.loads((prev/'candidates.json').read_text());dest.mkdir(parents=True,exist_ok=False)
shutil.copyfile(ROOT/'weld_pipeline/server_sampled_contact_chains.py',dest/'source_snapshot.py')
(dest/'candidates.json').write_text(json.dumps(old+found,ensure_ascii=False,indent=2))
receipt=dict(status='UNCERTAIN_PROPOSALS_COMPLETE',source_receipt_sha256=sha(prev/'receipt.json'),source_mesh_receipt_sha256=sha(src/'receipt.json'),input_sha256=sha(xfile),files={n:sha(dest/n) for n in ['candidates.json','source_snapshot.py']},previous=len(old),new_uncertain_sampled_chains=len(found),seconds=time.perf_counter()-start,GT_reads=0,STEP_reads=0,controls=controls(),resources=guard(),model_accuracy_test=False)
(dest/'receipt.json').write_text(json.dumps(receipt,indent=2));print(a.key,len(found),flush=True)
