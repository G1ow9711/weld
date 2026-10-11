import json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.sjg_observed_cylinders_919 import canonical_normals
x=np.load(ROOT/'server_rebuild/inputs_v1/21/xyz_mm.npy',mmap_mode='r');start=time.perf_counter()
with threadpool_limits(limits=2):
 d=canonical_normals(x);e=d['eigen'];mask=d['rank2']&(e[:,0]<=1e-8)
 n=d['normal'][mask];p=d['xyz'][mask];offset=(n*p).sum(1)
 keys=np.c_[np.round(n/1e-5),np.round(offset/.01)].astype(np.int64)
 unique,inv,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
 rows=[]
 for k in np.argsort(-counts)[:50]:
  if counts[k]<16:break
  q=p[inv==k];center=q.mean(0);values,vectors=np.linalg.eigh((q-center).T@(q-center)/len(q));normal=vectors[:,0]
  if normal[np.argmax(abs(normal))]<0:normal=-normal
  rows.append(dict(seed_points=int(counts[k]),normal=normal.tolist(),offset=float(center@normal),plane_residual_mm=float(abs((q-center)@normal).max()),in_plane_scales_mm=np.sqrt(np.maximum(values[1:],0)).tolist()))
result=dict(points=len(x),canonical=len(d['xyz']),exact_planar_seeds=int(mask.sum()),groups_ge16=int((counts>=16).sum()),largest_groups=rows,seconds=time.perf_counter()-start,GT_reads=0,STEP_reads=0)
(ROOT/'migration_checks/plane_seed_diagnosis_21.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
