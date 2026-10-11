"""Offline input-only dominant plane audit with no acceptance claim."""
import pathlib,json,time
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=pathlib.Path(__file__).resolve().parents[1];x=np.load(ROOT/'server_rebuild/inputs_v1/21/xyz_mm.npy',mmap_mode='r');rng=np.random.default_rng(20261010);remaining=np.arange(len(x));planes=[];start=time.perf_counter()
with threadpool_limits(limits=2):
 for k in range(32):
  if len(remaining)<32:break
  sample=x[rng.choice(remaining,min(8192,len(remaining)),replace=False)]
  triples=sample[rng.integers(len(sample),size=(512,3))];normal=np.cross(triples[:,1]-triples[:,0],triples[:,2]-triples[:,0]);length=np.linalg.norm(normal,axis=1);good=length>1e-6
  normal=normal[good]/length[good,None];offset=(triples[good,0]*normal).sum(1)
  scores=(abs(sample@normal.T-offset)<.01).sum(0);best=int(scores.argmax());n=normal[best];off=offset[best]
  ids=remaining[abs(x[remaining]@n-off)<.01]
  if len(ids)<32:break
  points=x[ids];center=points.mean(0);e,v=np.linalg.eigh((points-center).T@(points-center));n=v[:,0]
  if n[np.argmax(abs(n))]<0:n=-n
  off=float(center@n);mask=abs(x[remaining]@n-off)<.01;ids=remaining[mask];remaining=remaining[~mask]
  planes.append(dict(normal=n.tolist(),offset=off,points=len(ids),sample_support=int(scores[best]),remaining=len(remaining)))
  print(k+1,planes[-1],flush=True)
result=dict(planes=planes,remaining=len(remaining),points=len(x),seconds=time.perf_counter()-start,plane_cap=32,GT_reads=0,STEP_reads=0,diagnosis_only=True)
(ROOT/'migration_checks/global_plane_diagnosis_21.json').write_text(json.dumps(result,indent=2))
