"""No-label rigid registration of conservative geometry-screen candidate pairs.

Sample-to-sample residuals are diagnostic, not proof of independent specimens.
"""
from pathlib import Path
import itertools,time
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'migration_checks';F=BASE/'input_geometry_fingerprints_v1';DEST=BASE/'duplicate_rigid_alignment_v1'
def kabsch(a,b):
 ac=a.mean(0);bc=b.mean(0);u,s,vt=np.linalg.svd((a-ac).T@(b-bc));r=vt.T@u.T
 if np.linalg.det(r)<0:vt[-1]*=-1;r=vt.T@u.T
 return r,bc-r@ac

def main():
 DEST.mkdir(exist_ok=True);source=read(BASE/'geometry_duplicate_pair_candidates_v1.json');results=[]
 for row in source['pairs']:
  if not row['within_conservative_retrieval_band']:continue
  dest=DEST/(row['a'][:16]+'_'+row['b'][:16]+'.json')
  if dest.exists():results.append(read(dest));continue
  start=time.perf_counter();a=read(F/(row['a']+'.json'));b=read(F/(row['b']+'.json'))
  for r in [a,b]:assert digest(F/(r['pcd_sha256']+'.npz'))==r['sample_sha256']
  with np.load(F/(row['a']+'.npz')) as z:ap=z['points'];am=z['center'];av=z['principal_axes']
  with np.load(F/(row['b']+'.npz')) as z:bp=z['points'];bm=z['center'];bv=z['principal_axes']
  ac=ap-am;bc=bp-bm;tree=cKDTree(bc);small=ac[np.linspace(0,len(ac)-1,min(1024,len(ac)),dtype=int)];hyp=[]
  for perm in itertools.permutations(range(3)):
   for signs in itertools.product([-1,1],repeat=3):
    r=bv[:,perm]@np.diag(signs)@av.T
    if np.linalg.det(r)<0:continue
    d,_=tree.query(small@r.T);hyp.append((float(np.mean(np.minimum(d,np.quantile(d,.9)))),r))
  best=None
  for _,r in sorted(hyp,key=lambda x:x[0])[:4]:
   t=np.zeros(3)
   for step in range(60):
    moved=ac@r.T+t;d,idx=tree.query(moved);keep=d<=np.quantile(d,.9);q,shift=kabsch(moved[keep],bc[idx[keep]]);r=q@r;t=q@t+shift
    if np.linalg.norm(q-np.eye(3))<1e-10 and np.linalg.norm(shift)<1e-10:break
   moved=ac@r.T+t;forward=tree.query(moved)[0];backward=cKDTree(moved).query(bc)[0];score=max(float(np.quantile(forward,.95)),float(np.quantile(backward,.95)))
   if best is None or score<best[0]:best=(score,r.copy(),t.copy(),forward,backward)
  score,r,t,forward,backward=best;scale=np.sqrt(sum(b['covariance_eigenvalues_raw_units_squared']));T=np.eye(4);T[:3,:3]=r;T[:3,3]=bm+t-r@am;spacing=tree.query(bc,k=2)[0][:,1];out=dict(**row,transform_a_to_b_raw_units=T.tolist(),forward_quantiles_raw_units=np.quantile(forward,[0,.5,.9,.95,.99,1]).tolist(),backward_quantiles_raw_units=np.quantile(backward,[0,.5,.9,.95,.99,1]).tolist(),bidirectional_p95_over_rms_radius=score/scale,b_sample_spacing_median_raw_units=float(np.median(spacing)),sample_proximity_not_identity_proof=True,all_points_validation_not_performed=True,GT_reads=0,STEP_reads=0,seconds=time.perf_counter()-start,source_sha256=digest(__file__));write(dest,out);results.append(out);print(row['a_directory'],row['b_directory'],'normalized_p95',round(score/scale,6),flush=True)
 write(DEST/'summary.json',dict(pairs=results,GT_reads=0,split_assigned=False,full_cropped_or_resampled_duplicate_coverage_not_proved=True));print('ALIGNMENTS_COMPLETE',len(results),flush=True)
if __name__=='__main__':main()
