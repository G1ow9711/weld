"""Full-point bidirectional distances for closest rigid-screen pairs, no labels."""
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from tools.align_possible_duplicate_inputs import kabsch
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'migration_checks';F=BASE/'input_geometry_fingerprints_v1';DATA=ROOT.parent/'data/G_Data';DEST=BASE/'duplicate_full_cloud_v1'
def load(record):
 p=DATA/record['pcd_path'];assert digest(p)==record['pcd_sha256']
 with p.open('rb') as f:
  while not f.readline().startswith(b'DATA ascii'):pass
  x=np.loadtxt(f,dtype=np.float64,ndmin=2)
 assert x.shape==(record['points'],3) and np.isfinite(x).all();return x

def distances(a,tree):
 return np.concatenate([tree.query(a[i:i+131072],workers=2)[0] for i in range(0,len(a),131072)])

def main():
 DEST.mkdir(exist_ok=True);rows=read(BASE/'duplicate_rigid_alignment_v1/summary.json')['pairs'];reports=[]
 for row in rows:
  if row['bidirectional_p95_over_rms_radius']>.005:continue
  dest=DEST/(row['a'][:16]+'_'+row['b'][:16]+'.json')
  if dest.exists():reports.append(read(dest));continue
  start=time.perf_counter();ar=read(F/(row['a']+'.json'));br=read(F/(row['b']+'.json'));a=load(ar);b=load(br);tree=cKDTree(b);T=np.asarray(row['transform_a_to_b_raw_units']);R=T[:3,:3];t=T[:3,3];sample=a[np.linspace(0,len(a)-1,min(8192,len(a)),dtype=int)]
  for i in range(20):
   moved=sample@R.T+t;d,ids=tree.query(moved,workers=2);keep=d<=np.quantile(d,.95);q,shift=kabsch(moved[keep],b[ids[keep]]);R=q@R;t=q@t+shift
   if np.linalg.norm(q-np.eye(3))<1e-12 and np.linalg.norm(shift)<1e-12:break
  a=a@R.T+t;forward=distances(a,tree);del tree;backward=distances(b,cKDTree(a));scale=np.sqrt(sum(br['covariance_eigenvalues_raw_units_squared']));T[:3,:3]=R;T[:3,3]=t
  out=dict(a=row['a'],b=row['b'],a_directory=row['a_directory'],b_directory=row['b_directory'],a_points=len(a),b_points=len(b),transform_a_to_b_raw_units=T.tolist(),forward_quantiles_raw_units=np.quantile(forward,[0,.5,.9,.95,.99,1]).tolist(),backward_quantiles_raw_units=np.quantile(backward,[0,.5,.9,.95,.99,1]).tolist(),forward_fraction_within_1e_6_rms=float(np.mean(forward<=scale*1e-6)),backward_fraction_within_1e_6_rms=float(np.mean(backward<=scale*1e-6)),rms_radius_raw_units=scale,full_point_queries=True,rigid_fit_used_XYZ_only=True,GT_reads=0,STEP_reads=0,same_specimen_identity_not_proved=True,source_sha256=digest(__file__),seconds=time.perf_counter()-start);write(dest,out);reports.append(out);print(row['a_directory'],row['b_directory'],out['forward_quantiles_raw_units'],out['backward_quantiles_raw_units'],flush=True)
 write(DEST/'summary.json',dict(pairs=reports,split_assigned=False,same_specimen_identity_not_proved=True));print('FULL_CLOUD_COMPLETE',len(reports),flush=True)
if __name__=='__main__':main()
