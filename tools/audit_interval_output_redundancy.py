"""Input prediction-only redundancy diagnostic; never suppress or relabel output."""
from pathlib import Path
import collections,time
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def main():
 reports=[]
 for branch in ['bounded_raw_raw_full_output_v1','joint_interval_residual_full_output_v1']:
  for key in ['new_cfc732e5f068','new_ceeae49d6777']:
   f=BASE/branch/key;path=f/'compact_predictions.npz';freeze=read(f/'prediction_freeze.json');assert digest(path)==freeze['files']['compact_predictions.npz'];z=np.load(path);iv=z['interval_mm'];valid=np.isfinite(iv).all(1)&(iv[:,1]>iv[:,0]);ids=np.flatnonzero(valid);groups=collections.defaultdict(list)
   for i in ids:groups[int(z['profile_ids'][i]),int(z['weld_type'][i])].append(int(i))
   counts={str(t):0 for t in [1e-8,.1,1.,3.]};nonisolated={str(t):0 for t in [1e-8,.1,1.,3.]}
   for g in groups.values():
    pts=iv[g];tree=cKDTree(pts)
    for t in [1e-8,.1,1.,3.]:
     # Count neighborhood degree without materializing quadratic pair lists.
     degree=tree.query_ball_point(pts,t,p=np.inf,return_length=True)-1
     counts[str(t)]+=int(degree.sum()//2);nonisolated[str(t)]+=int((degree>0).sum())
   length=iv[ids,1]-iv[ids,0];r=dict(branch=branch,key=key,predictions=len(iv),valid=len(ids),profile_role_groups=len(groups),length_quantiles_mm=np.quantile(length,[0,.1,.5,.9,.99,1]).tolist(),length_le_3mm=int((length<=3).sum()),near_interval_pairs=counts,predictions_with_neighbor=nonisolated,compact_sha256=digest(path));reports.append(r);print(r,flush=True)
 out=BASE/'interval_output_redundancy_v1';out.mkdir(exist_ok=True);write(out/'receipt.json',dict(cases=reports,GT_reads=0,fit_calls=0,scope='Endpoint-coordinate proximity within same input profile and predicted type only. Does not prove duplicate physical weld identity, same parent/side, or authorize merging. No runtime thresholds changed.',source_sha256=digest(__file__)))
if __name__=='__main__':main()
