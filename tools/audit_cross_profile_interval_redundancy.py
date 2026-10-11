"""Geometric redundancy bound; no assumption of identical physical welds."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def main():
 rows=[]
 for key in ['new_cfc732e5f068','new_ceeae49d6777']:
  folder=BASE/'bounded_raw_raw_full_output_v1'/key;path=folder/'compact_predictions.npz';assert digest(path)==read(folder/'prediction_freeze.json')['files']['compact_predictions.npz'];z=np.load(path);ends=z['endpoints_mm'];valid=np.isfinite(ends).all((1,2));ids=np.flatnonzero(valid);pts=ends[ids];forward=pts.reshape(-1,6);reverse=pts[:,::-1].reshape(-1,6);roles=z['weld_type'][ids];counts={}
  for t in [1e-8,.1,1.,3.]:
   neighbor=np.zeros(len(ids),bool);pairs=0
   for role in np.unique(roles):
    g=np.flatnonzero(roles==role);tree=cKDTree(forward[g]);degree=np.maximum(tree.query_ball_point(forward[g],t,p=np.inf,return_length=True),tree.query_ball_point(reverse[g],t,p=np.inf,return_length=True))-1
    # Either orientation; max-degree is a lower bound on union neighborhood,
    # while non-isolated predicate is exact after self is present forward.
    has=(tree.query_ball_point(forward[g],t,p=np.inf,return_length=True)>1)|(tree.query_ball_point(reverse[g],t,p=np.inf,return_length=True)>0)
    # Reverse self can satisfy the radius for very short segments; remove self
    # by requiring reverse count > self-overlap indicator.
    revself=np.max(np.abs(forward[g]-reverse[g]),axis=1)<=t
    has=(tree.query_ball_point(forward[g],t,p=np.inf,return_length=True)>1)|(tree.query_ball_point(reverse[g],t,p=np.inf,return_length=True)>revself.astype(int))
    neighbor[g]=has
   counts[str(t)]=int(neighbor.sum())
  rows.append(dict(key=key,valid_predictions=len(ids),predictions_with_geometric_neighbor=counts,compact_sha256=digest(path)));print(rows[-1],flush=True)
 out=BASE/'cross_profile_interval_redundancy_v1';out.mkdir(exist_ok=True);write(out/'receipt.json',dict(cases=rows,metric='Either orientation, max absolute difference across six endpoint coordinates <= radius; same predicted weld type; all profiles.',scope='Diagnostic only. Coordinate-wise cube proximity does not imply Euclidean3mm, identical side/material/parent, or permission to merge.',GT_reads=0,fit_calls=0,source_sha256=digest(__file__)))
if __name__=='__main__':main()
