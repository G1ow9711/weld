"""Whole fixed-detector query output, before/after grid-head adaptation.

The detector/query/type scope is held fixed to isolate endpoint adaptation.
This is not deployment of the ungated branch or a complete parent selector.
"""
from pathlib import Path
import time
import numpy as np
import torch
from weld_pipeline.server_fixed_fragment_queries import FixedQueries,GRID
from weld_pipeline.server_grid_endpoint_selector import GridEndpointSelector
from weld_pipeline.server_grid_candidate_runtime import score_grid_profile
from weld_pipeline.server_compact_line_predictions import build_compact_lines
from weld_pipeline.server_array_cached_d3_evaluation import score_native_parents_cached
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'

def run(key,arm,model,state,model_sha):
 start=time.perf_counter();out=BASE/'grid_endpoint_comparison_v1'/arm/key;out.mkdir(parents=True,exist_ok=False)
 source=BASE/'frozen_features/legacy_runtime'/('source_'+key);inp=FixedQueries(source,key)
 old=BASE/'reference_detector_balanced_full_output_v1'/key;freeze=read(old/'prediction_freeze.json');assert digest(old/'predictions.npz')==freeze['predictions_sha256']
 with np.load(old/'predictions.npz') as z:
  ids=np.flatnonzero(z['valid']&z['line_eligible']&(z['reference_occupancy_score']>=.5));roles=z['weld_type'][ids].copy()
 pids=inp.pid[ids//4];pair_bounds=np.full((len(ids),64,2),np.nan);pair_valid=np.zeros((len(ids),64),bool);bounds=np.zeros((len(ids),2));eventids=np.zeros((len(ids),2),dtype=np.int64);seen=np.zeros(len(ids),bool)
 for index,pid in enumerate(np.unique(pids)):
  a,b=inp.offsets[pid:pid+2];spans=inp.spans[a:b,:2];length=float(spans[-1,1]);reps=inp.representative[np.arange(a,b)];rows=np.searchsorted(inp.canonical,reps);assert np.array_equal(inp.canonical[rows],reps)
  matching=np.flatnonzero(pids==pid)
  for begin in range(0,len(matching),16):
   local=matching[begin:begin+16];query_ids=ids[local];x=inp.features(query_ids);qs=inp.spans[inp.canonical[query_ids//4],:2];qm=qs[:,0]+GRID[query_ids%4]*(qs[:,1]-qs[:,0])
   result=score_grid_profile(model,x,qm,length,spans,rows,inp.raw,state['mean'],state['std']);pairs=result['pairs'];n=pairs['valid'].shape[1]
   pair_bounds[local,:n,0]=pairs['left_mm'].cpu().numpy();pair_bounds[local,:n,1]=pairs['right_mm'].cpu().numpy();pair_valid[local,:n]=pairs['valid'].cpu().numpy()
   selection=pairs['base_logits'].argmax(1);batch=torch.arange(len(local),device=selection.device);has=pairs['valid'].any(1).cpu().numpy()
   bounds[local]=torch.stack((pairs['left_mm'][batch,selection],pairs['right_mm'][batch,selection]),1).cpu().numpy();bounds[local[~has]]=0
   eventids[local]=torch.stack((pairs['left_ids'][batch,selection],pairs['right_ids'][batch,selection]),1).cpu().numpy();seen[local]=True
  if index%200==0:print(arm,key,'profile',index,'queries',int(seen.sum()),flush=True)
 assert seen.all()
 np.savez_compressed(out/'pair_proposals.npz',selected_query_ids=ids,profile_ids=pids,interval_mm=pair_bounds,valid=pair_valid,weld_type=roles)
 decoded,compact=build_compact_lines(inp.profiles,pids,bounds,roles,ids,eventids);np.savez_compressed(out/'compact_predictions.npz',**compact)
 write(out/'receipt.json',dict(status='INPUT_ONLY_FIXED_PAIR_POOL_FROZEN',queries=len(ids),budget=64,valid_pairs=int(pair_valid.sum()),GT_reads=0,STEP_reads=0,pairs_sha256=digest(out/'pair_proposals.npz'),model_sha256=model_sha,input_bindings=inp.bindings,detector_freeze_sha256=digest(old/'prediction_freeze.json'),compact_sha256=digest(out/'compact_predictions.npz'),scope='Fixed old detector query/type scope; grid-only input endpoints, singleton decode; not final deployment',runtime_sha256=digest('weld_pipeline/server_grid_candidate_runtime.py')))
 # Predictions are frozen before evaluation reads TRAIN reference labels.
 target=read(BASE/'labels'/key/'targets.json');score=score_native_parents_cached(decoded,target,[False]*len(target['parents']));write(out/'new_score.json',score)
 write(out/'evaluation_receipt.json',dict(status='SCORED_NOT_PROMOTED',overall=score['categories']['overall'],seconds=time.perf_counter()-start,DEV_reads=0,fit_calls=0));print('SCORED',arm,key,score['categories']['overall'],flush=True);inp.close()

def main():
 torch.set_num_threads(2);folder=BASE/'grid_endpoint_head_v1';r=read(folder/'receipt.json');assert digest(folder/'last.pt')==r['model_sha256']
 raw=BASE/'bounded_raw_selector_v1/model.pt';old=torch.load(raw,map_location='cpu',weights_only=False);new=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
 for arm,state,sha in [('initial',old,digest(raw)),('trained',new,r['model_sha256'])]:
  model=GridEndpointSelector().cuda().eval()
  if arm=='initial':model.load_raw_state_dict(state['state_dict'])
  else:model.load_state_dict(state['state_dict'])
  for key in ['new_cfc732e5f068','new_ceeae49d6777']:run(key,arm,model,state,sha)
if __name__=='__main__':main()
