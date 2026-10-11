"""Frozen neural query head on complete input grid; no reference label reads."""
import argparse,time
import numpy as np
import torch
from tools.query_reference_utility_runtime import Inputs,BASE
from tools.train_query_context_neural_comparison import Model
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--arm',choices=['local457'],required=True);a=p.parse_args();start=time.perf_counter();torch.set_num_threads(2);folder=BASE/'positive_query_head_v1/neural_expanded_ten_v1';receipt=read(folder/'receipt.json');assert digest(folder/'model.pt')==receipt['model_sha256'];state=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);model=Model(state['columns']);model.load_state_dict(state['state_dict']);model=model.cuda().eval();inp=Inputs(a.key);dest=BASE/'positive_query_neural_expanded_ten'/a.key;dest.mkdir(parents=True,exist_ok=False)
ctx=None
if a.arm=='context913':
 unique,first,counts=np.unique(inp.pid,return_index=True,return_counts=True);ctx=np.zeros((len(inp.profiles),456),np.float32)
 for pid,begin,count in zip(unique,first,counts):
  vals=np.asarray(inp.x[begin*2:(begin+count)*2],np.float64).reshape(count,2,152);span=inp.spans[inp.canonical[begin:begin+count],:2];weights=span[:,1]-span[:,0];ctx[pid]=np.r_[np.average(vals.mean(1),axis=0,weights=weights),vals.min((0,1)),vals.max((0,1))]
n=len(inp.valid);bounds=np.empty((n,2),np.float64);roles=np.empty(n,np.int8)
with torch.no_grad():
 for first in range(0,n,4096):
  ids=np.arange(first,min(first+4096,n));local=inp.features(ids)[:,:457];x=np.column_stack((local,ctx[inp.pid[ids//4]])) if ctx is not None else local;xx=np.clip((x-state['mean'])/state['std'],-20,20);prediction,role=model(torch.as_tensor(xx,device='cuda'),torch.as_tensor(local[:,456],device='cuda'));bounds[ids]=prediction.cpu().numpy();roles[ids]=role.argmax(1).cpu().numpy()
valid=np.isfinite(bounds).all(1)&(bounds[:,0]>=0)&(bounds[:,1]<=1)&(bounds[:,0]<bounds[:,1]);np.save(dest/'interval_ratios.npy',bounds.reshape(-1,4,2));np.save(dest/'weld_type.npy',roles.reshape(-1,4));np.save(dest/'valid.npy',valid.reshape(-1,4));np.save(dest/'profile_ids.npy',inp.pid);np.save(dest/'canonical_fragments.npy',inp.canonical)
write(dest/'receipt.json',dict(status='FULL_INPUT_GRID_NEURAL_PROPOSALS_NOT_SELECTED',key=a.key,arm=a.arm,queries=n,valid_queries=int(valid.sum()),invalid_queries=int((~valid).sum()),files={p.name:digest(p) for p in dest.glob('*.npy')},model_sha256=receipt['model_sha256'],source_geometry_receipt_sha256=digest(inp.folder/'receipt.json'),GT_reads=0,STEP_reads=0,fit_calls=0,old_predictions_not_used_in_features=True,not_existence_or_parent_output=True,seconds=time.perf_counter()-start));print(a.arm,a.key,n,int(valid.sum()),flush=True)
