"""Full input grid with complete ordered profile tokens; no GT reads."""
import argparse,time
import numpy as np
import torch
from tools.query_reference_utility_runtime import Inputs,BASE
from weld_pipeline.server_ordered_profile_context import ProfileSequences,OrderedContextModel
from weld_pipeline.sjg_local_runtime_904 import read,write,digest

def run(key):
 start=time.perf_counter();torch.set_num_threads(2);folder=BASE/'positive_query_head_v1/ordered_context_v1';r=read(folder/'receipt.json');assert digest(folder/'model.pt')==r['model_sha256'];s=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);model=OrderedContextModel().cuda().eval();model.load_state_dict(s['state_dict']);inp=Inputs(key);provider=ProfileSequences(BASE/'frozen_features/legacy_runtime'/('source_'+key),key);dest=BASE/'positive_query_ordered_context'/key;dest.mkdir(parents=True,exist_ok=False);n=len(inp.valid);bounds=np.empty((n,2),np.float64);roles=np.empty(n,np.int8);seen=np.zeros(n,bool);tensor=lambda v:torch.as_tensor(v,device='cuda',dtype=torch.float32);maximum=0
 unique,first,counts=np.unique(inp.pid,return_index=True,return_counts=True)
 with torch.no_grad():
  for pid,begin,count in zip(unique,first,counts):
   assert np.all(inp.pid[begin:begin+count]==pid);raw,length=provider.tokens(int(pid));maximum=max(maximum,len(raw));tokens=tensor(np.clip((raw-s['token_mean'])/s['token_std'],-20,20))[None];mask=torch.ones((1,len(raw)),device='cuda',dtype=torch.bool)
   for offset in range(begin*4,(begin+count)*4,128):
    ids=np.arange(offset,min(offset+128,(begin+count)*4));local=inp.features(ids)[:,:457];p,t=model(tensor(np.clip((local-s['mean'])/s['std'],-20,20)),tensor(local[:,456]),tokens.expand(len(ids),-1,-1),mask.expand(len(ids),-1));bounds[ids]=p.cpu().numpy();roles[ids]=t.argmax(1).cpu().numpy();seen[ids]=True
 assert seen.all();provider.close();valid=np.isfinite(bounds).all(1)&(bounds[:,0]>=0)&(bounds[:,1]<=1)&(bounds[:,0]<bounds[:,1]);np.save(dest/'interval_ratios.npy',bounds.reshape(-1,4,2));np.save(dest/'weld_type.npy',roles.reshape(-1,4));np.save(dest/'valid.npy',valid.reshape(-1,4));np.save(dest/'profile_ids.npy',inp.pid);np.save(dest/'canonical_fragments.npy',inp.canonical)
 write(dest/'receipt.json',dict(status='FULL_INPUT_GRID_ORDERED_PROPOSALS_NOT_SELECTED',key=key,queries=n,valid_queries=int(valid.sum()),invalid_queries=int((~valid).sum()),maximum_context_tokens=maximum,context_truncation=False,files={p.name:digest(p) for p in dest.glob('*.npy')},model_sha256=r['model_sha256'],source_geometry_receipt_sha256=digest(inp.folder/'receipt.json'),input_bindings=provider.bindings,GT_reads=0,STEP_reads=0,fit_calls=0,not_existence_or_parent_output=True,seconds=time.perf_counter()-start));print(key,n,int(valid.sum()),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);run(p.parse_args().key)
