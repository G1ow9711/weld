"""Eight frozen complete interval hypotheses at every fixed input query."""
import argparse,pickle
import numpy as np
from tools.query_reference_utility_runtime import Inputs,BASE
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args();protocol=read(BASE/'paired_tree_interval_modes_v1/protocol.json');modelpath=BASE/'positive_query_head_v1/model_initial_seven_families/model.pkl';assert digest(modelpath)==protocol['model_sha256']
with modelpath.open('rb') as f:model=pickle.load(f)
inp=Inputs(a.key);n=len(inp.valid);K=8;dest=BASE/'paired_tree_runtime_v1'/a.key;dest.mkdir(parents=True,exist_ok=False);raw=np.lib.format.open_memmap(dest/'interval_ratios.npy',mode='w+',dtype=np.float64,shape=(n//4*K,4,2));roles=np.lib.format.open_memmap(dest/'weld_type.npy',mode='w+',dtype=np.int8,shape=(n//4*K,4));members=protocol['members']
for first in range(0,n,4096):
 last=min(first+4096,n);ids=np.arange(first,last);x=inp.features(ids)[:,:457];pred=np.stack([model['interval'].predict(x)]+[model['interval'].estimators_[i].predict(x) for i in members],axis=1);t=model['role'].predict(x);begin=first//4*K;end=last//4*K;raw[begin:end]=pred.reshape(-1,4,K,2).transpose(0,2,1,3).reshape(-1,4,2);roles[begin:end]=np.repeat(t[:,None],K,axis=1).reshape(-1,4,K).transpose(0,2,1).reshape(-1,4)
raw.flush();roles.flush();valid=np.isfinite(raw).all(2)&(raw[:,:,0]>=0)&(raw[:,:,1]<=1)&(raw[:,:,0]<raw[:,:,1]);np.save(dest/'valid.npy',valid);np.save(dest/'profile_ids.npy',np.repeat(inp.pid,K));np.save(dest/'canonical_fragments.npy',np.repeat(inp.canonical,K));raw._mmap.close();roles._mmap.close();write(dest/'receipt.json',dict(key=a.key,status='K8_FULL_INPUT_PROPOSALS_NOT_SELECTED',input_queries=n,proposals=n*K,K=K,members=members,valid_proposals=int(valid.sum()),files={p.name:digest(p) for p in dest.glob('*.npy')},model_sha256=protocol['model_sha256'],source_geometry_receipt_sha256=digest(inp.folder/'receipt.json'),GT_reads=0,STEP_reads=0,fit_calls=0,layout='fragment, hypothesis, query_fraction; original_query_id=(proposal_id//32)*4+proposal_id%4',not_extraction_accuracy=True));print(a.key,n*K,flush=True)
