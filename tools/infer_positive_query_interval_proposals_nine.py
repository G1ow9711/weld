"""Run the experimental interval head on every fixed input query, with zero GT reads.

Outputs compact candidate intervals only, never final welds or presence scores.
"""
import argparse,pickle,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.sjg_local_runtime_904 import read,write,digest,check_resources
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';MODEL=BASE/'positive_query_head_v1/model_initial_nine_families';GRID=np.array([.125,.375,.625,.875])
def run(key):
 assert key in {c['key'] for c in read(BASE/'coverage_cohort_protocol.json')['cases']};mr=read(MODEL/'receipt.json');assert digest(MODEL/'model.pkl')==mr['model_sha256']
 with (MODEL/'model.pkl').open('rb') as f:model=pickle.load(f)
 assert model['proposal_only'] and model['columns']==457;assert model.get('interval_target_schema','CARRIER_RATIO_V1')=='CARRIER_RATIO_V1','Encoded-distance model requires its explicit decoder';source=BASE/'frozen_features/legacy_runtime'/('source_'+key);feature=source/'run329'/('source_'+key);fr=read(feature/'receipt.json');assert digest(feature/'features.npy')==fr['features_sha256'];x=np.load(feature/'features.npy',mmap_mode='r')
 binding=next(r for r in read(source/'run285/all_geometry_frozen_before_labels.json')['cases'] if r['source']==key);geometry=source/'run285'/binding['geometry_file'];assert digest(geometry)==binding['geometry_sha256']
 with np.load(geometry) as z:offsets=z['offsets'];spans=z['spans']
 alias=source/'run288'/('source_'+key+'_alias_map.npz');assert digest(alias)==fr['alias_sha256']
 with np.load(alias) as z:canonical=z['canonical_fragments']
 n=len(canonical);assert x.shape==(n*2,152);pid=np.searchsorted(offsets,canonical,side='right')-1;length=spans[offsets[pid+1]-1,1];dest=BASE/'positive_query_runtime_v3'/key;dest.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
 bounds=np.lib.format.open_memmap(dest/'interval_ratios.npy',mode='w+',dtype=np.float64,shape=(n,4,2));roles=np.lib.format.open_memmap(dest/'weld_type.npy',mode='w+',dtype=np.int8,shape=(n,4));valid=np.lib.format.open_memmap(dest/'valid.npy',mode='w+',dtype=np.bool_,shape=(n,4));np.save(dest/'profile_ids.npy',pid);np.save(dest/'canonical_fragments.npy',canonical)
 for first in range(0,n,2048):
  last=min(first+2048,n);xx=np.asarray(x[2*first:2*last],np.float64).reshape(-1,2,152);symmetric=np.concatenate([xx.mean(1),xx.min(1),xx.max(1)],axis=1);s=spans[canonical[first:last],:2];q=(s[:,0,None]+GRID*(s[:,1]-s[:,0])[:,None])/length[first:last,None];features=np.column_stack((np.repeat(symmetric,4,axis=0),q.ravel())).astype(np.float32)
  p=model['interval'].predict(features).reshape(-1,4,2);t=model['role'].predict(features).reshape(-1,4);bounds[first:last]=p;roles[first:last]=t;valid[first:last]=np.isfinite(p).all(2)&(p[:,:,0]>=0)&(p[:,:,1]<=1)&(p[:,:,0]<p[:,:,1]);check_resources()
  if first%16384==0:print('INPUT_QUERY_PROPOSALS',key,last,'/',n,flush=True)
 bounds.flush();roles.flush();valid.flush();valid_count=int(valid.sum());bounds._mmap.close();roles._mmap.close();valid._mmap.close();x._mmap.close()
 write(dest/'receipt.json',dict(status='ALL_FIXED_INPUT_QUERY_PROPOSALS_FROZEN_NOT_SELECTED',key=key,fragments=n,queries=4*n,valid_queries=valid_count,invalid_queries=4*n-valid_count,grid=GRID.tolist(),model_sha256=mr['model_sha256'],feature_sha256=fr['features_sha256'],geometry_sha256=digest(geometry),alias_sha256=digest(alias),files={p.name:digest(p) for p in dest.glob('*.npy')},GT_reads=0,STEP_reads=0,fit_calls=0,no_presence_or_count_claim=True,not_final_weld_output=True,source_sha256=digest(__file__),seconds=time.perf_counter()-start));print('INPUT_QUERY_PROPOSALS_COMPLETE',key,4*n,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
 with threadpool_limits(limits=2):run(a.key)
