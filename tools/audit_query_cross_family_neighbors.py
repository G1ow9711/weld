"""Small deterministic TRAIN-only cross-family feature locality diagnostic.

This is labeled-neighbor analysis, never a deployed retrieval predictor.
"""
import numpy as np
from threadpoolctl import threadpool_limits
from scipy.spatial.distance import cdist
from weld_pipeline.sjg_local_runtime_904 import ROOT,read,write,digest
BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'query_cross_family_neighbors_v1'

def main():
 receipt=read(BASE/'positive_query_head_v1/joint_reference_v1/receipt.json');data={};OUT.mkdir(exist_ok=False)
 write(OUT/'protocol.json',dict(training_keys=receipt['training_keys'],max_reference_rows_per_family=512,max_query_rows_per_family=128,selection='Evenly spaced row indices, label blind; finite-line filter only',metric='Euclidean standardized 457 input features; mean/std from other-family reference subset only, std floor1e-3',scope='TRAIN-only conditional feature locality probe. Not full reference pool nearest neighbor, not full inference, not proof of physical unidentifiability. No tuning on protected DEV.'))
 for key in receipt['training_keys']:
  folder=__import__('pathlib').Path(receipt['binding_paths'][key]);assert digest(folder/'receipt.json')==receipt['bindings'][key]
  x=np.load(folder/'features.npy',mmap_mode='r');z=np.load(folder/'targets.npz');records=read(folder/'records.json')
  src=(ROOT/'server_rebuild/best_baseline_restore/legacy_runtime' if key in ('21','BM79A_native_local','10','25','36') else BASE/'frozen_features/legacy_runtime')/('source_'+key)
  profiles=read(src/'run185'/('source_'+key)/'profiles.json');ids=np.array([i for i,r in enumerate(records) if profiles[r['profile']]['representation']['kind']=='line'],int)
  take=ids[np.unique(np.linspace(0,len(ids)-1,min(512,len(ids)),dtype=int))];length=np.array([np.linalg.norm(np.diff(np.asarray(profiles[records[i]['profile']]['representation']['endpoints_mm']),axis=0)) for i in take])
  data[key]=dict(x=np.array(x[take],float),y=z['interval_ratios'][take],length=length,rows=take);x._mmap.close()
 reports=[]
 for key,d in data.items():
  others=[k for k in data if k!=key];x=np.concatenate([data[k]['x'] for k in others]);y=np.concatenate([data[k]['y'] for k in others]);sources=np.concatenate([np.full(len(data[k]['x']),k) for k in others])
  mean=x.mean(0);scale=np.maximum(x.std(0),1e-3);qi=np.unique(np.linspace(0,len(d['x'])-1,min(128,len(d['x'])),dtype=int));distance=cdist((d['x'][qi]-mean)/scale,(x-mean)/scale);nearest=distance.argmin(1);error=np.max(abs(y[nearest]-d['y'][qi]),axis=1)*d['length'][qi]
  report=dict(source=key,query_rows=len(qi),reference_rows=len(x),nearest_distance_quantiles=np.quantile(distance[np.arange(len(qi)),nearest],[0,.25,.5,.75,1]).tolist(),optimistic_along_line_endpoint_error_mm_quantiles=np.quantile(error,[0,.25,.5,.75,1]).tolist(),endpoint_error_le3=int((error<=3).sum()),neighbor_sources={k:int((sources[nearest]==k).sum()) for k in others},scope='Endpoint ratios copied only for diagnosis; ignores type, parent and candidate detection.')
  reports.append(report);print(report,flush=True)
 write(OUT/'receipt.json',dict(status='COMPLETE',cases=reports,fit_calls=0,protected_DEV_reads=0))

if __name__=='__main__':
 with threadpool_limits(limits=2):main()
