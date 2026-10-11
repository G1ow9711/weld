"""Offline TRAIN/held-feature and endpoint-error audit; no fit or threshold choice."""
from pathlib import Path
import json
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
MODEL=BASE/'positive_query_head_v1/neural_expanded_ten_v1'
def main():
 receipt=read(MODEL/'receipt.json');state=torch.load(MODEL/'model.pt',map_location='cpu',weights_only=False);assert digest(MODEL/'model.pt')==receipt['model_sha256'];rows=[]
 for key,path in receipt['binding_paths'].items():
  folder=Path(path);assert digest(folder/'receipt.json')==receipt['bindings'][key];x=np.load(folder/'features.npy').astype(np.float64);records=read(folder/'records.json');z=np.load(folder/'targets.npz');split='training' if key in receipt['training_keys'] else 'heldout';pred=np.load(MODEL/(key+'_'+split+'.npz'));p=pred['interval_ratios'].astype(np.float64);y=z['interval_ratios'];standard=(x-state['mean'])/state['std'];clip=abs(standard)>20
  original=key in ('21','BM79A_native_local','10','25','36');src=(BASE.parents[1]/'server_rebuild/best_baseline_restore/legacy_runtime' if original else BASE/'frozen_features/legacy_runtime')/('source_'+key)
  binding=next(v for v in read(src/'run285/all_geometry_frozen_before_labels.json')['cases'] if v['source']==key)
  with np.load(src/'run285'/binding['geometry_file']) as g:
   offsets=g['offsets'];spans=g['spans'];length=np.array([spans[offsets[r['profile']+1]-1,1] for r in records])
  errors=np.max(abs(p-y)*length[:,None],axis=1);owners={}
  for i,r in enumerate(records):
   for owner in r['owners']:owners.setdefault(tuple(owner),[]).append(i)
  best=np.array([min(errors[ids]) for ids in owners.values()]);saturated=(p[:,0]<=1e-6)|(p[:,1]>=1-1e-6)
  columns=np.flatnonzero(clip.mean(0)>.01)
  rows.append(dict(key=key,split=split,queries=len(x),rows_with_any_normalization_clip=int(clip.any(1).sum()),clipped_value_fraction=float(clip.mean()),frequently_clipped_columns=[dict(column=int(c),fraction=float(clip[:,c].mean())) for c in columns],query_endpoint_max_error_mm_quantiles=np.quantile(errors,[0,.25,.5,.75,.9,.99,1]).tolist(),best_query_per_child_endpoint_error_mm_quantiles=np.quantile(best,[0,.25,.5,.75,.9,.99,1]).tolist(),children_best_error_at_most_3mm=int((best<=3).sum()),represented_children=len(best),saturated_queries=int(saturated.sum()),saturated_queries_with_exact_boundary_target=int((saturated&((y[:,0]<=1e-6)|(y[:,1]>=1-1e-6))).sum())))
 out=MODEL/'feature_shift_audit.json';assert not out.exists();write(out,dict(cases=rows,model_sha256=receipt['model_sha256'],source_sha256=digest(__file__),fit_calls=0,DEV_reads=0,scope='Positive-query endpoint projection errors, ignores type and full D3; no new acceptance score; normalization diagnostics only'))
 for r in rows:print({k:v for k,v in r.items() if k!='frequently_clipped_columns'},flush=True)
if __name__=='__main__':main()
