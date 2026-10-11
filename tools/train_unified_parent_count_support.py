"""TRAIN-only masked D3 utility regression; no extraction promotion."""
from pathlib import Path
import pickle,time
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/unified_parent_scoring_v1';DEST=BASE/'count_head_support_v2'
SETTINGS=dict(n_estimators=96,max_depth=18,min_samples_leaf=2,max_features=.75,bootstrap=False,random_state=1031,n_jobs=2)
def main():
 DEST.mkdir(exist_ok=False);data={};bindings={};start=time.perf_counter()
 write(DEST/'protocol.json',dict(settings=SETTINGS,target='D3 TP / child_count, FP / child_count',utility='expected_TP - expected_FP',lambda_FP=1,unknown_geometry_masked=True,known_finite_annotated_support_included=True,equal_case_weight=True,equal_positive_nonpositive_utility_mass=True,full_chain_OOF=False,baseline_unchanged=True,DEV_reads=0))
 for key in ['21','BM79A_native_local','10','25','36']:
  f=BASE/('source_'+key);r=read(f/'receipt.json');s=read(f/'d3_supervision_support_v2.json')
  assert s['features_sha256']==digest(f/'features.npy')==r['files']['features.npy']
  assert s['parents_sha256']==digest(f/'parents.json')==r['files']['parents.json']
  x=np.load(f/'features.npy');rows=s['rows'];n=np.array([a['children'] for a in rows]);y=np.array([[a['D3_TP'],a['D3_FP']] for a in rows])/n[:,None];mask=np.array([a['parent_loss_valid'] for a in rows]);assert x.shape==(len(rows),1458) and np.isfinite(x).all()
  data[key]=(x,y,n,mask);bindings[key]=dict(receipt_sha256=digest(f/'receipt.json'),supervision_sha256=digest(f/'d3_supervision_support_v2.json'))
 def fit(keys):
  xs=[];ys=[];ws=[]
  for k in keys:
   x,y,n,mask=data[k];cls=y[:,0]>y[:,1];w=np.zeros(len(x));groups=np.unique(cls[mask])
   for c in groups:w[mask&(cls==c)]=1/(len(keys)*len(groups)*sum(mask&(cls==c)))
   xs.append(x[mask]);ys.append(y[mask]);ws.append(w[mask])
  return ExtraTreesRegressor(**SETTINGS).fit(np.concatenate(xs),np.concatenate(ys),sample_weight=np.concatenate(ws))
 folds=[]
 for held,(x,y,n,mask) in data.items():
  model=fit([k for k in data if k!=held]);pred=model.predict(x)*n[:,None];np.save(DEST/(held+'_heldout_counts.npy'),pred)
  row=dict(held_out=held,conditional_head_only=True,known_rows=int(mask.sum()),count_MAE=np.mean(abs(pred[mask]-y[mask]*n[mask,None]),axis=0).tolist(),unknown_rows=int((~mask).sum()),unknown_positive_utility=int(((pred[:,0]>pred[:,1])&~mask).sum()));folds.append(row);print(row,flush=True)
 model=fit(list(data));model.set_params(n_jobs=1)
 with (DEST/'model.pkl').open('wb') as f:pickle.dump(model,f,protocol=5)
 with (DEST/'model.pkl').open('rb') as f:loaded=pickle.load(f)
 for key,(x,y,n,mask) in data.items():
  p=model.predict(x);assert np.array_equal(p,loaded.predict(x));assert np.array_equal(p,loaded.predict(x[::-1])[::-1]);np.save(DEST/(key+'_training_counts.npy'),p*n[:,None])
 write(DEST/'training_receipt.json',dict(status='TRAINED_NOT_PROMOTED',bindings=bindings,folds=folds,fit_calls=6,model_sha256=digest(DEST/'model.pkl'),protocol_sha256=digest(DEST/'protocol.json'),seconds=time.perf_counter()-start,DEV_reads=0,not_an_extraction_result=True,baseline_unchanged=True,full_chain_OOF=False))
 print('COUNT_HEAD_COMPLETE',flush=True)
if __name__=='__main__':main()
