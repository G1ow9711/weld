"""Masked parent-structure shadow head on frozen original features.

No baseline replacement or candidate output change. Leave-one-case results
describe only this head: supervised upstream models already saw these cases.
"""
from pathlib import Path
import json,pickle,time
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from weld_pipeline.sjg_local_runtime_904 import read,write,digest

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/parent_residual_v1';DEST=BASE/'shadow_head_v1'
SETTINGS=dict(n_estimators=64,max_depth=16,min_samples_leaf=2,max_features=.75,bootstrap=False,random_state=1031,n_jobs=2)

def main():
 DEST.mkdir(exist_ok=False);manifest=BASE/'inputs_frozen.json';inputs=read(manifest);datasets={};bindings={}
 write(DEST/'protocol.json',dict(settings=SETTINGS,threshold=.5,input_schema=inputs['schema'],positive='Complete original parent compatibility under one whole correspondence',negative='No complete parent match, but EVERY candidate child has a geometry-compatible annotated child; annotated structure violation only',unknown='Any unannotated child geometry remains masked',baseline_unchanged=True,DEV_reads=0,full_chain_OOF=False,selection_or_acceptance_not_performed=True))
 for case in inputs['cases']:
  key=case['source'];assert key!='64' and case['role']=='TRAIN'
  for f in case['files'].values():assert digest(ROOT/f['path'])==f['sha256']
  folder=BASE/('source_'+key);compat=read(folder/'compatibility.json');ownership=read(folder/'child_ownership.json')
  assert compat['input_manifest_sha256']==ownership['input_manifest_sha256']==digest(manifest)
  assert compat['target_sha256']==ownership['target_sha256']
  assert ownership['prediction_sha256']==case['files']['parents']['sha256']
  by={}
  for r in ownership['rows']:by.setdefault(r['candidate_parent'],[]).append(bool(r['owners']))
  y=np.array([1 if edges else (0 if all(by[i]) else -1) for i,edges in enumerate(compat['compatible_targets'])],np.int8)
  x=np.load(ROOT/case['files']['parent_features']['path']);assert x.shape==(len(y),160) and np.isfinite(x).all()
  datasets[key]=(x,y);np.save(DEST/(key+'_labels.npy'),y)
  bindings[key]=dict(features_sha256=case['files']['parent_features']['sha256'],compatibility_sha256=digest(folder/'compatibility.json'),ownership_sha256=digest(folder/'child_ownership.json'),labels_sha256=digest(DEST/(key+'_labels.npy')),positive=int((y==1).sum()),negative=int((y==0).sum()),unknown=int((y<0).sum()))
 def fit(keys):
  xs=[];ys=[];ws=[]
  for key in keys:
   x,y=datasets[key];mask=y>=0;classes=np.unique(y[mask]);weight=np.zeros(len(y))
   for c in classes:weight[y==c]=1/(len(keys)*len(classes)*int((y==c).sum()))
   xs.append(x[mask]);ys.append(y[mask]);ws.append(weight[mask])
  x=np.concatenate(xs);y=np.concatenate(ys);assert set(y.tolist())=={0,1}
  model=ExtraTreesClassifier(**SETTINGS).fit(x,y,sample_weight=np.concatenate(ws));model.set_params(n_jobs=1);return model
 start=time.perf_counter();folds=[];keys=list(datasets)
 for held in keys:
  model=fit([k for k in keys if k!=held]);x,y=datasets[held];prob=model.predict_proba(x)[:,1];known=y>=0;prediction=prob>.5
  np.save(DEST/(held+'_heldout_probability.npy'),prob)
  row=dict(held_out=held,conditional_head_only=True,positive=int((y==1).sum()),negative=int((y==0).sum()),TP=int(((y==1)&prediction).sum()),FP=int(((y==0)&prediction).sum()),FN=int(((y==1)&~prediction).sum()),unknown=int((y<0).sum()),unknown_predicted_positive=int(((y<0)&prediction).sum()));folds.append(row);print(row,flush=True)
 model=fit(keys)
 with (DEST/'model.pkl').open('wb') as f:pickle.dump(model,f,protocol=5)
 with (DEST/'model.pkl').open('rb') as f:loaded=pickle.load(f)
 for key,(x,y) in datasets.items():
  p=model.predict_proba(x);assert np.array_equal(p,loaded.predict_proba(x));assert np.array_equal(p,loaded.predict_proba(x[::-1])[::-1]);np.save(DEST/(key+'_training_probability.npy'),p[:,1])
 write(DEST/'training_receipt.json',dict(status='SHADOW_HEAD_TRAINED_NOT_PROMOTED',bindings=bindings,folds=folds,fit_calls=6,model_sha256=digest(DEST/'model.pkl'),protocol_sha256=digest(DEST/'protocol.json'),input_manifest_sha256=digest(manifest),TRAIN=keys,DEV_reads=0,old_weights_unchanged=True,full_chain_OOF=False,not_an_extraction_result=True,seconds=time.perf_counter()-start))
 print('SHADOW_HEAD_COMPLETE',flush=True)
if __name__=='__main__':main()
