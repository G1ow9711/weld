"""TRAIN-only pair competition classifier; full compatible alternatives compete.

Unannotated or structurally incomplete candidates are not assigned negative
competition labels. Validation is conditional on already trained upstream heads.
"""
from pathlib import Path
import pickle,time
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/unified_parent_scoring_v1';DEST=BASE/'competition_head_v4'
SETTINGS=dict(n_estimators=96,max_depth=16,min_samples_leaf=2,max_features=1.,random_state=1031,n_jobs=2)
def main():
 start=time.perf_counter();DEST.mkdir(exist_ok=False);data={};bindings={}
 write(DEST/'protocol.json',dict(settings=SETTINGS,threshold=.5,positive='Both fully compatible candidates have the same unique annotated parent',negative='Both fully compatible, but compatible annotated parent sets disjoint',unknown='Either candidate lacks complete compatibility, has no target, or has ambiguous multiple targets',equal_case_class_weight=True,DEV_reads=0,full_chain_OOF=False,learned_competition_not_material_identity=True,baseline_unchanged=True))
 for key in ['21','BM79A_native_local','10','25','36']:
  f=BASE/('source_'+key);d=f/'competition_v4';r=read(d/'receipt.json');assert r['schema']=='PARENT_COMPETITION29_V4'
  control=read(BASE/('competition_geometry_v4_'+key+'.json'));assert control['all_pass']
  for name,sha in r['files'].items():assert digest(d/name)==sha
  supervision=read(f/'d3_supervision.json');assert supervision['parents_sha256']==r['parents_sha256'];rows=supervision['rows'];pairs=np.load(d/'pairs.npy');x=np.load(d/'features.npy');owners=[set(row.get('compatible_best_parents',[])) if row['parent_loss_valid'] and row['D3_TP']==row['children'] else set() for row in rows]
  y=np.array([int(bool(owners[a]&owners[b])) if len(owners[a])==len(owners[b])==1 else -1 for a,b in pairs],np.int8);data[key]=(x,y);np.save(DEST/(key+'_labels.npy'),y);bindings[key]=dict(control_sha256=digest(BASE/('competition_geometry_v4_'+key+'.json')),feature_receipt_sha256=digest(d/'receipt.json'),supervision_sha256=digest(f/'d3_supervision.json'),positive=int((y==1).sum()),negative=int((y==0).sum()),unknown=int((y<0).sum()))
  print('LABELS',key,bindings[key],flush=True)
 def fit(keys):
  xs=[];ys=[];ws=[]
  for k in keys:
   x,y=data[k];mask=y>=0;classes=np.unique(y[mask]);w=np.zeros(len(y))
   for c in classes:w[y==c]=1/(len(keys)*len(classes)*sum(y==c))
   xs.append(x[mask]);ys.append(y[mask]);ws.append(w[mask])
  return ExtraTreesClassifier(**SETTINGS).fit(np.concatenate(xs),np.concatenate(ys),sample_weight=np.concatenate(ws))
 folds=[]
 for held,(x,y) in data.items():
  model=fit([k for k in data if k!=held]);model.set_params(n_jobs=1)
  with (DEST/(held+'_heldout_model.pkl')).open('wb') as stream:pickle.dump(model,stream,protocol=5)
  p=model.predict_proba(x)[:,list(model.classes_).index(1)];np.save(DEST/(held+'_heldout_probability.npy'),p);pred=p>.5
  row=dict(held_out=held,TP=int(((y==1)&pred).sum()),FN=int(((y==1)&~pred).sum()),FP=int(((y==0)&pred).sum()),TN=int(((y==0)&~pred).sum()),unknown=int((y<0).sum()),conditional_head_only=True);folds.append(row);print(row,flush=True)
 model=fit(list(data));model.set_params(n_jobs=1)
 with (DEST/'model.pkl').open('wb') as f:pickle.dump(model,f,protocol=5)
 with (DEST/'model.pkl').open('rb') as f:loaded=pickle.load(f)
 for key,(x,y) in data.items():
  p=model.predict_proba(x);assert np.array_equal(p,loaded.predict_proba(x));assert np.array_equal(p,loaded.predict_proba(x[::-1])[::-1]);np.save(DEST/(key+'_training_probability.npy'),p[:,list(model.classes_).index(1)])
 write(DEST/'training_receipt.json',dict(status='TRAINED_NOT_PROMOTED',bindings=bindings,folds=folds,fit_calls=6,model_sha256=digest(DEST/'model.pkl'),heldout_model_sha256={k:digest(DEST/(k+'_heldout_model.pkl')) for k in data},source_sha256=digest(__file__),protocol_sha256=digest(DEST/'protocol.json'),seconds=time.perf_counter()-start,DEV_reads=0,not_an_extraction_result=True,baseline_unchanged=True));print('COMPETITION_HEAD_COMPLETE',flush=True)
if __name__=='__main__':main()
