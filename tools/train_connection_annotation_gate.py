"""Fixed-budget family-held-out binary annotation-compatibility diagnostic.

This is conditional edge supervision, not a full extraction metric. Upstream
models have seen these development families. No protected DEV data is accessed.
"""
import gc,pickle,time
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits
from weld_pipeline.sjg_local_runtime_904 import ROOT,read,write,digest

DATA=ROOT/'server_rebuild/connection_annotation_features_v1'
DEST=ROOT/'server_rebuild/connection_annotation_gate_v1'
KEYS=('21','10','25','36')

def score(y,p):
 q=p>=.5;positive=y>0
 tp=int(np.sum(q&positive));fp=int(np.sum(q&~positive));fn=int(np.sum(~q&positive));tn=int(np.sum(~q&~positive))
 return dict(TP=tp,FP=fp,FN=fn,TN=tn,recall=tp/max(1,tp+fn),precision=tp/max(1,tp+fp),false_positive_rate=fp/max(1,fp+tn),brier=float(np.mean((p-positive)**2)))

def main():
 assert read(DATA/'receipt.json')['status']=='ANNOTATION_FEATURES_FROZEN'
 DEST.mkdir(exist_ok=False)
 settings=dict(n_estimators=64,max_depth=16,min_samples_leaf=2,max_features='sqrt',bootstrap=False,n_jobs=2,random_state=1031)
 write(DEST/'protocol.json',dict(settings=settings,threshold=.5,folds=['joint','leave_10','leave_25','leave_36'],training_keys=list(KEYS),excluded_empty='BM79A_native_local has no known supervised rows; not a negative-only case',sample_weight='Equal total family mass, equal mass among binary classes present within each family; fixed before evaluation',data_receipt_sha256=digest(DATA/'receipt.json'),scope='Annotation compatibility conditional on known line owners/legality. Shared upstream saw families, so NOT full independent validation. No replacement of original four-way model, unknown or arcs.',model_selection=False))
 data={};audit=[]
 for key in KEYS:
  folder=DATA/key;r=read(folder/'receipt.json');n=r['training_rows']
  for name,sha in r['files'].items():assert digest(folder/name)==sha
  x=np.memmap(folder/'features.f32',dtype='<f4',mode='r',shape=(n,32));y=np.memmap(folder/'labels.i8',dtype='i1',mode='r',shape=(n,));ix=np.memmap(folder/'edge_indices.i64',dtype='<i8',mode='r',shape=(n,))
  assert np.isfinite(x).all() and np.all((y>=0)&(y<=3)) and len(np.unique(ix))==n
  original=np.load(ROOT/'server_rebuild/connection_annotation_targets_v1'/key/'labels.npy');assert np.array_equal(original[ix],y)
  counts=np.bincount(y,minlength=4);assert counts.tolist()==r['class_counts']
  oldfolder=ROOT/'server_rebuild/best_baseline_restore/legacy_runtime'/('source_'+key)/'run379'/('test_'+key)
  m=r['original_edges'];old=np.memmap(oldfolder/'probabilities.f64',dtype='<f8',mode='r',shape=(m,4));base=score(y,np.asarray(old[ix,1:].sum(1)));old._mmap.close()
  data[key]=(x,y,ix);audit.append(dict(source=key,rows=n,original_connection_known_edge_metrics=base))
 write(DEST/'data_audit.json',dict(status='PASS',cases=audit,all_targets_aligned=True,protected_DEV_reads=0))
 allreports=[]
 for fold in ('joint','leave_10','leave_25','leave_36'):
  folder=DEST/fold;folder.mkdir();hold=fold.removeprefix('leave_') if fold!='joint' else None;train=[k for k in KEYS if k!=hold]
  x=np.concatenate([data[k][0] for k in train]);y=np.concatenate([np.asarray(data[k][1]>0,dtype=np.int8) for k in train]);weights=[]
  for key in train:
   binary=data[key][1]>0;classes=np.unique(binary);w=np.empty(len(binary),float)
   for c in classes:w[binary==c]=1./(len(classes)*np.sum(binary==c))
   weights.append(w)
  w=np.concatenate(weights);w*=len(w)/w.sum();start=time.perf_counter();print('FIT',fold,x.shape,flush=True)
  with threadpool_limits(limits=2):model=ExtraTreesClassifier(**settings).fit(x,y,sample_weight=w)
  with (folder/'model.pkl').open('wb') as f:pickle.dump(model,f,protocol=5)
  del x,y,w,weights;gc.collect();results=[]
  for key in KEYS:
   x,y,ix=data[key];p=np.empty(len(y),np.float32)
   for first in range(0,len(y),65536):p[first:first+65536]=model.predict_proba(x[first:first+65536])[:,list(model.classes_).index(1)]
   np.save(folder/(key+'_probability.npy'),p)
   results.append(dict(source=key,role='held_family' if key==hold else 'fit_family',**score(y,p)))
  report=dict(fold=fold,train_keys=train,seconds=time.perf_counter()-start,results=results,model_sha256=digest(folder/'model.pkl'),promoted=False)
  write(folder/'receipt.json',report);allreports.append(report);print('COMPLETE',report,flush=True);del model;gc.collect()
 write(DEST/'receipt.json',dict(status='FIXED_FOLD_DIAGNOSTIC_COMPLETE',folds=allreports,fit_calls=4,promoted=False,full_extraction_evaluated=False))

if __name__=='__main__':main()
