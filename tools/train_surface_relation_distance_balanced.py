"""Matched material-relation control: balance known classes within distance strata."""
from pathlib import Path
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild';SOURCE=BASE/'surface_material_relation_pilot_v1';OUT=BASE/'surface_material_relation_distance_balanced_v1'
def main():
 OUT.mkdir(exist_ok=False);data={k:np.load(SOURCE/(k+'_pairs.npz')) for k in ['10','25','36']};edges=np.array([10,35,100,300,1000])/1000;reports=[]
 write(OUT/'protocol.json',dict(source_protocol_sha256=digest(SOURCE/'protocol.json'),change='Equal family, then equal nonempty distance strata, then equal available classes within each stratum. No held-family statistics used for fitting.',distance_bin_edges_mm=(edges*1000).tolist(),trees=64,max_depth=16,min_samples_leaf=2,seed=1031,threshold=.5,scope='Development comparison after prior held-family diagnostics; not independent test.',data_sha256={k:digest(SOURCE/(k+'_pairs.npz')) for k in data}))
 for held in data:
  keys=[k for k in data if k!=held];weights=[]
  for key in keys:
   x,y=data[key]['features'],data[key]['labels'];bins=np.searchsorted(edges,x[:,0],side='right');w=np.zeros(len(y));present=np.unique(bins)
   for b in present:
    classes=np.unique(y[bins==b])
    for c in classes:
     mask=(bins==b)&(y==c);w[mask]=1/(len(keys)*len(present)*len(classes)*mask.sum())
   weights.append(w)
  model=RandomForestClassifier(n_estimators=64,max_depth=16,min_samples_leaf=2,max_features='sqrt',random_state=1031,n_jobs=4);model.fit(np.concatenate([data[k]['features'] for k in keys]),np.concatenate([data[k]['labels'] for k in keys]),sample_weight=np.concatenate(weights));x,y=data[held]['features'],data[held]['labels'];pred=model.predict_proba(x)[:,list(model.classes_).index(1)]>=.5;strata=[]
  for name,mask in [('all',np.ones(len(y),bool)),('distance_lt100mm',x[:,0]<.1)]:
   tn,fp,fn,tp=confusion_matrix(y[mask],pred[mask],labels=[0,1]).ravel();strata.append(dict(scope=name,TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn)))
  path=OUT/('leave_'+held+'.pkl')
  with path.open('wb') as f:pickle.dump(model,f)
  reports.append(dict(held=held,strata=strata,model_sha256=digest(path)));print(reports[-1],flush=True)
 write(OUT/'receipt.json',dict(status='EXPERIMENTAL_NOT_DEPLOYED',scores=reports,fit_calls=3,DEV_reads=0,source_sha256=digest(__file__)))
if __name__=='__main__':main()
