"""Reference-child quality classifier with family-held quality-layer diagnostics."""
from pathlib import Path
import pickle,time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1';OUT=BASE/'endpoint_reference_quality_model_v1'
def main():
 OUT.mkdir(exist_ok=False);started=time.perf_counter();labels=BASE/'endpoint_quality_training_labels_v1';assert read(labels/'receipt.json')['status']=='SIX_TRAIN_REFERENCE_QUALITY_LABELS_COMPLETE';keys=read(labels/'protocol.json')['training_keys'];parameters=dict(learning_rate=.05,max_iter=200,max_leaf_nodes=15,min_samples_leaf=25,l2_regularization=1.,early_stopping=False,random_state=1031);write(OUT/'protocol.json',dict(training_keys=keys,parameters=parameters,threshold=.5,threshold_not_tuned=True,target='Reference child geometric quality, not physical absence or unique instance validity',quality_layer_leave_one_family_out=True,full_chain_OOF=False,upstream_saw_these_TRAIN_families=True,heldout_label_reads=0,DEV_reads=0,no_duplicate_suppression=True,weights='Equal family/class; positive covered-child mass. Rescale concatenated weights to mean1 for regularized tree objective'))
 data={};bindings={}
 for key in keys:
  f=BASE/'endpoint_candidate_quality_features_v1'/key;r=read(labels/key/'receipt.json');assert digest(f/'receipt.json')==r['feature_receipt_sha256'];fr=read(f/'receipt.json');assert digest(f/'features.npy')==fr['features_sha256'];assert digest(labels/key/'labels.npz')==r['labels_sha256'];z=np.load(labels/key/'labels.npz');x=np.load(f/'features.npy');assert len(x)==len(z['quality']);data[key]=(x,z['quality'],z['weights']);bindings[key]=dict(features=digest(f/'receipt.json'),labels=digest(labels/key/'receipt.json'))
 def fit(families):
  x=np.concatenate([data[k][0] for k in families]);y=np.concatenate([data[k][1] for k in families]);w=np.concatenate([data[k][2]/len(families) for k in families]);w*=len(w)/w.sum();return HistGradientBoostingClassifier(**parameters).fit(x,y,sample_weight=w)
 def metrics(y,score):
  selected=score>=.5;tp=int((selected&(y==1)).sum());fp=int((selected&(y==0)).sum());fn=int((~selected&(y==1)).sum());return dict(reference_quality_TP=tp,reference_quality_FP=fp,reference_quality_FN=fn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),average_precision=float(average_precision_score(y,score)),selected=int(selected.sum()))
 folds=[]
 for key in keys:
  families=[k for k in keys if k!=key];model=fit(families);score=model.predict_proba(data[key][0])[:,1];np.save(OUT/(key+'_quality_layer_held.npy'),score);report=dict(key=key,**metrics(data[key][1],score),scope='Only quality-layer family exclusion; upstream trained on this family, not independent full-chain validation');folds.append(report);print(report,flush=True)
 model=fit(keys)
 with (OUT/'model.pkl').open('wb') as f:pickle.dump(model,f,pickle.HIGHEST_PROTOCOL)
 training=[]
 for key in keys:
  score=model.predict_proba(data[key][0])[:,1];np.save(OUT/(key+'_training.npy'),score);training.append(dict(key=key,**metrics(data[key][1],score)))
 write(OUT/'receipt.json',dict(status='EXPERIMENTAL_REFERENCE_QUALITY_MODEL_NOT_PROMOTED',bindings=bindings,model_sha256=digest(OUT/'model.pkl'),folds=folds,training=training,fit_calls=len(keys)+1,heldout_label_reads=0,DEV_reads=0,feature_module_sha256=digest('weld_pipeline/server_endpoint_candidate_quality_features.py'),source_sha256=digest(__file__),seconds=time.perf_counter()-started))
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
