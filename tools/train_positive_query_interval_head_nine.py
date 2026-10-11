"""Fit a positive-only interval proposal head and conditional family-heldout probes.

This cannot classify absence or count welds. Every runtime proposal must later
pass the separately trained parent selection and finite contact checks.
"""
import pickle,time
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier
from threadpoolctl import threadpool_limits
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1/positive_query_head_v1';DEST=BASE/'model_initial_nine_families'
KEYS=('21','BM79A_native_local','10','25','36','new_ac2008a03607','new_fcaf2af1abee','new_cfc732e5f068','new_ceeae49d6777');SETTINGS=dict(n_estimators=96,max_depth=18,min_samples_leaf=2,max_features=.75,random_state=1031,n_jobs=2,bootstrap=False)
def main():
 DEST.mkdir(exist_ok=False);start=time.perf_counter();data={};bindings={};write(DEST/'protocol.json',dict(status='EXPERIMENTAL_PROPOSAL_ONLY',settings=SETTINGS,keys=KEYS,target='Whole child carrier interval ratios plus H/V role, conditioned on positive fixed-grid query',not_existence_or_count_classifier=True,unknown_queries_not_negative=True,equal_family_weight=True,equal_GT_child_mass_within_family=True,baseline_weights_unchanged=True,conditional_validation_only=True,full_chain_OOF=False,DEV_reads=0))
 for key in KEYS:
  f=BASE/key;r=read(f/'receipt.json')
  for name,sha in r['files'].items():assert digest(f/name)==sha
  x=np.load(f/'features.npy');z=np.load(f/'targets.npz');data[key]=(x,z['interval_ratios'],z['weld_type'],z['weights']);assert x.shape==(r['rows'],457);bindings[key]=digest(f/'receipt.json')
 def fit(keys):
  x=np.concatenate([data[k][0] for k in keys]);y=np.concatenate([data[k][1] for k in keys]);t=np.concatenate([data[k][2] for k in keys]);w=np.concatenate([data[k][3]/len(keys) for k in keys]);assert np.isclose(w.sum(),1)
  interval=ExtraTreesRegressor(**SETTINGS).fit(x,y,sample_weight=w);role=ExtraTreesClassifier(**SETTINGS).fit(x,t,sample_weight=w);interval.set_params(n_jobs=1);role.set_params(n_jobs=1);return dict(interval=interval,role=role,schema='SIDE_SYMMETRIC_FRAGMENT152_MEAN_MIN_MAX_QUERY1_V1',columns=457,training_keys=list(keys),baseline_weights_unchanged=True,proposal_only=True)
 reports=[]
 for held in KEYS:
  model=fit([k for k in KEYS if k!=held]);x,y,t,w=data[held];prediction=model['interval'].predict(x);role=model['role'].predict(x);np.savez(DEST/(held+'_heldout.npz'),interval_ratios=prediction,weld_type=role)
  with (DEST/(held+'_heldout.pkl')).open('wb') as f:pickle.dump(model,f,protocol=5)
  reports.append(dict(held=held,known_positive_query_rows=len(x),ratio_MAE=np.mean(abs(prediction-y),axis=0).tolist(),conditional_role_accuracy=float(np.mean(role==t)),not_extraction_metrics=True));print(reports[-1],flush=True)
 model=fit(KEYS)
 with (DEST/'model.pkl').open('wb') as f:pickle.dump(model,f,protocol=5)
 with (DEST/'model.pkl').open('rb') as f:loaded=pickle.load(f)
 for key in KEYS:
  x=data[key][0];p=model['interval'].predict(x);t=model['role'].predict(x);assert np.array_equal(p,loaded['interval'].predict(x)) and np.array_equal(t,loaded['role'].predict(x));assert np.array_equal(p,loaded['interval'].predict(x[::-1])[::-1]);np.savez(DEST/(key+'_training.npz'),interval_ratios=p,weld_type=t)
 write(DEST/'receipt.json',dict(status='TRAINED_EXPERIMENTAL_PROPOSAL_HEAD_NOT_PROMOTED',bindings=bindings,folds=reports,fit_calls=2*(len(KEYS)+1),model_sha256=digest(DEST/'model.pkl'),files={p.name:digest(p) for p in DEST.glob('*') if p.is_file()},source_sha256=digest(__file__),seconds=time.perf_counter()-start,DEV_reads=0,baseline_weights_unchanged=True,whole_model_extraction_accuracy_not_measured=True));print('POSITIVE_QUERY_PROPOSAL_HEAD_TRAINED',flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
