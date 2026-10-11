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
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1/positive_query_head_v1';DEST=BASE/'model_query_local_distance_seven'
KEYS=('21','BM79A_native_local','10','25','36','new_ac2008a03607','new_fcaf2af1abee');SETTINGS=dict(n_estimators=96,max_depth=18,min_samples_leaf=2,max_features=.75,random_state=1031,n_jobs=2,bootstrap=False)
def main():
 DEST.mkdir(exist_ok=False);start=time.perf_counter();data={};bindings={};contexts={};write(DEST/'protocol.json',dict(status='EXPERIMENTAL_PROPOSAL_ONLY',settings=SETTINGS,keys=KEYS,target='Log1p millimetre distances from the fixed input query to each whole-child endpoint, plus H/V role; controlled target-representation comparison to model_initial_seven_families',not_existence_or_count_classifier=True,unknown_queries_not_negative=True,equal_family_weight=True,equal_GT_child_mass_within_family=True,baseline_weights_unchanged=True,conditional_validation_only=True,full_chain_OOF=False,DEV_reads=0))
 for key in KEYS:
  f=BASE/key;r=read(f/'receipt.json')
  for name,sha in r['files'].items():assert digest(f/name)==sha
  x=np.load(f/'features.npy');z=np.load(f/'targets.npz');original_y=z['interval_ratios'];records=read(f/'records.json');old=key in ('21','BM79A_native_local','10','25','36');runtime=ROOT/('server_rebuild/best_baseline_restore/legacy_runtime' if old else 'server_rebuild/expansion_v1/frozen_features/legacy_runtime');profilepath=runtime/('source_'+key)/'run185'/('source_'+key)/'profiles.json';profiles=read(profilepath);witness_root=BASE.parent/'original_train_query_inputs_v1/positive_intervals' if old else BASE.parent/'positive_intervals';wr=read(witness_root/key/'receipt.json');assert digest(profilepath)==wr['profile_sha256'];length=np.array([profiles[v['profile']]['length_mm'] for v in records]);query=np.array([v['carrier_query_ratio'] for v in records]);dist=np.column_stack((query-original_y[:,0],original_y[:,1]-query))*length[:,None];assert np.isfinite(dist).all() and dist.min()>=-1e-6;encoded=np.log1p(np.maximum(dist,0));contexts[key]=(length,query,original_y);np.savez(DEST/(key+'_query_context.npz'),carrier_length_mm=length,input_query_ratio=query);data[key]=(x,encoded,z['weld_type'],z['weights']);assert x.shape==(r['rows'],457);bindings[key]=digest(f/'receipt.json')
 def fit(keys):
  x=np.concatenate([data[k][0] for k in keys]);y=np.concatenate([data[k][1] for k in keys]);t=np.concatenate([data[k][2] for k in keys]);w=np.concatenate([data[k][3]/len(keys) for k in keys]);assert np.isclose(w.sum(),1)
  interval=ExtraTreesRegressor(**SETTINGS).fit(x,y,sample_weight=w);role=ExtraTreesClassifier(**SETTINGS).fit(x,t,sample_weight=w);interval.set_params(n_jobs=1);role.set_params(n_jobs=1);return dict(interval=interval,role=role,schema='SIDE_SYMMETRIC_FRAGMENT152_MEAN_MIN_MAX_QUERY1_V1',columns=457,training_keys=list(keys),baseline_weights_unchanged=True,proposal_only=True,interval_target_schema='QUERY_LOCAL_LOG_DISTANCE_MM_V1')
 def decode(key,p):
  length,query,_=contexts[key];distance=np.expm1(p);return np.column_stack((query-distance[:,0]/length,query+distance[:,1]/length))
 reports=[]
 for held in KEYS:
  model=fit([k for k in KEYS if k!=held]);x,y,t,w=data[held];prediction=decode(held,model['interval'].predict(x));role=model['role'].predict(x);np.savez(DEST/(held+'_heldout.npz'),interval_ratios=prediction,weld_type=role)
  with (DEST/(held+'_heldout.pkl')).open('wb') as f:pickle.dump(model,f,protocol=5)
  reports.append(dict(held=held,known_positive_query_rows=len(x),ratio_MAE=np.mean(abs(prediction-contexts[held][2]),axis=0).tolist(),conditional_role_accuracy=float(np.mean(role==t)),not_extraction_metrics=True));print(reports[-1],flush=True)
 model=fit(KEYS)
 with (DEST/'model.pkl').open('wb') as f:pickle.dump(model,f,protocol=5)
 with (DEST/'model.pkl').open('rb') as f:loaded=pickle.load(f)
 for key in KEYS:
  x=data[key][0];p=model['interval'].predict(x);t=model['role'].predict(x);assert np.array_equal(p,loaded['interval'].predict(x)) and np.array_equal(t,loaded['role'].predict(x));assert np.array_equal(p,loaded['interval'].predict(x[::-1])[::-1]);np.savez(DEST/(key+'_training.npz'),interval_ratios=decode(key,p),weld_type=t)
 write(DEST/'receipt.json',dict(status='TRAINED_EXPERIMENTAL_PROPOSAL_HEAD_NOT_PROMOTED',bindings=bindings,folds=reports,fit_calls=2*(len(KEYS)+1),model_sha256=digest(DEST/'model.pkl'),files={p.name:digest(p) for p in DEST.glob('*') if p.is_file()},source_sha256=digest(__file__),seconds=time.perf_counter()-start,DEV_reads=0,baseline_weights_unchanged=True,whole_model_extraction_accuracy_not_measured=True));print('POSITIVE_QUERY_PROPOSAL_HEAD_TRAINED',flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
