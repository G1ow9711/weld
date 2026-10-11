"""Separate fixed-reference task experiment; does not change strict supervision."""
import pickle,time
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits
from tools.query_reference_endpoint_context_runtime import Inputs,BASE
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents
KEYS=('new_cfc732e5f068','new_ceeae49d6777');DEST=BASE/'reference_utility_endpoint_context_v2'
def main():
 DEST.mkdir(exist_ok=False);start=time.perf_counter();protocol=dict(feature_change='Add frozen 152-side-symmetric input context inside each predicted endpoint, plus fraction and length of containing input fragment; no reference endpoint used', comparison='Same frozen predictions, calibration split, sampling, weights, model budget and threshold as v1; only input feature context changes', task='Fixed-reference child matching utility, NOT physical existence',physical_unmatched_status='UNKNOWN',reference_reward=dict(match=1,unmatched=-1),threshold=.5,keys=KEYS,endpoint_model='model_initial_seven_families frozen before either calibration family',upstream_shared=True,full_chain_independent_validation=False,unmatched_sample_budget_per_family=32768,unmatched_sampling='seed1031 uniform without replacement with inverse inclusion weights',family_weight='equal, within family empirical population priors',settings=dict(n_estimators=96,max_depth=18,min_samples_leaf=2,max_features=.75,random_state=1031,n_jobs=2),duplicates='exact profile/interval/type aliases only; no coordinate averaging',assembly='singleton prototype only; multi-child reference parents are evaluated unchanged, not flattened',old_output='frozen379 preserved in union comparison',no_promotion=True)
 write(DEST/'protocol.json',protocol);data={};inputs={};bindings={}
 for key in KEYS:
  inp=Inputs(key);inputs[key]=inp;r=read(inp.folder/'child_reference_capacity.json');assert digest(inp.folder/'receipt.json')==r['runtime_receipt_sha256'];pos=np.array(sorted({i for row in r['rows'] for i in row['compatible_query_ids']}),dtype=np.int64);mask=inp.valid.copy();mask[pos]=False;neg=np.flatnonzero(mask);rng=np.random.default_rng(1031);take=rng.choice(neg,size=min(len(neg),32768),replace=False);ids=np.r_[pos,take];y=np.r_[np.ones(len(pos),np.int8),np.zeros(len(take),np.int8)];w=np.r_[np.ones(len(pos)),np.full(len(take),len(neg)/len(take))]/int(inp.valid.sum());data[key]=(inp.features(ids),y,w);bindings[key]=dict(runtime_receipt=digest(inp.folder/'receipt.json'),capacity_receipt=digest(inp.folder/'child_reference_capacity.json'),positive_queries=len(pos),unmatched_queries=len(neg),training_rows=len(ids))
 reports=[]
 for held in KEYS:
  train=[k for k in KEYS if k!=held];x=np.concatenate([data[k][0] for k in train]);y=np.concatenate([data[k][1] for k in train]);w=np.concatenate([data[k][2]/len(train) for k in train]);model=ExtraTreesClassifier(**protocol['settings']).fit(x,y,sample_weight=w);model.set_params(n_jobs=1)
  with (DEST/(held+'_filter.pkl')).open('wb') as f:pickle.dump(dict(model=model,training_keys=train,protocol=protocol),f)
  inp=inputs[held];scores=np.zeros(len(inp.valid));valid=np.flatnonzero(inp.valid)
  for first in range(0,len(valid),4096):
   ids=valid[first:first+4096];scores[ids]=model.predict_proba(inp.features(ids))[:,list(model.classes_).index(1)]
  np.save(DEST/(held+'_scores.npy'),scores);ids=np.flatnonzero(inp.valid&(scores>=.5));decoded=inp.decode(ids,scores[ids]);write(DEST/(held+'_new_decoded.json'),decoded)
  oldpath=BASE/'frozen_features/legacy_runtime'/('source_'+held)/'run379'/('test_'+held)/'final_decoded.json';old=read(oldpath);union=dict(old,parents=old['parents']+decoded['parents'],total_prediction_children=old['total_prediction_children']+decoded['total_prediction_children']);write(DEST/(held+'_union_decoded.json'),union)
  # Only after actual output files are frozen do we open reference targets.
  target=read(BASE/'labels'/held/'targets.json');rows=[]
  for name,output in [('old',old),('new',decoded),('union',union)]:
   score=score_native_parents(output,target,[False]*len(target['parents']));write(DEST/(held+'_'+name+'_score.json'),score);rows.append(dict(branch=name,**score['categories']['overall']))
  report=dict(key=held,selected_queries=len(ids),exported_singletons=len(decoded['parents']),rows=rows,scope='Cross-family selector with shared frozen endpoint/upstream; complete prototype singleton output, not contract completion');reports.append(report);print(report,flush=True)
 write(DEST/'receipt.json',dict(protocol_sha256=digest(DEST/'protocol.json'),bindings=bindings,results=reports,fit_calls=2,DEV_reads=0,baseline_weights_unchanged=True,status='EXPERIMENTAL_NOT_PROMOTED',seconds=time.perf_counter()-start))
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
