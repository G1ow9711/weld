"""Input-only whole-parent inference with a learned candidate competition head."""
import argparse,collections,pickle,time,shutil
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/unified_parent_scoring_v1'
def run(key,destination_name='set_competition_support_v3'):
 start=time.perf_counter();f=BASE/('source_'+key);dest=BASE/destination_name/('source_'+key);dest.mkdir(parents=True,exist_ok=False)
 r=read(f/'receipt.json');resources=read(f/'resource_constraints.json');modelpath=BASE/'count_head_support_v2/model.pkl';mr=read(modelpath.parent/'training_receipt.json')
 assert digest(modelpath)==mr['model_sha256'];assert digest(f/'receipt.json')==resources['input_receipt_sha256'];assert digest(f/'features.npy')==r['files']['features.npy'];assert digest(f/'parents.json')==r['files']['parents.json']==resources['parents_sha256']
 with modelpath.open('rb') as stream:model=pickle.load(stream)
 pool=read(f/'parents.json');parents=pool['parents'];x=np.load(f/'features.npy');counts=np.array([p['child_count'] for p in parents]);pred=model.predict(x)*counts[:,None];utility=pred[:,0]-pred[:,1];np.save(dest/'predicted_counts.npy',pred)
 default=resources['default_parent_indices'];baseline=BASE/'frozen_defaults'/(key+'.json');assert digest(baseline)==r['baseline_sha256'];reason=None
 chosen=default;solver={}
 if not resources['default_feasible']:reason='DEFAULT_RESOURCE_INFEASIBLE_REQUIRES_AUDIT'
 else:
  byresource=collections.defaultdict(list)
  for p,rs in enumerate(resources['parent_resources']):
   for res in rs:byresource[res].append(p)
  groups=[ps for ps in byresource.values() if len(ps)>1]
  pairfolder=f/'competition_v2';pairreceipt=read(pairfolder/'receipt.json');pairmodel=BASE/'competition_head_v2/model.pkl';pairtraining=read(pairmodel.parent/'training_receipt.json')
  assert pairreceipt['parents_sha256']==digest(f/'parents.json')
  for name,sha in pairreceipt['files'].items():assert digest(pairfolder/name)==sha
  assert digest(pairmodel)==pairtraining['model_sha256']
  with pairmodel.open('rb') as stream:competition=pickle.load(stream)
  pairs=np.load(pairfolder/'pairs.npy');prob=competition.predict_proba(np.load(pairfolder/'features.npy'))[:,list(competition.classes_).index(1)]
  np.save(dest/'competition_probability.npy',prob);learned=pairs[prob>.5];groups.extend(learned.tolist());rr=[];cc=[]
  write(dest/'competition_receipt.json',dict(model_sha256=digest(pairmodel),pair_receipt_sha256=digest(pairfolder/'receipt.json'),threshold=.5,learned_conflicts=len(learned),default_learned_conflicts=sum(int(a) in default and int(b) in default for a,b in learned),learned_competition_not_material_identity=True,GT_reads=0))
  for gi,ps in enumerate(groups):rr.extend([gi]*len(ps));cc.extend(ps)
  matrix=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(groups),len(parents))).tocsc();matrix.indices=matrix.indices.astype(np.int32);matrix.indptr=matrix.indptr.astype(np.int32)
  upper=np.ones(len(parents));upper[resources['internally_conflicting_parents']]=0
  # Small declared default preference is a numerical tie breaker, not an exact lexicographic proof.
  objective=utility.copy();objective[default]+=1e-9
  result=milp(-objective,integrality=np.ones(len(parents)),bounds=Bounds(0,upper),constraints=LinearConstraint(matrix,-np.inf,1),options={'time_limit':60.,'mip_rel_gap':0.})
  solver=dict(status=int(result.status),message=result.message)
  if result.x is None:reason='NO_FEASIBLE_INCUMBENT'
  else:
   candidate=np.flatnonzero(result.x>.5);check=np.zeros(len(parents));check[candidate]=1
   assert not len(groups) or np.max(matrix@check)<=1
   assert not set(candidate)&set(resources['internally_conflicting_parents'])
   if utility[candidate].sum()+1e-8<utility[default].sum():reason='DEFAULT_HAS_HIGHER_PREDICTED_UTILITY'
   else:chosen=candidate.tolist()
 if chosen==default:
  shutil.copyfile(baseline,dest/'prediction.json');assert digest(dest/'prediction.json')==digest(baseline)
 else:
  ps=[parents[i] for i in chosen];write(dest/'prediction.json',dict(parents=ps,invalid_predictions=pool['invalid_predictions'],total_prediction_children=sum(p['child_count'] for p in ps)+sum(p['counted_false_children'] for p in pool['invalid_predictions'])))
 report=dict(source=key,selected_parent_indices=chosen,default_parent_indices=default,removed_default_parent_indices=sorted(set(default)-set(chosen)),fallback_reason=reason,solver=solver,predicted_utility=float(utility[chosen].sum()),default_predicted_utility=float(utility[default].sum()),model_sha256=digest(modelpath),features_sha256=digest(f/'features.npy'),resources_sha256=digest(f/'resource_constraints.json'),baseline_sha256=digest(baseline),prediction_sha256=digest(dest/'prediction.json'),source_sha256=digest(__file__),GT_reads=0,DEV_reads=0,parent_geometry_unchanged=True,no_GT_mask_at_inference=True,not_promoted=True,seconds=time.perf_counter()-start)
 write(dest/'receipt.json',report);print(dict(source=key,selected=len(chosen),fallback=reason,utility=report['predicted_utility'],seconds=report['seconds']),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--key',choices=['21','BM79A_native_local','10','25','36']);a=ap.parse_args()
 for k in ([a.key] if a.key else ['21','BM79A_native_local','10','25','36']):run(k)
