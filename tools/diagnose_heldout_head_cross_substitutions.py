"""Offline partial-known cross substitutions, NEVER deployable inference.

Quality intervention corrects only fully compatible positive parent candidates.
Competition intervention uses original singleton known labels; unknown pairs keep
heldout predictions. This is not a complete oracle or independent evaluation.
"""
from pathlib import Path
import collections,time
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.server_parent_set_optimization import parent_keys,solve
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/unified_parent_scoring_v1';DEST=BASE/'heldout_partial_cross_diagnosis_v1'
def main():
 DEST.mkdir(exist_ok=False);write(DEST/'protocol.json',dict(offline_GT_diagnostic_only=True,no_fit=True,not_a_model_score=True,not_a_complete_oracle=True,quality_scope='Only original-supervision fully compatible positive parents',competition_scope='Original fully-compatible singleton target pairs; all other predictions retained',unknown_not_set_negative=True,solver_module_sha256=digest(ROOT/'weld_pipeline/server_parent_set_optimization.py'),source_sha256=digest(__file__)));summary=[]
 for key in ['21','BM79A_native_local','10','25','36']:
  f=BASE/('source_'+key);pool=read(f/'parents.json');parents=pool['parents'];supervision=read(f/'d3_supervision.json');rows=supervision['rows'];res=read(f/'resource_constraints.json');held=BASE/'set_competition_dual_heldout_v1'/('source_'+key);prior=read(held/'receipt.json');pred=np.load(held/'predicted_counts.npy');prob=np.load(held/'competition_probability.npy');pairs=np.load(f/'competition_v4/pairs.npy');assert len(prob)==len(pairs);pc=read(f/'parent_children.json');keys=parent_keys(parents,pc,read(f/'child_records.json'));truthpath=ROOT/'server_rebuild/labels_v1'/key/'targets.json';tr=read(truthpath.parent/'receipt.json');assert tr['role']=='TRAIN' and tr['targets_sha256']==digest(truthpath);truth=read(truthpath)
  assert supervision['parents_sha256']==digest(f/'parents.json');byresource=collections.defaultdict(list)
  for i,resources in enumerate(res['parent_resources']):
   for r in resources:byresource[r].append(i)
  physical=[v for v in byresource.values() if len(v)>1];positive=np.array([r['parent_loss_valid'] and r['D3_TP']==r['children'] for r in rows]);owners=[set(r.get('compatible_best_parents',[])) if p else set() for r,p in zip(rows,positive)];known=np.array([len(owners[a])==len(owners[b])==1 for a,b in pairs]);labels=np.array([bool(owners[a]&owners[b]) for a,b in pairs]);default=res['default_parent_indices'];case=[]
  for mode in ['quality_positive_corrected','competition_known_corrected','both_partial_corrected']:
   start=time.perf_counter();counts=pred.copy();competition=prob>.5
   if mode!='competition_known_corrected':counts[positive]=np.array([[r['children'],0] for r,p in zip(rows,positive) if p])
   if mode!='quality_positive_corrected':competition[known]=labels[known]
   utility=counts[:,0]-counts[:,1];chosen,reason,solver=solve(utility,physical+pairs[competition].tolist(),res['internally_conflicting_parents'],default,keys);ps=[parents[i] for i in chosen];output=dict(parents=ps,invalid_predictions=pool['invalid_predictions'],total_prediction_children=sum(p['child_count'] for p in ps)+sum(p['counted_false_children'] for p in pool['invalid_predictions']));score=score_native_parents(output,truth,[False]*len(truth['parents']));d=DEST/('source_'+key)/mode;d.mkdir(parents=True);write(d/'prediction.json',output);report=dict(source=key,mode=mode,score=score,known_positive_parent_interventions=int(positive.sum()) if mode!='competition_known_corrected' else 0,known_competition_pairs=int(known.sum()) if mode!='quality_positive_corrected' else 0,selected_parent_indices=chosen,solver=solver,fallback=reason,offline_GT_diagnostic_only=True,not_model_score=True,prior_receipt_sha256=digest(held/'receipt.json'),supervision_sha256=digest(f/'d3_supervision.json'),seconds=time.perf_counter()-start);write(d/'diagnosis.json',report);case.append(dict(mode=mode,categories=score['categories']));print(key,mode,score['categories']['overall'],flush=True)
  summary.append(dict(source=key,baseline=read(held/'evaluation.json')['score']['categories'],interventions=case))
 write(DEST/'summary.json',dict(cases=summary,offline_GT_diagnostic_only=True,not_a_complete_oracle=True,not_a_model_score=True,no_fit=True));print('DIAGNOSIS_COMPLETE',flush=True)
if __name__=='__main__':main()
