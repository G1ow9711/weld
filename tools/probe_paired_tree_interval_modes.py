"""Frozen forest mean versus preselected complete paired tree hypotheses.

Conditional positive-query probe only; eight hypotheses cost eight candidates,
so coverage improvement is not an accuracy or equal-candidate-budget claim.
"""
import pickle
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.world_set_runtime_285 import line_segment
from weld_pipeline.contract_curve_geometry import CircularArc,sample_arc
from weld_pipeline.contract_geometry_d3_409 import child_error
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';MODEL=BASE/'positive_query_head_v1/model_initial_seven_families';DEST=BASE/'paired_tree_interval_modes_v1';KEYS=('new_cfc732e5f068','new_ceeae49d6777')
def main():
 DEST.mkdir(exist_ok=False);mr=read(MODEL/'receipt.json');assert digest(MODEL/'model.pkl')==mr['model_sha256']
 with (MODEL/'model.pkl').open('rb') as f:model=pickle.load(f)
 assert all(k not in model['training_keys'] for k in KEYS);members=np.linspace(0,len(model['interval'].estimators_)-1,7,dtype=int).tolist();write(DEST/'protocol.json',dict(model_sha256=mr['model_sha256'],members=members,include_forest_mean=True,complete_endpoint_pairs_not_independent_marginals=True,K=8,fit_calls=0,conditional_only=True,not_equal_candidate_budget=True,no_GT_used_to_choose_tree_indices=True,keys=KEYS));reports=[]
 for key in KEYS:
  folder=BASE/'positive_query_head_v1'/key;r=read(folder/'receipt.json');assert digest(folder/'features.npy')==r['files']['features.npy'];x=np.load(folder/'features.npy');records=read(folder/'records.json');bounds=np.stack([model['interval'].predict(x)]+[model['interval'].estimators_[i].predict(x) for i in members],axis=1);roles=model['role'].predict(x);dest=DEST/key;dest.mkdir();np.savez(dest/'predictions.npz',interval_ratios=bounds,weld_type=roles);profiles=read(BASE/'frozen_features/legacy_runtime'/('source_'+key)/'run185'/('source_'+key)/'profiles.json');truth=native_targets(read(BASE/'labels'/key/'targets.json'));curves={};hits=[set() for _ in range(8)]
  for row,pred,role in zip(records,bounds,roles):
   pid=row['profile']
   if pid not in curves:curves[pid]=CurveCarrier(profiles[pid]['representation'])
   curve=curves[pid]
   for mi,ratio in enumerate(pred):
    if not (0<=ratio[0]<ratio[1]<=1):continue
    ends=curve.at(ratio*curve.length)
    for pi,ci in row['owners']:
     child=truth['parents'][pi]['children'][ci];wt='horizontal' if role==0 else 'vertical'
     if wt!=child['weld_type'] or (pi,ci) in hits[mi]:continue
     ref=reference_segment(child);e=np.asarray(ref['points'])[[0,-1]];closed=bool(truth['parents'][pi]['is_closed'] and len(truth['parents'][pi]['children'])==1)
     if not closed and min(np.linalg.norm(ends-e,axis=1).max(),np.linalg.norm(ends[::-1]-e,axis=1).max())>3:continue
     rep=curve.segment(*(ratio*curve.length));s=line_segment(*rep['endpoints_mm']) if curve.kind=='line' else sample_arc(CircularArc.from_parameters(rep['arc_parameters']));s['weld_type']=wt
     if any(child_error(v,child,ref,closed,False) is not None for v in (s,dict(s,**reverse_segment(s)))):hits[mi].add((pi,ci))
  union=set().union(*hits);report=dict(key=key,represented_children=r['represented_GT_children'],known_positive_queries=len(x),forest_mean_covered=len(hits[0]),K8_union_covered=len(union),per_member_covered=[len(h) for h in hits],covered_children=sorted(union),prediction_sha256=digest(dest/'predictions.npz'),scope='Conditional proposal capacity only; eight candidates per query, no filtering or parent assembly; shared frozen upstream',fit_calls=0,DEV_reads=0);write(dest/'receipt.json',report);reports.append(report);print(report,flush=True)
 write(DEST/'summary.json',dict(cases=reports,fit_calls=0))
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
