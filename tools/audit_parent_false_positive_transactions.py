"""Original-D3 whole-parent deletion ledger, evaluation only; never deployment."""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/unified_parent_scoring_v1'
def main():
 cases=[]
 for key in ['21','BM79A_native_local','10','25','36']:
  f=BASE/('source_'+key);d=BASE/'set_competition_support_v4'/('source_'+key);r=read(d/'receipt.json');e=read(d/'evaluation.json');pred=read(d/'prediction.json');assert digest(d/'prediction.json')==r['prediction_sha256']==e['prediction_sha256'];targetpath=ROOT/'server_rebuild/labels_v1'/key/'targets.json';assert digest(targetpath)==e['target_sha256'];truth=read(targetpath);old=e['score']['categories']['overall'];matched={(m['predicted_parent_index'],m['predicted_child_index']) for m in e['score']['matches']};labels=read(f/'d3_supervision_support_v2.json')['rows'];estimated=np.load(d/'predicted_counts.npy');pairprob=np.load(d/'competition_probability.npy');pairs=np.load(f/'competition_v2/pairs.npy');prob={tuple(map(int,p)):float(v) for p,v in zip(pairs,pairprob)};resources=read(f/'resource_constraints.json')['parent_resources'];reports=[]
  for pi,p in enumerate(pred['parents']):
   fp=[ci for ci in range(p['child_count']) if (pi,ci) not in matched]
   if not fp:continue
   candidate=r['selected_parent_indices'][pi];ps=pred['parents'][:pi]+pred['parents'][pi+1:];after=dict(pred,parents=ps,total_prediction_children=pred['total_prediction_children']-p['child_count']);score=score_native_parents(after,truth,[False]*len(truth['parents']))['categories']['overall'];alone=score_native_parents(dict(parents=[p],invalid_predictions=[],total_prediction_children=p['child_count']),truth,[False]*len(truth['parents']))['categories']['overall'];relations=[]
   for other in r['selected_parent_indices']:
    if other==candidate:continue
    pair=tuple(sorted((candidate,other)));shared=sorted(set(resources[candidate])&set(resources[other]));relations.append(dict(candidate=other,pair_retrieved=pair in prob,predicted_competition=prob.get(pair),shared_resource_count=len(shared)))
   reports.append(dict(selected_parent_index=pi,candidate_parent=candidate,false_child_indices=fp,complete_parent=p,supervision=labels[candidate],predicted_TP_FP=estimated[candidate].tolist(),single_parent_D3=alone,deletion_whole_parent_D3=score,deletion_TP_delta=score['TP']-old['TP'],deletion_FP_delta=score['FP']-old['FP'],selected_context_relations=relations,geometry_or_output_not_modified=True))
   print(key,candidate,'children',p['child_count'],'single',alone,'delete_delta',score['TP']-old['TP'],score['FP']-old['FP'],flush=True)
  report=dict(source=key,parents=reports,false_positive_parent_count=len(reports),false_positive_child_count=old['FP'],prediction_sha256=digest(d/'prediction.json'),target_sha256=digest(targetpath),GT_diagnostic_only=True,no_candidate_blacklist_created=True,local_annotation_scope_not_certified=True);write(d/'false_positive_transactions.json',report);cases.append({k:v for k,v in report.items() if k!='parents'})
 write(BASE/'false_positive_transaction_summary.json',dict(cases=cases,fit_calls=0,DEV_reads=0,source_sha256=digest(__file__)))
if __name__=='__main__':main()
