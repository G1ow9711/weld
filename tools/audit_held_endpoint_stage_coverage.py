"""Offline stage coverage on already-evaluated expansion TRAIN families."""
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,read,write,digest
B=ROOT/'server_rebuild/expansion_v1';OUT=B/'held_endpoint_stage_coverage_v1'

def main():
 OUT.mkdir(exist_ok=False);summaries=[];roles={v['key']:v['role'] for v in read(B/'group_manifest.json')['groups']}
 for key in ('new_cfc732e5f068','new_ceeae49d6777'):
  assert roles[key]=='EXPANSION_TRAIN'
  q=B/'positive_query_head_v1'/key;qr=read(q/'receipt.json');assert digest(q/'records.json')==qr['files']['records.json']
  records=read(q/'records.json');folder=B/'bounded_raw_raw_full_output_v1'/key;freeze=read(folder/'prediction_freeze.json');assert digest(folder/'predictions.npz')==freeze['files']['predictions.npz']
  with np.load(folder/'predictions.npz') as z:selected=set(z['selected_query_ids'].tolist())
  represented=set();admitted=set();matched_queries=0
  for r in records:
   owners={tuple(o) for o in r['owners']};represented.update(owners);qid=r['feature_rows'][0]//2*4+r['query_index']
   if qid in selected:admitted.update(owners);matched_queries+=1
  capacity=read(folder/'child_reference_capacity.json');assert capacity['runtime_receipt_sha256']==digest(folder/'prediction_freeze.json') and capacity['target_sha256']==qr['target_sha256']
  final={ (r['parent'],r['child']) for r in capacity['rows'] if r['compatible_query_ids'] }
  summary=dict(source=key,reference_children=capacity['GT_children'],positive_witness_subset_children=len(represented),subset_children_with_detector_selected_query=len(admitted),subset_positive_queries=len(records),subset_positive_queries_admitted=matched_queries,total_detector_selected_queries=len(selected),children_with_correct_endpoint_and_type_after_detection=len(final),children_admitted_in_subset_but_no_final_geometry=len(admitted-final),children_final_outside_subset_admission=len(final-admitted),final_unique_matching_queries=capacity['unique_queries_matching_any_GT_child'],reference_unmatched_valid_queries=capacity['reference_unmatched_valid_queries'])
  write(OUT/(key+'.json'),dict(summary=summary,represented=sorted(represented),admitted=sorted(admitted),final_geometry=sorted(final),bindings=dict(query_receipt=digest(q/'receipt.json'),prediction_freeze=digest(folder/'prediction_freeze.json'),capacity=digest(folder/'child_reference_capacity.json')),scope='Positive witness subset uses at most three carriers. Admission count is a lower bound on positive query coverage; final geometry may come from different queries. Already-held expansion TRAIN data, no new model fit/threshold choice, no protected DEV.'))
  summaries.append(summary);print(summary,flush=True)
 write(OUT/'receipt.json',dict(status='COMPLETE',cases=summaries,fit_calls=0,protected_DEV_reads=0))

if __name__=='__main__':main()
