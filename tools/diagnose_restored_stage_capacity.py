"""TRAIN-only whole-parent coverage diagnostics on frozen, unselected candidates."""
import argparse,time
from pathlib import Path
from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/best_baseline_restore'
def parent_coverage(score,truth):
 matched={}
 for m in score['matches']:
  matched.setdefault(m['original_parent_index'],set()).add(m['child_index'])
 incomplete=[]
 for i,p in enumerate(truth['parents']):
  seen=matched.get(i,set());missing=sorted(set(range(p['original_child_count']))-seen)
  if missing:incomplete.append(dict(index=i,id=p['original_parent_id'],children=p['original_child_count'],matched_children=len(seen),missing_child_indices=missing))
 return dict(zero_match_parents=[p for p in incomplete if p['matched_children']==0],incomplete_parents=incomplete,fully_matched_parent_count=len(truth['parents'])-len(incomplete),scoring_semantics='Parent structure must match; TP counts individual geometry-passing children under one allowed whole-parent correspondence. Not all-or-nothing parent geometry.')

def run(key):
 case=BASE/'legacy_runtime'/('source_'+key);target=ROOT/'server_rebuild/labels_v1'/key/'targets.json'
 assert digest(target)==read(target.parent/'receipt.json')['targets_sha256'];truth=read(target);dest=BASE/'capacity_diagnostics'/('source_'+key);dest.mkdir(parents=True,exist_ok=False)
 stages={'baseline_unselected':case/'run379'/('test_'+key)/'decoded.json','baseline_selected':case/'run379'/('test_'+key)/'final_decoded.json','pair_unselected':case/'run432'/('proposal_'+key)/'decoded.json','composite_selected':case/'run446'/('test_'+key)/'final_decoded.json'}
 rows=[]
 for stage,path in stages.items():
  started=time.perf_counter();decoded=read(path)
  # Pair storage may encode shared arrays; use original reader for both formats.
  from weld_pipeline.sjg_shared_prediction_reader_1236 import read_shared_predictions
  decoded=read_shared_predictions(path)
  score=score_native_parents(decoded,truth,[False]*len(truth['parents']))
  write(dest/(stage+'.json'),dict(score,diagnostic_only=True,unselected_pool_is_not_model_output=stage.endswith('unselected'),prediction_sha256=digest(path),target_sha256=digest(target)))
  row=dict(stage=stage,**score['categories']['overall'],**parent_coverage(score,truth),seconds=time.perf_counter()-started);rows.append(row);print(row,flush=True)
 write(dest/'summary.json',dict(source=key,role='TRAIN',stages=rows,fit_calls=0,not_an_acceptance_result=True))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--key',required=True,choices=['21','BM79A_native_local','10','25','36']);run(ap.parse_args().key)
