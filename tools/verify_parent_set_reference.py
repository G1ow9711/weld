from pathlib import Path
import numpy as np
from weld_pipeline.server_compact_line_predictions import line_segment_array
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents
from weld_pipeline.server_offline_parent_set_reference import compile_reference_table,evaluate_reference_subset
from weld_pipeline.sjg_local_runtime_904 import write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def segment(a,b):return dict(line_segment_array(a,b),weld_type='horizontal')
def parent(segments,gaps=None):return dict(child_count=len(segments),segments=segments,gaps=gaps or [],is_closed=False)
def main():
 a=segment([0,0,0],[10,0,0]);b=segment([10,0,0],[20,0,0]);bad=segment([10,4,0],[20,4,0]);c=segment([40,0,0],[50,0,0])
 children=lambda ss:[dict(original_child_id=str(i),original_kind='line',weld_type='horizontal',representation=dict(kind='line',endpoints_mm=s['endpoints_mm'])) for i,s in enumerate(ss)]
 target=dict(original_children=3,parents=[dict(original_child_count=2,is_closed=False,gaps=['SHARED'],children=children([a,b])),dict(original_child_count=1,is_closed=False,gaps=[],children=children([c]))])
 pool=[parent([a,b],['SHARED']),parent([a,b],['SHARED']),parent([a,bad],['SHARED']),parent([a,b],['POSITIVE_GAP']),parent([c])]
 decoded=dict(parents=pool,invalid_predictions=[],total_prediction_children=9)
 table,score=compile_reference_table(decoded,target,[False,False]);checks=0
 for bits in range(1<<len(pool)):
  ids=[i for i in range(len(pool)) if bits>>i&1]
  for order in [ids,ids[::-1]]:
   part=dict(decoded,parents=[pool[i] for i in order],total_prediction_children=sum(pool[i]['child_count'] for i in order))
   actual=score_native_parents(part,target,[False,False])['categories']['overall'];fast=evaluate_reference_subset(table,order)
   assert all(actual[k]==fast[k] for k in ['TP','FP','predicted']);checks+=1
 assert evaluate_reference_subset(table,[0,1])['reference_utility']==0
 assert evaluate_reference_subset(table,[0])['reference_utility']==2
 assert evaluate_reference_subset(table,[2])['TP']==1
 assert evaluate_reference_subset(table,[3])['TP']==0
 assert evaluate_reference_subset(table,[])['predicted']==0
 out=BASE/'parent_set_reference_controls_v1';out.mkdir(exist_ok=False);write(out/'receipt.json',dict(status='PASS',all_subset_order_controls=checks,duplicate_set_penalty=True,partial_parent_TP_preserved=True,gap_semantics_preserved=True,empty_subset=True,module_sha256=digest('weld_pipeline/server_offline_parent_set_reference.py')));print('PASS',checks,flush=True)
if __name__=='__main__':main()
