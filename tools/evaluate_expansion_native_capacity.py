"""TRAIN-only coverage of frozen native parents; not an extraction promotion."""
import argparse,time
from pathlib import Path
from threadpoolctl import threadpool_limits
from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents
from tools.diagnose_restored_stage_capacity import parent_coverage
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1'
def run(key):
 entry=next(c for c in read(BASE/'coverage_cohort_protocol.json')['cases'] if c['key']==key)
 case=BASE/'frozen_features/legacy_runtime'/('source_'+key);assert read(case/'select_reconstruction_receipt.json')['GT_reads']==0
 target=BASE/'labels'/key/'targets.json';assert digest(target.parent/'receipt.json')==entry['labels_receipt_sha256'];assert digest(target)==read(target.parent/'receipt.json')['targets_sha256'];truth=read(target)
 dest=BASE/'native_capacity'/('source_'+key);dest.mkdir(parents=True,exist_ok=False);rows=[]
 for stage,name in [('native_unselected','decoded.json'),('frozen379_selected','final_decoded.json')]:
  path=case/'run379'/('test_'+key)/name;start=time.perf_counter();score=score_native_parents(read(path),truth,[False]*len(truth['parents']))
  write(dest/(stage+'.json'),dict(score,target_sha256=digest(target),prediction_sha256=digest(path),diagnostic_only=True,unselected_pool_is_not_model_output=stage=='native_unselected'))
  row=dict(stage=stage,**score['categories']['overall'],**parent_coverage(score,truth),seconds=time.perf_counter()-start);rows.append(row);print(key,stage,score['categories'],flush=True)
 write(dest/'receipt.json',dict(key=key,role='EXPANSION_TRAIN',stages=rows,GT_used_after_input_freeze=True,fit_calls=0,DEV_reads=0,not_an_acceptance_result=True,source_sha256=digest(__file__)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
 with threadpool_limits(limits=2):run(a.key)
