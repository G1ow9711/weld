"""Compare unchanged D3 results on actual labels, gates and whole-parent policies."""
from pathlib import Path
from types import FunctionType
import time,copy
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents,native_targets,controls
from weld_pipeline.server_array_cached_d3_evaluation import score_native_parents_cached
from weld_pipeline.world_set_evaluation_285 import reference_segment
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def main():
 out=BASE/'array_cached_d3_validation_v1';out.mkdir(exist_ok=False)
 env=dict(controls.__globals__);env['score_native_parents']=score_native_parents_cached;control_copy=FunctionType(controls.__code__,env,controls.__name__,controls.__defaults__);assert control_copy()==controls();rows=[]
 for key in ['new_ac2008a03607','new_fcaf2af1abee','new_cfc732e5f068']:
  raw=read(BASE/'labels'/key/'targets.json');target=native_targets(raw);selected=target['parents'][:12];truth=dict(raw,parents=raw['parents'][:12],original_children=sum(p['original_child_count'] for p in selected));parents=[]
  for p in selected:
   segments=[dict(reference_segment(c),weld_type=c['weld_type']) for c in p['children']]
   parents.append(dict(child_count=len(segments),is_closed=p['is_closed'],gaps=p['gaps'],segments=segments))
  decoded=dict(parents=parents,invalid_predictions=[],total_prediction_children=truth['original_children']);before=copy.deepcopy(decoded);raw_before=copy.deepcopy(truth)
  for policy in [None,[False]*len(selected),[True]*len(selected)]:
   t=time.perf_counter();a=score_native_parents(decoded,truth,policy);original_seconds=time.perf_counter()-t;t=time.perf_counter();b=score_native_parents_cached(decoded,truth,policy);cached_seconds=time.perf_counter()-t;assert a==b;assert decoded==before and truth==raw_before;rows.append(dict(key=key,policy=str(policy[:1]) if policy else None,original_seconds=original_seconds,cached_seconds=cached_seconds,result_exact=True))
 write(out/'receipt.json',dict(status='PASS',native_adapter_boundary_and_reversal_controls=True,actual_label_cases=rows,prediction_and_target_inputs_unchanged=True,original_scorer_source_sha256=digest('weld_pipeline/contract_geometry_d3_409.py'),cache_source_sha256=digest('weld_pipeline/server_array_cached_d3_evaluation.py'),no_sample_reduction=True,no_gate_or_matching_change=True));print(rows,flush=True)
if __name__=='__main__':main()
