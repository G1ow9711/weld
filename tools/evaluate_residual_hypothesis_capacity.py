"""Evaluate frozen TRAIN hypothesis capacity, not selected output precision."""
import argparse,time
from pathlib import Path
from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents
from weld_pipeline.sjg_parent_compatibility_1161 import compile_compatibility
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/parent_hypotheses_v1'
def run(key):
 started=time.perf_counter();folder=BASE/('source_'+key);receipt=read(folder/'receipt.json')
 for name,sha in receipt['files'].items():assert digest(folder/name)==sha
 target=ROOT/'server_rebuild/labels_v1'/key/'targets.json';tr=read(target.parent/'receipt.json');assert tr['role']=='TRAIN' and digest(target)==tr['targets_sha256']
 truth=read(target);decoded=read(folder/'hypotheses.json');score=score_native_parents(decoded,truth,[False]*len(truth['parents']));compat=compile_compatibility(decoded,truth)
 write(folder/'capacity_score.json',dict(score,diagnostic_only=True,unselected_hypotheses_are_not_final_predictions=True,target_sha256=digest(target),prediction_sha256=digest(folder/'hypotheses.json')))
 write(folder/'full_parent_compatibility.json',dict(compat,target_sha256=digest(target),prediction_sha256=digest(folder/'hypotheses.json')))
 summary=dict(source=key,TP=score['categories']['overall']['TP'],G=truth['original_children'],hypotheses=len(decoded['parents']),native_parents=receipt['native_parents'],new_parents=receipt['new_parents'],whole_parent_complete_children=compat['maximum_matched_children'],uncovered_full_parents=compat['uncovered_target_parents'],joint_feasibility_not_yet_checked=True,not_an_extraction_result=True,fit_calls=0,seconds=time.perf_counter()-started)
 write(folder/'capacity_summary.json',summary);print(summary,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--key',required=True,choices=['21','BM79A_native_local','10','25','36']);run(ap.parse_args().key)
