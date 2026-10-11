"""Offline audit of residual errors in a frozen support-supervised selection."""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.world_set_evaluation_285 import reference_segment
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/unified_parent_scoring_v1'
def main():
 reports=[]
 for key in ['21','BM79A_native_local','10','25','36']:
  source=BASE/('source_'+key);folder=BASE/'set_competition_support_v2'/('source_'+key);evaluation=read(folder/'evaluation.json');r=read(folder/'receipt.json');prediction=read(folder/'prediction.json');labels=read(source/'d3_supervision_support_v2.json')['rows'];raw=read(source/'receipt.json');counts=np.load(folder/'predicted_counts.npy');target=ROOT/'server_rebuild/labels_v1'/key/'targets.json';truth=native_targets(read(target));targets=[]
  for pi,p in enumerate(truth['parents']):
   for ci,c in enumerate(p['children']):
    ref=reference_segment(c)
    if ref['kind']=='line':targets.append((pi,ci,c['weld_type'],np.asarray(ref['points'])[[0,-1]]))
  passed={(m['predicted_parent_index'],m['predicted_child_index']) for m in evaluation['score']['matches']};errors=[]
  for pi,p in enumerate(prediction['parents']):
   idx=r['selected_parent_indices'][pi]
   for ci,s in enumerate(p['segments']):
    if (pi,ci) in passed:continue
    e=np.asarray(s['points'])[[0,-1]];choices=[]
    for ti,tc,wt,te in targets:
     if wt!=s['weld_type']:continue
     v=te[1]-te[0];length=np.linalg.norm(v);v/=length;q=(e-te[0])@v;perp=np.linalg.norm(e-te[0]-q[:,None]*v,axis=1).max();finite=np.linalg.norm(e-te[0]-np.clip(q,0,length)[:,None]*v,axis=1).max();endpoint=min(np.linalg.norm(e-te,axis=1).max(),np.linalg.norm(e-te[::-1],axis=1).max());choices.append((endpoint,finite,perp,ti,tc,q.tolist(),length))
    best=min(choices);row=dict(candidate_parent=idx,parent_children=p['child_count'],candidate_child=ci,kind=s['kind'],weld_type=s['weld_type'],profile=s['profile'],length_mm=s['length_mm'],endpoints=e.tolist(),supervision=labels[idx],predicted_TP_FP=counts[idx].tolist(),default=idx>=raw['hypotheses'],nearest_label=dict(endpoint_max_mm=best[0],finite_support_max_mm=best[1],infinite_line_perp_max_mm=best[2],parent=best[3],child=best[4],projected_interval_mm=best[5],GT_length_mm=best[6]));errors.append(row)
  report=dict(source=key,errors=errors,prediction_sha256=digest(folder/'prediction.json'),target_sha256=digest(target),diagnostic_GT_only=True);write(folder/'residual_errors.json',report);reports.append(report)
  for row in errors:print(key,row,flush=True)
 write(BASE/'set_competition_support_v2/residual_errors.json',reports)
if __name__=='__main__':main()
