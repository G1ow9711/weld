"""TRAIN endpoint fit separated by parent outer endpoint versus internal child endpoint."""
from pathlib import Path
import collections
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def main():
 model=BASE/'boundary_event_provenance_selector_v1';r=read(model/'receipt.json');keys=read(BASE/'reference_query_task_v1/protocol.json')['training_keys'];reports=[];out=BASE/'boundary_provenance_endpoint_scope_v1';out.mkdir(exist_ok=False)
 for key in keys:
  data=BASE/'exhaustive_carrier_supervision_v1/positive_query'/key;assert digest(data/'receipt.json')==r['bindings'][key]['query'];records=read(data/'records.json');truth=read(BASE/'labels'/key/'targets.json');profiles=read(BASE/'frozen_features/legacy_runtime'/('source_'+key)/'run185'/('source_'+key)/'profiles.json');pred=np.load(model/(key+'_training_event_predictions.npz'));stats=collections.defaultdict(lambda:collections.Counter());endpoints={}
  for row,known,correct in zip(pred['query_rows'],pred['known_head_mask'],pred['correct_head_mask']):
   rec=records[row];ends=np.asarray(profiles[rec['profile']]['representation']['endpoints_mm']);direction=ends[1]-ends[0]
   for pi,ci in rec['owners']:
    parent=truth['parents'][pi];c=parent['children'][ci];ref=np.asarray(c['representation']['endpoints_mm']);forward=np.dot(ref[1]-ref[0],direction)>=0
    for head in (0,1):
     side=head if forward else 1-head;scope='PARENT_OUTER' if (side==0 and ci==0) or (side==1 and ci==len(parent['children'])-1) else 'INTERNAL_CHILD';stats[scope]['query_owner_heads']+=1;stats[scope]['known_query_owner_heads']+=int(known[head]);stats[scope]['correct_query_owner_heads']+=int(correct[head]);e=endpoints.setdefault((pi,ci,side),dict(scope=scope,known=False,correct=False));e['known']|=bool(known[head]);e['correct']|=bool(correct[head])
  for e in endpoints.values():stats[e['scope']]['represented_child_endpoints']+=1;stats[e['scope']]['endpoints_with_known_target']+=e['known'];stats[e['scope']]['endpoints_with_any_correct_query']+=e['correct']
  report=dict(key=key,categories={k:dict(v) for k,v in stats.items()},prediction_sha256=digest(model/(key+'_training_event_predictions.npz')),scope='Conditional TRAIN fit; parent outer versus internal position only, not inferred split cause; child outer endpoints on same parent counted separately',fit_calls=0,DEV_reads=0);write(out/(key+'.json'),report);reports.append(report);print(key,report['categories'],flush=True)
 write(out/'summary.json',dict(cases=reports,model_sha256=r['model_sha256'],heldout_reads=0,not_full_extraction_accuracy=True))
if __name__=='__main__':main()
