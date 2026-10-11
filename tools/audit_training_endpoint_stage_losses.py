"""TRAIN-only child-set bottlenecks, retaining conditional-vs-full D3 distinction."""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def load_arrays(path):
 with np.load(path) as z:return {k:z[k] for k in z.files}
def main():
 out=BASE/'training_endpoint_stage_losses_v1';out.mkdir(exist_ok=False);model=BASE/'boundary_event_provenance_selector_v1';mr=read(model/'receipt.json');reports=[]
 for key in read(BASE/'reference_query_task_v1/protocol.json')['training_keys']:
  data=BASE/'exhaustive_carrier_supervision_v1/positive_query'/key;assert digest(data/'receipt.json')==mr['bindings'][key]['query'];records=read(data/'records.json');p=load_arrays(model/(key+'_training_event_predictions.npz'));sup=BASE/'boundary_anchor_supervision_v1'/key;assert digest(sup/'receipt.json')==mr['bindings'][key]['supervision'];s=load_arrays(sup/'targets.npz');assert np.array_equal(s['query_rows'],p['query_rows']);cache=BASE/'sampled_boundary_anchor_training_v1'/key;assert digest(cache/'receipt.json')==mr['bindings'][key]['events'];events=np.load(cache/'events.npz');pos=events['positions_mm'];detdir=BASE/'reference_detector_balanced_full_output_v1'/key;freeze=read(detdir/'prediction_freeze.json');assert digest(detdir/'predictions.npz')==freeze['predictions_sha256'];det=np.load(detdir/'predictions.npz');selected=det['valid']&det['line_eligible']&(det['reference_occupancy_score']>=.5)
  groups={n:set() for n in ['represented','both_events_available','both_selected_conditionally','detector_query_present','both_events_and_detector','both_selected_and_detector']};errors=[];counts=dict(known_heads=0,wrong_known_heads=0,wrong_nearest_valid_event_distance_le3=0,wrong_nearest_valid_event_distance_le10=0,wrong_nearest_valid_event_distance_gt10=0)
  for i,row in enumerate(p['query_rows']):
   rec=records[int(row)];owners={tuple(o) for o in rec['owners']};qid=rec['feature_rows'][0]//2*4+rec['query_index'];known=p['known_head_mask'][i];correct=bool(p['nonempty_both_correct'][i]);d=bool(selected[qid]);groups['represented'].update(owners)
   for name,ok in [('both_events_available',known.all()),('both_selected_conditionally',correct),('detector_query_present',d),('both_events_and_detector',known.all() and d),('both_selected_and_detector',correct and d)]:
    if ok:groups[name].update(owners)
   for h,(off,ids) in enumerate([('left_offsets','left_event_ids'),('right_offsets','right_event_ids')]):
    if not known[h]:continue
    counts['known_heads']+=1
    if p['correct_head_mask'][i,h]:continue
    counts['wrong_known_heads']+=1;a,b=s[off][i:i+2];distance=float(abs(pos[s[ids][a:b]]-pos[p['chosen_event_ids'][i,h]]).min());errors.append(distance)
    if distance<=3:counts['wrong_nearest_valid_event_distance_le3']+=1
    if distance<=10:counts['wrong_nearest_valid_event_distance_le10']+=1
    else:counts['wrong_nearest_valid_event_distance_gt10']+=1
  runtime=BASE/'reference_detector_balanced_endpoint_full_output_v1'/key;capacity=read(runtime/'child_reference_capacity.json');score=read(runtime/'new_score.json');report=dict(key=key,conditional_child_sets={k:sorted(v) for k,v in groups.items()},counts={k:len(v) for k,v in groups.items()},head_errors=counts,wrong_head_nearest_positive_event_mm_quantiles=np.quantile(errors,[0,.25,.5,.75,.9,.99,1]).tolist() if errors else [],actual_complete_D3=score['categories']['overall'],scope='TRAIN conditional line endpoints, ignores type/parent constraints until actual D3; detector presence uses only reliable positive-query records; actual runtime can find other query/profile witnesses. Distance is to valid candidate event, not true endpoint error.',bindings=dict(model=mr['model_sha256'],conditional_predictions=digest(model/(key+'_training_event_predictions.npz')),detector_freeze=digest(detdir/'prediction_freeze.json'),runtime_quality=digest(runtime/'child_reference_capacity.json')),heldout_reads=0,DEV_reads=0,fit_calls=0);write(out/(key+'.json'),report);reports.append(report);print(key,report['counts'],counts,flush=True)
 write(out/'summary.json',dict(cases=reports,heldout_reads=0,DEV_reads=0,source_sha256=digest(__file__),not_acceptance=True))
if __name__=='__main__':main()
