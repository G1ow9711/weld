"""Offline TRAIN error attribution; never repairs runtime output with labels."""
from weld_pipeline.sjg_local_runtime_904 import ROOT,read,write,digest
from weld_pipeline.sjg_shared_prediction_reader_1236 import read_shared_predictions
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.contract_geometry_d3_409 import child_error
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment


def main():
 base=ROOT/'server_rebuild/roundoff_gap_baseline_probe_v1';predpath=base/'legacy_runtime/source_10/run379/test_10/final_decoded.json';pred=read_shared_predictions(predpath);score=read(base/'evaluation.json')['score'];old=read(ROOT/'server_rebuild/best_baseline_restore/legacy_runtime/source_10/run379/test_10/evaluation.json');truth=native_targets(read(ROOT/'server_rebuild/labels_v1/10/targets.json'))
 matched={(m['predicted_parent_index'],m['predicted_child_index']) for m in score['matches']};oldids={m['original_child_id'] for m in old['matches']};newids={m['original_child_id'] for m in score['matches']}
 targets=[(pi,ci,c,reference_segment(c)) for pi,p in enumerate(truth['parents']) for ci,c in enumerate(p['children'])];rows=[]
 for pi,parent in enumerate(pred['parents']):
  for ci,s in enumerate(parent['segments']):
   if (pi,ci) in matched:continue
   compatible=[]
   for ti,tc,t,ref in targets:
    for rev,segment in [(False,s),(True,dict(s,**reverse_segment(s)))]:
     err=child_error(segment,t,ref,False,False)
     if err is not None:compatible.append(dict(target_parent=ti,target_child=tc,target_id=t['original_child_id'],diagnostic_child_reverse=rev,error=err,reference_parent_children=len(truth['parents'][ti]['children']),reference_parent_gaps=truth['parents'][ti]['gaps']))
   rows.append(dict(predicted_parent=pi,predicted_child=ci,parent_children=len(parent['segments']),parent_gaps=parent['gaps'],weld_type=s['weld_type'],kind=s['kind'],endpoints_mm=s['points'][[0,-1]].tolist(),standalone_compatible=compatible))
 report=dict(gained_target_ids=sorted(newids-oldids),lost_target_ids=sorted(oldids-newids),unmatched_children=rows,invalid_predictions=pred['invalid_predictions'],prediction_sha256=digest(predpath),scope='Offline TRAIN standalone child geometry diagnosis; independent child reversal here is diagnostic only, not accepted parent scoring or runtime repair.')
 write(base/'error_attribution.json',report);print('GAINED',len(newids-oldids),'LOST',len(oldids-newids),'UNMATCHED',[(r['predicted_parent'],r['predicted_child'],len(r['standalone_compatible'])) for r in rows],flush=True)

if __name__=='__main__':main()
