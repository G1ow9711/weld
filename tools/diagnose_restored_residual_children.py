"""TRAIN-only residual coverage; child capacity is not parent acceptance."""
import argparse,json
from pathlib import Path
import numpy as np
from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_shared_prediction_reader_1236 import read_shared_predictions
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.contract_geometry_d3_409 import child_error
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/best_baseline_restore'
def run(key):
 target=ROOT/'server_rebuild/labels_v1'/key/'targets.json'
 assert digest(target)==read(target.parent/'receipt.json')['targets_sha256']
 truth=native_targets(read(target));case=BASE/'legacy_runtime'/('source_'+key)
 evaluation=read(case/'run446'/('test_'+key)/'evaluation.json')
 assert evaluation['target_sha256']==digest(target)
 matched={(m['original_parent_index'],m['child_index']) for m in evaluation['matches']}
 missing=[(pi,ci,c) for pi,p in enumerate(truth['parents']) for ci,c in enumerate(p['children']) if (pi,ci) not in matched]
 result=[]
 for stage,path in [('baseline_unselected',case/'run379'/('test_'+key)/'decoded.json'),('pair_unselected',case/'run432'/('proposal_'+key)/'decoded.json')]:
  decoded=read_shared_predictions(path)
  segments=[(pi,ci,s) for pi,p in enumerate(decoded['parents']) for ci,s in enumerate(p['segments'])]
  endpoints=np.array([np.asarray(s['points'])[[0,-1]] for _,_,s in segments]);rows=[]
  for pi,ci,c in missing:
   ref=reference_segment(c);ends=np.asarray(ref['points'])[[0,-1]]
   forward=np.linalg.norm(endpoints-ends,axis=2).max(axis=1)
   backward=np.linalg.norm(endpoints[:,::-1]-ends,axis=2).max(axis=1)
   eligible=np.array([s['kind']==c['original_kind'] and s['weld_type']==c['weld_type'] for _,_,s in segments])
   best=np.minimum(forward,backward);hits=[]
   closed=truth['parents'][pi]['is_closed'] and truth['parents'][pi]['original_child_count']==1
   for i in np.flatnonzero(eligible & ((best<=3.) | closed)):
    for reverse in (False,True):
     s=segments[i][2];s=dict(s,**reverse_segment(s)) if reverse else s
     error=child_error(s,c,ref,closed,False)
     if error is not None:hits.append(dict(parent=segments[i][0],child=segments[i][1],reverse=reverse,**error))
   rows.append(dict(target_parent=pi,target_child=ci,weld_type=c['weld_type'],kind=c['original_kind'],geometry_match_count=len(hits),examples=hits[:3],nearest_type_compatible_endpoint_error_mm=float(best[eligible].min()) if eligible.any() else None))
  result.append(dict(stage=stage,prediction_sha256=digest(path),residual_children=rows))
 report=dict(source=key,role='TRAIN',target_sha256=digest(target),stages=result,fit_calls=0,DEV_reads=0,diagnostic_only=True,child_coverage_does_not_prove_parent_structure=True)
 dest=BASE/'capacity_diagnostics'/('source_'+key);dest.mkdir(exist_ok=True,parents=True);write(dest/'residual_children.json',report)
 print(json.dumps(report),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--key',choices=['21','BM79A_native_local','10','25','36'],required=True);run(ap.parse_args().key)
