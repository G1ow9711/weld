"""TRAIN-only child compatibility for frozen unscreened parent candidates.

Store all allowed correspondences. Do not derive negatives from missing labels.
"""
from pathlib import Path
import numpy as np
from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.contract_geometry_d3_409 import child_error
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment

ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'server_rebuild/parent_residual_v1'
def main():
 manifest=DEST/'inputs_frozen.json';inputs=read(manifest);reports=[]
 for case in inputs['cases']:
  key=case['source'];assert case['role']=='TRAIN' and key!='64'
  path=ROOT/case['files']['parents']['path'];assert digest(path)==case['files']['parents']['sha256']
  targetpath=ROOT/'server_rebuild/labels_v1'/key/'targets.json'
  assert digest(targetpath)==read(targetpath.parent/'receipt.json')['targets_sha256']
  target=native_targets(read(targetpath));truth=[]
  for pi,p in enumerate(target['parents']):
   for ci,c in enumerate(p['children']):truth.append((pi,ci,c,reference_segment(c),bool(p['is_closed'] and len(p['children'])==1)))
  targetends=np.array([np.asarray(ref['points'])[[0,-1]] for _,_,c,ref,closed in truth]);closedmask=np.array([t[-1] for t in truth]);rows=[]
  for pi,p in enumerate(read(path)['parents']):
   for ci,segment in enumerate(p['segments']):
    ends=np.asarray(segment['points'])[[0,-1]];errors=np.minimum(np.linalg.norm(targetends-ends,axis=2).max(1),np.linalg.norm(targetends-ends[::-1],axis=2).max(1))
    eligible=np.flatnonzero((errors<=3.)|closedmask);owners=[]
    for reversed_ in (False,True):
     s=dict(segment,**reverse_segment(segment)) if reversed_ else segment
     for j in eligible:
      tp,tc,c,ref,closed=truth[j]
      if child_error(s,c,ref,closed,False) is not None:owners.append(dict(parent=tp,child=tc,reversed=reversed_))
    rows.append(dict(candidate_parent=pi,candidate_child=ci,owners=owners))
  folder=DEST/('source_'+key);report=dict(source=key,rows=rows,target_sha256=digest(targetpath),prediction_sha256=digest(path),input_manifest_sha256=digest(manifest),GT_training_only=True,unmatched_is_unknown=True,not_an_extraction_result=True)
  write(folder/'child_ownership.json',report);row=dict(source=key,children=len(rows),children_with_known_matches=sum(bool(x['owners']) for x in rows),sha256=digest(folder/'child_ownership.json'));reports.append(row);print(row,flush=True)
 write(DEST/'child_ownership_receipt.json',dict(cases=reports,fit_calls=0,DEV_reads=0,negative_labels_not_assigned=True))
if __name__=='__main__':main()
