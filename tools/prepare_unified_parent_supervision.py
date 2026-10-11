"""Bind offline TRAIN D3 count supervision to the unified real-feature pool."""
from pathlib import Path
import numpy as np
from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets,score_native_parents
from weld_pipeline.contract_geometry_d3_409 import child_error
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'server_rebuild/unified_parent_scoring_v1'
def run(key):
 folder=BASE/('source_'+key);receipt=read(folder/'receipt.json')
 for name,sha in receipt['files'].items():assert digest(folder/name)==sha
 source=ROOT/'server_rebuild/parent_hypotheses_v1'/('source_'+key)/'d3_supervision.json';prior=read(source)
 targetpath=ROOT/'server_rebuild/labels_v1'/key/'targets.json';target=read(targetpath)
 assert digest(targetpath)==prior['target_sha256']==read(targetpath.parent/'receipt.json')['targets_sha256']
 assert prior['hypotheses_sha256']==receipt['hypothesis_sha256']
 parents=read(folder/'parents.json')['parents'];rows=list(prior['rows']);assert len(rows)==receipt['hypotheses']
 truth=[]
 for p in native_targets(target)['parents']:
  for c in p['children']:truth.append((c,reference_segment(c),bool(p['is_closed'] and len(p['children'])==1)))
 ends=np.array([np.asarray(ref['points'])[[0,-1]] for c,ref,closed in truth]);closed=np.array([v[2] for v in truth]);known_cache={}
 def known(s):
  sig=str((s['kind'],s['weld_type'],s['points']))
  if sig in known_cache:return known_cache[sig]
  e=np.asarray(s['points'])[[0,-1]];err=np.minimum(np.linalg.norm(ends-e,axis=2).max(1),np.linalg.norm(ends-e[::-1],axis=2).max(1))
  eligible=np.flatnonzero((err<=3.)|closed)
  result=any(child_error(ss,*truth[j],False) is not None for ss in (s,dict(s,**reverse_segment(s))) for j in eligible)
  known_cache[sig]=result;return result
 for i in range(len(rows),len(parents)):
  p=parents[i];n=p['child_count'];score=score_native_parents(dict(parents=[p],invalid_predictions=[],total_prediction_children=n),target,[False]*len(target['parents']))['categories']['overall']
  k=sum(known(s) for s in p['segments'])
  rows.append(dict(hypothesis=i,children=n,D3_TP=score['TP'],D3_FP=score['FP'],known_geometry_children=k,parent_loss_valid=k==n,source='B0_DEFAULT_DIRECT_D3'))
 report=dict(source=key,rows=rows,parents_sha256=digest(folder/'parents.json'),features_sha256=digest(folder/'features.npy'),prior_supervision_sha256=digest(source),target_sha256=digest(targetpath),unknown_geometry_parent_loss_masked=True,partial_TP_not_collapsed_to_binary=True,DEV_reads=0,fit_calls=0)
 write(folder/'d3_supervision.json',report)
 print(dict(source=key,parents=len(rows),masked=sum(not r['parent_loss_valid'] for r in rows),defaults=len(rows)-receipt['hypotheses']),flush=True)
if __name__=='__main__':
 for key in ['21','BM79A_native_local','10','25','36']:run(key)
