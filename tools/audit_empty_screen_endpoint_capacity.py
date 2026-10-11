"""TRAIN-only count/endpoint capacity diagnosis on existing positive witnesses.

Optimistic side/slot choice is offline only and never becomes inference output.
"""
from collections import defaultdict
import numpy as np
from threadpoolctl import threadpool_limits
import tools.rebuild_legacy_downstream as runtime
from weld_pipeline.sjg_local_runtime_904 import ROOT,read,write,digest
from weld_pipeline.sjg_ordered_interval_targets_1128 import decode_count_category
BASE=ROOT/'server_rebuild/expansion_v1';DEST=BASE/'empty_screen_endpoint_capacity_v1'
KEYS=('new_07380fe2eac4','new_0d355d5d0cbe','new_034230635b7a')

def main():
 DEST.mkdir(exist_ok=False);roles={r['key']:r['role'] for r in read(BASE/'group_manifest.json')['groups']};summaries=[]
 for key in KEYS:
  assert roles[key]=='EXPANSION_TRAIN';runtime.CASE=BASE/'frozen_features/legacy_runtime'/('source_'+key)
  case=runtime.module('sjg_relative_ordinal_runtime_1150').load_case(key);models,binding=runtime.module('sjg_recall_set_worker_1162').load_upstream(runtime.CASE/'run379')
  wp=BASE/'positive_intervals'/key/'witnesses.json';witnesses=[w for w in read(wp)['positive_witnesses'] if w['kind']=='line']
  rows=np.array(sorted({row for w in witnesses for f in w['fragments'] for row in f['feature_rows_both_unresolved_sides']}),int)
  x=np.asarray(case['x'][rows],np.float32);counts=np.array([decode_count_category(c)[1] for c in models['object'].classes_]);presence=models['object'].predict_proba(x)[:,counts>=1].sum(1)
  K=int(models.get('K',12));pred=np.empty((len(rows),K,2))
  for k in range(1,K+1):pred[:,k-1]=models['endpoint'].predict(np.column_stack([x,np.full(len(rows),k/K,dtype=np.float32)]))
  index={int(r):i for i,r in enumerate(rows)};records=[];bychild=defaultdict(list)
  for w in witnesses:
   fragments=[]
   for f in w['fragments']:
    rr=[index[r] for r in f['feature_rows_both_unresolved_sides']];sp=case['graph'].spans[f['canonical_fragment'],:2];q=pred[rr];truth=np.asarray(f['positive_ratios']);error=np.max(np.abs(q-truth),axis=2)*(sp[1]-sp[0]);valid=np.isfinite(q).all(2)&(q[:,:,0]>=0)&(q[:,:,1]<=1)&(q[:,:,1]>q[:,:,0]);error[~valid]=np.inf
    fragments.append(dict(fragment=f['canonical_fragment'],max_presence=float(presence[rr].max()),best_endpoint_error_mm=float(error.min()) if np.isfinite(error.min()) else None,endpoint_any_side_slot_within3=bool((error<=3).any()),target_ratio=truth.tolist()))
   row=dict(parent=w['parent'],child=w['child'],profile=w['profile'],fragments=fragments,all_fragments_presence_pass=all(f['max_presence']>=.5 for f in fragments),all_fragments_endpoint_oracle_pass=all(f['endpoint_any_side_slot_within3'] for f in fragments));records.append(row);bychild[w['parent'],w['child']].append(row)
  summary=dict(source=key,witness_children=len(bychild),witnesses=len(records),feature_rows=len(rows),children_with_any_all_fragment_presence=sum(any(w['all_fragments_presence_pass'] for w in ws) for ws in bychild.values()),children_with_any_all_fragment_endpoint_oracle=sum(any(w['all_fragments_endpoint_oracle_pass'] for w in ws) for ws in bychild.values()),maximum_presence=float(presence.max()),scope='Existing at-most-three-carrier positive witness subset; endpoint best of both unresolved sides and every ordinal slot. Does not prove full coverage, side identity, type, parent assembly, or extraction. No threshold changes.')
  write(DEST/(key+'.json'),dict(summary=summary,rows=records,witness_sha256=digest(wp),model_binding=binding));summaries.append(summary);print(summary,flush=True);case['x']._mmap.close()
 write(DEST/'receipt.json',dict(status='DIAGNOSTIC_COMPLETE',cases=summaries,fit_calls=0,protected_DEV_reads=0))

if __name__=='__main__':
 with threadpool_limits(limits=2):main()
