"""Offline geometric oracle inside a frozen input-only bounded interval pool."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.server_compact_line_predictions import line_segment_array
from weld_pipeline.contract_geometry_d3_409 import child_error
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def run(key, pool_root='stable_pair_pool_freeze_v1'):
 folder=BASE/pool_root/key;r=read(folder/'receipt.json');assert r['GT_reads']==0 and r['budget']==64;path=folder/'pair_proposals.npz';assert digest(path)==r['pairs_sha256'];z=np.load(path);query,pair=np.nonzero(z['valid']);bounds=z['interval_mm'][query,pair];pids=z['profile_ids'][query];roles=z['weld_type'][query];profiles=read(BASE/'frozen_features/legacy_runtime'/('source_'+key)/'run185'/('source_'+key)/'profiles.json');assert np.all(pids[:-1]<=pids[1:]);ends=np.empty((len(query),2,3),np.float64)
 unique,starts,counts=np.unique(pids,return_index=True,return_counts=True)
 for profile,start,count in zip(unique,starts,counts):
  curve=CurveCarrier(profiles[int(profile)]['representation']);assert curve.kind=='line';ends[start:start+count]=curve.at(bounds[start:start+count])
 trees={};indices={}
 for role in [0,1]:
  ix=np.flatnonzero(roles==role);indices[role]=ix;trees[role]=cKDTree(ends[ix,0]) if len(ix) else None
 targetpath=BASE/'labels'/key/'targets.json';truth=native_targets(read(targetpath));rows=[];hits=set();matching_queries=set();direct_checks=0
 for pi,parent in enumerate(truth['parents']):
  for ci,child in enumerate(parent['children']):
   row=dict(parent=pi,child=ci,kind=child['original_kind'],compatible_pair_count=0,compatible_query_count=0)
   if child['original_kind']=='line':
    ref=reference_segment(child);ep=np.asarray(ref['points'])[[0,-1]];role=0 if child['weld_type']=='horizontal' else 1;tree=trees[role]
    if tree is not None:
     local=set(tree.query_ball_point(ep[0],3.))|set(tree.query_ball_point(ep[1],3.));ix=indices[role][list(local)];err=np.minimum(np.linalg.norm(ends[ix]-ep,axis=2).max(1),np.linalg.norm(ends[ix,::-1]-ep,axis=2).max(1));ix=ix[err<=3.]
     if len(ix):
      seg=line_segment_array(*ends[ix[0]]);seg['weld_type']=child['weld_type'];assert any(child_error(s,child,ref,False,False) is not None for s in (seg,dict(seg,**reverse_segment(seg))));direct_checks+=1;hits.add((pi,ci));matching_queries.update(query[ix].tolist());row.update(compatible_pair_count=len(ix),compatible_query_count=len(np.unique(query[ix])),witness_query_id=int(z['selected_query_ids'][query[ix[0]]]),witness_pair_slot=int(pair[ix[0]]))
   rows.append(row)
 report=dict(key=key,total_queries=len(z['selected_query_ids']),valid_input_pairs=len(query),reference_children=len(rows),children_with_geometric_type_compatible_pair=len(hits),queries_matching_any_reference=len(matching_queries),direct_D3_witness_checks=direct_checks,rows=rows,pool_receipt_sha256=digest(folder/'receipt.json'),target_sha256=digest(targetpath),scope='Oracle over already-frozen line interval pairs with predicted type. Each reference considered independently; not simultaneous parent/resource feasibility, learned selection, FP score or full acceptance. No GT event insertion.',fit_calls=0,DEV_reads=0,source_sha256=digest(__file__));write(folder/'pair_capacity.json',report);print({k:v for k,v in report.items() if k!='rows'},flush=True)
if __name__=='__main__':
 for key in ['new_cfc732e5f068','new_ceeae49d6777']:run(key)
