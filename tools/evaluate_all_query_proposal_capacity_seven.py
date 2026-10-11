"""Offline child geometry correspondences for the complete frozen input query pool.

Unmatched queries remain reference-unmatched/unknown; this is not a selected
weld output, physical-negative labeling, or parent-assembly acceptance score.
"""
import argparse,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.world_set_runtime_285 import line_segment
from weld_pipeline.contract_curve_geometry import CircularArc,sample_arc
from weld_pipeline.contract_geometry_d3_409 import child_error
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1'
def run(key):
 entry=next(c for c in read(BASE/'coverage_cohort_protocol.json')['cases'] if c['key']==key);folder=BASE/'positive_query_runtime_v2'/key;r=read(folder/'receipt.json');start=time.perf_counter()
 for name,sha in r['files'].items():assert digest(folder/name)==sha
 targetpath=BASE/'labels'/key/'targets.json';assert digest(targetpath.parent/'receipt.json')==entry['labels_receipt_sha256'];truth=native_targets(read(targetpath));profiles=read(BASE/'frozen_features/legacy_runtime'/('source_'+key)/'run185'/('source_'+key)/'profiles.json')
 pid=np.repeat(np.load(folder/'profile_ids.npy'),4);ratio=np.load(folder/'interval_ratios.npy').reshape(-1,2);role=np.load(folder/'weld_type.npy').ravel();valid=np.load(folder/'valid.npy').ravel();kinds=np.array([p['representation']['kind']=='arc' for p in profiles],bool)[pid];ends=np.empty((len(pid),2,3));curves={}
 assert np.all(pid[:-1]<=pid[1:]);unique,starts,counts=np.unique(pid,return_index=True,return_counts=True)
 for profile,first,count in zip(unique,starts,counts):
  ids=np.arange(first,first+count);curve=CurveCarrier(profiles[profile]['representation']);curves[int(profile)]=curve;ok=ids[valid[ids]];ends[ok]=curve.at(ratio[ok]*curve.length)
 indexes={};trees={}
 for kind in (False,True):
  for typ in (0,1):
   ix=np.flatnonzero(valid&(kinds==kind)&(role==typ));indexes[kind,typ]=ix;trees[kind,typ]=cKDTree(ends[ix,0]) if len(ix) else None
 rows=[];all_matches=set();controls=[];cache={}
 for pi,p in enumerate(truth['parents']):
  for ci,c in enumerate(p['children']):
   ref=reference_segment(c);e=np.asarray(ref['points'])[[0,-1]];kind=c['original_kind']=='arc';typ=0 if c['weld_type']=='horizontal' else 1;tree=trees[kind,typ];closed=bool(p['is_closed'] and len(p['children'])==1)
   if tree is None:eligible=[]
   elif closed:eligible=indexes[kind,typ].tolist()
   else:
    local=set(tree.query_ball_point(e[0],3.))|set(tree.query_ball_point(e[1],3.));ii=indexes[kind,typ][list(local)];err=np.minimum(np.linalg.norm(ends[ii]-e,axis=2).max(1),np.linalg.norm(ends[ii,::-1]-e,axis=2).max(1));eligible=ii[err<=3.].tolist()
   matches=[];direct_controls=0
   for i in eligible:
    # For two finite lines, paired endpoint <=3mm implies continuous mutual
    # inclusion <=3mm by convex interpolation. Verify one actual D3 per target.
    need_direct=kind or direct_controls==0
    passed=True
    if need_direct:
     signature=(int(pid[i]),ratio[i].tobytes(),pi,ci,int(role[i]))
     if signature in cache:passed=cache[signature]
     else:
      curve=curves[int(pid[i])];lo,hi=ratio[i]*curve.length;rep=curve.segment(float(lo),float(hi));s=line_segment(*rep['endpoints_mm']) if not kind else sample_arc(CircularArc.from_parameters(rep['arc_parameters']));s['weld_type']=c['weld_type'];passed=any(child_error(ss,c,ref,closed,False) is not None for ss in (s,dict(s,**reverse_segment(s))));cache[signature]=passed
     if not kind:assert passed,'Endpoint convexity disagrees with original line D3';direct_controls+=1
    if passed:matches.append(i)
   matches.sort();all_matches.update(matches);rows.append(dict(parent=pi,child=ci,kind=c['original_kind'],weld_type=c['weld_type'],compatible_query_ids=matches));controls.append(dict(parent=pi,child=ci,endpoint_candidates=len(eligible),D3_compatible=len(matches)));print(key,pi,ci,len(matches),flush=True)
 report=dict(key=key,total_input_queries=len(pid),valid_queries=int(valid.sum()),GT_children=len(rows),children_with_compatible_query=sum(bool(v['compatible_query_ids']) for v in rows),unique_queries_matching_any_GT_child=len(all_matches),reference_unmatched_valid_queries=int(valid.sum())-len(all_matches),rows=rows,controls=controls,target_sha256=digest(targetpath),runtime_receipt_sha256=digest(folder/'receipt.json'),source_sha256=digest(__file__),scope='Full frozen query pool child correspondence capacity; no selection, parent structure, duplicate penalty or physical-negative claim.',unknown_not_labeled_negative=True,GT_only_after_prediction_freeze=True,fit_calls=0,DEV_reads=0,seconds=time.perf_counter()-start);write(folder/'child_reference_capacity.json',report);print({k:v for k,v in report.items() if k not in ('rows','controls')},flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
 with threadpool_limits(limits=2):run(a.key)
