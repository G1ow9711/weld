"""TRAIN-only projected interval capacity; GT-derived intervals are not outputs."""
import argparse
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.contract_curve_geometry import CircularArc,sample_arc
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.contract_geometry_d3_409 import child_error
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment
from weld_pipeline.server_circle_capacity_prefilter import possible_circle_indices
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1'
def run(key, folder_name='local_graph_arc_candidates_v1', prefilter=False):
 folder=BASE/folder_name/key;binding=read(folder/'receipt.json');assert binding['status']=='INPUT_ARC_CANDIDATES_FROZEN';profiles=[]
 assert digest(folder/'protocol.json')==binding['protocol_sha256']
 for face in binding['faces']:
  path=folder/face['file'];assert digest(path)==face['sha256']
  for i,candidate in enumerate(read(path)['candidates']):profiles.append(dict(face=face['face'],candidate=i,representation=dict(kind='arc',arc_parameters=candidate['arc_parameters'])))
 curves=[CurveCarrier(p['representation']) for p in profiles];targetpath=BASE/'labels'/key/'targets.json';target=native_targets(read(targetpath));rows=[]
 centers=np.asarray([c.arc.center_mm for c in curves]);normals=np.asarray([c.arc.plane_normal for c in curves]);radii=np.asarray([c.arc.radius_mm for c in curves])
 for pi,p in enumerate(target['parents']):
  for ci,c in enumerate(p['children']):
   if c['original_kind']!='arc':continue
   ref=reference_segment(c);closed=bool(p['is_closed'] and len(p['children'])==1);matches=[];full_matches=[]
   eligible=possible_circle_indices(centers,normals,radii,ref['points']) if prefilter else range(len(curves))
   for i in eligible:
    curve=curves[i]
    q,d=curve.project(np.asarray(ref['points']))
    if float(d.max())>3.:continue
    ends=curve.at(np.array([0.,curve.length]));refends=np.asarray(ref['points'])[[0,-1]]
    endpoint_distance=min(float(np.linalg.norm(ends-refends,axis=1).max()),float(np.linalg.norm(ends[::-1]-refends,axis=1).max()))
    if closed or endpoint_distance<=3.:
     full=sample_arc(CircularArc.from_parameters(profiles[i]['representation']['arc_parameters']));full['weld_type']=c['weld_type']
     for reverse in (False,True):
      ff=dict(full,**reverse_segment(full)) if reverse else full;error=child_error(ff,c,ref,closed,False)
      if error is not None:full_matches.append(dict(profile=i,face=profiles[i]['face'],candidate=profiles[i]['candidate'],reverse=reverse,**error));break
    lo,hi=sorted((float(q[0]),float(q[-1])))
    if not hi>lo or np.any((q<lo-1e-8)|(q>hi+1e-8)):continue
    s=sample_arc(CircularArc.from_parameters(curve.segment(lo,hi)['arc_parameters']));s['weld_type']=c['weld_type']
    for reverse in (False,True):
     ss=dict(s,**reverse_segment(s)) if reverse else s;error=child_error(ss,c,ref,closed,False)
     if error is not None:matches.append(dict(profile=i,face=profiles[i]['face'],candidate=profiles[i]['candidate'],interval_mm=[lo,hi],reverse=reverse,**error));break
   rows.append(dict(parent=pi,child=ci,matches=matches,full_candidate_matches=full_matches))
 report=dict(key=key,arc_children=len(rows),children_with_original_D3_witness=sum(bool(r['matches']) for r in rows),children_with_full_candidate_geometry_witness=sum(bool(r['full_candidate_matches']) for r in rows),candidate_count=len(curves),reference_type_supplied_for_geometry_capacity_only=True,rows=rows,GT_projected_endpoints_offline_only=True,not_runtime_prediction=True,not_parent_assembly_or_contact_proof=True,candidate_receipt_sha256=digest(folder/'receipt.json'),target_sha256=digest(targetpath),fit_calls=0,DEV_reads=0);write(folder/('d3_interval_capacity_prefilter.json' if prefilter else 'd3_interval_capacity.json'),report);print(key,report['children_with_original_D3_witness'],'/',len(rows),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--folder-name',choices=['local_graph_arc_candidates_v1','local_graph_arc_extended_candidates_v1','union_nearest_arc_candidates_v1','union_graph_arc_seeds_v1','direction_union_arc_seeds_v1','direction_union_arc_candidates_v1','geometric_refined_arc_seeds_v1','multiscale_arc_seeds_v1','joint_fragment_arc_seeds_v1'],default='local_graph_arc_candidates_v1');p.add_argument('--prefilter',action='store_true');args=p.parse_args();run(args.key,args.folder_name,args.prefilter)
