"""Offline witnessed-carrier endpoint representation audit, not full inference."""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.server_face_boundary_endpoint_events import FaceBoundaryEvents as FaceEvents
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.world_set_runtime_285 import line_segment
from weld_pipeline.contract_curve_geometry import CircularArc,sample_arc
from weld_pipeline.contract_geometry_d3_409 import child_error
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';DEST=BASE/'face_boundary_endpoint_event_audit_v1'
def main():
 DEST.mkdir(exist_ok=False)
 for case in read(BASE/'coverage_cohort_protocol.json')['cases']:
  key=case['key'];folder=BASE/'frozen_features/legacy_runtime'/('source_'+key)/'run185'/('source_'+key);faces=read(folder/'surfaces.json');profiles=read(folder/'profiles.json');xyz=np.load(BASE/'inputs'/key/'xyz_mm.npy',mmap_mode='r');freeze=read(folder/'input_freeze.json');assert digest(BASE/'inputs'/key/'xyz_mm.npy')==freeze['input_sha256']
  for n in ['surfaces.json','surface_support.npz']:assert digest(folder/n)==freeze['files'][n]
  with np.load(folder/'surface_support.npz') as s:generator=FaceEvents(xyz,faces,s)
  witnesses=read(BASE/'positive_intervals'/key/'witnesses.json')['positive_witnesses'];truth=native_targets(read(BASE/'labels'/key/'targets.json'));cache={};hits={};records=[]
  for w in witnesses:
   pid=w['profile'];pi,ci=w['parent'],w['child'];curve=CurveCarrier(profiles[pid]['representation'])
   if pid not in cache:cache[pid]=generator.events(curve)
   q,faceids=cache[pid];refchild=truth['parents'][pi]['children'][ci];ref=reference_segment(refchild);closed=bool(truth['parents'][pi]['is_closed'] and len(truth['parents'][pi]['children'])==1);lo,hi=w['interval_mm'];left=np.unique(q[abs(q-lo)<=3]);right=np.unique(q[abs(q-hi)<=3]);left=sorted(left,key=lambda t:abs(t-lo))[:5];right=sorted(right,key=lambda t:abs(t-hi))[:5];found=None
   for a in left:
    if found:break
    for b in right:
     if b<=a:continue
     rep=curve.segment(float(a),float(b));s=line_segment(*rep['endpoints_mm']) if curve.kind=='line' else sample_arc(CircularArc.from_parameters(rep['arc_parameters']));s['weld_type']=w['weld_type']
     if any(child_error(v,refchild,ref,closed,False) is not None for v in (s,dict(s,**reverse_segment(s)))):found=[float(a),float(b)];break
   if found:hits[pi,ci]=dict(profile=pid,event_interval_mm=found)
   records.append(dict(parent=pi,child=ci,profile=pid,event_count=len(q),nearest_endpoint_parameter_error_mm=[float(np.min(abs(q-lo))),float(np.min(abs(q-hi)))],passing_interval=found))
  dest=DEST/key;dest.mkdir();write(dest/'receipt.json',dict(key=key,GT_children=sum(len(p['children']) for p in truth['parents']),children_with_input_event_D3_witness=len(hits),witnessed_profiles=len(cache),finite_face_hulls=len(generator.domains),scope='GT selects prior witnessed profiles for offline diagnosis only; events on each profile depend solely on input geometry. Tests closest five left/right events, found capacity not exhaustive upper bound. Convex extent cannot certify holes/material/weldability.',not_full_runtime=True,GT_endpoints_not_inserted_into_events=True,fit_calls=0,DEV_reads=0,hits=[dict(parent=p,child=c,**v) for (p,c),v in hits.items()],records=records));print(key,len(hits),'/',sum(len(p['children']) for p in truth['parents']),flush=True);xyz._mmap.close()
if __name__=='__main__':main()
