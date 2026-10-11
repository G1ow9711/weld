"""Offline separation of absent plane events and finite-extent rejection."""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'server_rebuild/expansion_v1';OUT=B/'finite_plane_endpoint_event_audit_v1'
reports=[]
for case in read(B/'coverage_cohort_protocol.json')['cases']:
 key=case['key'];folder=B/'frozen_features/legacy_runtime'/('source_'+key)/'run185'/('source_'+key);faces=read(folder/'surfaces.json');profiles=read(folder/'profiles.json');normals=np.asarray([f.get('normal',[0,0,0]) for f in faces],float);offset=np.asarray([f.get('offset',0) for f in faces],float);audit=read(OUT/key/'receipt.json');lookup={(r['parent'],r['child'],r['profile']):r for r in audit['records']};infinite=set();finite=set();all_lines=set();rows=[];cache={}
 for w in read(B/'positive_intervals'/key/'witnesses.json')['positive_witnesses']:
  pid=w['profile'];curve=CurveCarrier(profiles[pid]['representation'])
  if curve.kind!='line':continue
  owner=w['parent'],w['child'];all_lines.add(owner)
  if pid not in cache:
   slope=normals@curve.direction;ok=abs(slope)>1e-8;t=(offset[ok]-normals[ok]@curve.ends[0])/slope[ok];cache[pid]=np.r_[0.,curve.length,t[(t>=0)&(t<=curve.length)]]
  lo,hi=w['interval_mm'];d=[float(np.min(abs(cache[pid]-v))) for v in (lo,hi)];fd=lookup[(*owner,pid)]['nearest_endpoint_parameter_error_mm']
  if max(d)<=3:infinite.add(owner)
  if max(fd)<=3:finite.add(owner)
  rows.append(dict(parent=owner[0],child=owner[1],profile=pid,infinite_plane_event_errors_mm=d,finite_hull_event_errors_mm=fd))
 result=dict(key=key,witnessed_line_children=len(all_lines),line_children_with_any_infinite_plane_endpoint_pair_within_3mm=len(infinite),line_children_with_any_finite_hull_endpoint_pair_within_3mm=len(finite),infinite_supported_but_finite_rejected=len(infinite-finite),no_near_plane_pair_even_without_extent=len(all_lines-infinite),rows=rows,scope='Offline conditional endpoint parameter proximity on previously witnessed line carriers; infinite-plane events are never authorized as weld output or finite contact',fit_calls=0,DEV_reads=0);write(OUT/key/'loss_diagnosis.json',result);reports.append({k:v for k,v in result.items() if k!='rows'});print(reports[-1],flush=True)
write(OUT/'loss_diagnosis_summary.json',dict(cases=reports))
