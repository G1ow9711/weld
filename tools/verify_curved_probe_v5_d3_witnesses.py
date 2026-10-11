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
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1'
def run(key):
 folder=BASE/'curved_upright_probe_v5'/key;binding=read(folder/'receipt.json');assert digest(folder/'profiles.json')==binding['profiles_sha256'];profiles=read(folder/'profiles.json');curves=[CurveCarrier(p['representation']) for p in profiles];targetpath=BASE/'labels'/key/'targets.json';target=native_targets(read(targetpath));rows=[]
 for pi,p in enumerate(target['parents']):
  for ci,c in enumerate(p['children']):
   if c['original_kind']!='arc':continue
   ref=reference_segment(c);closed=bool(p['is_closed'] and len(p['children'])==1);matches=[]
   for i,curve in enumerate(curves):
    q,d=curve.project(np.asarray(ref['points']))
    if float(d.max())>3.:continue
    lo,hi=sorted((float(q[0]),float(q[-1])))
    if not hi>lo or np.any((q<lo-1e-8)|(q>hi+1e-8)):continue
    s=sample_arc(CircularArc.from_parameters(curve.segment(lo,hi)['arc_parameters']));s['weld_type']=c['weld_type']
    for reverse in (False,True):
     ss=dict(s,**reverse_segment(s)) if reverse else s;error=child_error(ss,c,ref,closed,False)
     if error is not None:matches.append(dict(profile=i,interval_mm=[lo,hi],reverse=reverse,**error));break
   rows.append(dict(parent=pi,child=ci,matches=matches))
 report=dict(key=key,arc_children=len(rows),children_with_original_D3_witness=sum(bool(r['matches']) for r in rows),rows=rows,GT_projected_endpoints_offline_only=True,not_runtime_prediction=True,not_parent_assembly_or_contact_proof=True,profiles_sha256=digest(folder/'profiles.json'),target_sha256=digest(targetpath),fit_calls=0,DEV_reads=0);write(folder/'d3_interval_capacity.json',report);print(key,report['children_with_original_D3_witness'],'/',len(rows),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);run(p.parse_args().key)
