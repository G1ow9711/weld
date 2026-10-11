"""Offline upper bound for geometric event pairs, never model predictions."""
import json,pathlib,sys,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from weld_pipeline.server_geometry_events import geometry,events
from weld_pipeline.world_set_evaluation_285 import reference_segment,distance_to_primitive
from weld_pipeline.contract_curve_geometry import CircularArc
out=ROOT/'server_rebuild/endpoint_head_v1';results=[]
for key in ['21','BM79A_native_local','10','25','36']:
 base=ROOT/'server_rebuild';cs=json.loads((base/'candidates_v7'/key/'candidates.json').read_text());faces=json.loads((base/'candidates_v5'/key/'faces.json').read_text());t=json.loads((base/'labels_v1'/key/'targets.json').read_text());cache=[]
 for c in cs:
  length,point,_,arc=geometry(c['representation']);s=events(c,faces);cache.append((s,point(s),point,arc))
 rows=[]
 for pi,p in enumerate(t['parents']):
  for ci,c in enumerate(p['children']):
   length,point,_,arc=geometry(c['representation']);q=point(np.array([0,length]));best=None
   truth=reference_segment(dict(world_representation=c['representation']))
   for k,(raw,(s,coords,pred_point,pred_arc)) in enumerate(zip(cs,cache)):
    if raw['representation']['kind']!=c['original_kind']:continue
    dist=np.linalg.norm(q[:,None,:]-coords[None,:,:],axis=2);ii=dist.argmin(1);err=dist[np.arange(2),ii].max()
    if err>3 or ii[0]==ii[1]:continue
    lo,hi=sorted(s[ii]);rep=dict(kind='line',endpoints_mm=pred_point(np.array([lo,hi])).tolist()) if pred_arc is None else dict(kind='arc',arc_parameters=pred_arc.subarc(float(lo),float(hi)).to_parameters())
    pred=reference_segment(dict(world_representation=rep));whole=max(float(distance_to_primitive(truth['points'],pred).max()),float(distance_to_primitive(pred['points'],truth).max()))
    if whole<=3 and (best is None or whole<best['geometry_error_mm']):best=dict(candidate=k,event_indices=ii.tolist(),geometry_error_mm=whole)
   rows.append(dict(parent=pi,child=ci,reachable=best is not None,witness_for_audit_only=best))
 result=dict(key=key,G=t['original_children'],reachable=sum(r['reachable'] for r in rows),rows=rows,model_predictions=False,orientation_parent_material_attributes_ignored=True)
 (out/(key+'_event_reachability.json')).write_text(json.dumps(result,indent=2));results.append({k:v for k,v in result.items() if k!='rows'});print(results[-1],flush=True)
(out/'event_reachability_summary.json').write_text(json.dumps(results,indent=2))
