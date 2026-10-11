"""Supervised diagnostic only; never imported by feature extraction."""
import json,pathlib,sys,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from weld_pipeline.server_geometry_events import geometry,events
from weld_pipeline.world_set_evaluation_285 import reference_segment,distance_to_primitive
base=ROOT/'server_rebuild';summary=[]
for key in ['BM79A_native_local','10','25','36']:
 cs=json.loads((base/'candidates_v7'/key/'candidates.json').read_text());fs=json.loads((base/'candidates_v5'/key/'faces.json').read_text());ts=json.loads((base/'labels_v1'/key/'targets.json').read_text());audit=json.loads((base/'endpoint_head_v1'/(key+'_event_reachability.json')).read_text());cache=[]
 for c in cs:
  length,point,tangent,arc=geometry(c['representation']);s=events(c,fs);cache.append((length,point,tangent,arc,s,point(s),reference_segment(dict(world_representation=c['representation']))))
 rows=[]
 for row in audit['rows']:
  if row['reachable']:continue
  t=ts['parents'][row['parent']]['children'][row['child']];length,point,_,arc=geometry(t['representation']);q=point(np.array([0,length]));best=None
  for k,(c,(cl,cp,ct,ca,s,coords,ref)) in enumerate(zip(cs,cache)):
   if c['representation']['kind']!=t['original_kind']:continue
   carrier=float(distance_to_primitive(q,ref).max());e=np.linalg.norm(q[:,None,:]-coords[None,:,:],axis=2);ix=e.argmin(1);err=float(e[np.arange(2),ix].max())
   criterion=(carrier,err)
   if best is None or criterion<best[0]:best=(criterion,dict(candidate=k,carrier_endpoint_error=carrier,event_endpoint_errors=e[np.arange(2),ix].tolist(),nearest_events=coords[ix].tolist(),face_indices=c['face_indices'],candidate_length=cl))
  rows.append(dict(parent=row['parent'],child=row['child'],kind=t['original_kind'],length=length,endpoints=q.tolist(),best=best[1]))
 (base/'endpoint_head_v1'/(key+'_gap_diagnosis.json')).write_text(json.dumps(rows,indent=2))
 print(key,'unreachable',len(rows),'within_carrier',sum(r['best']['carrier_endpoint_error']<=3 for r in rows),'examples',json.dumps(rows[:3]),flush=True)
