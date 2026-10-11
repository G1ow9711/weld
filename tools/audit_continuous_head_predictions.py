"""Measure endpoint-component reachability; not final weld extraction accuracy."""
import json,pathlib,sys,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from weld_pipeline.server_geometry_events import geometry
from weld_pipeline.world_set_evaluation_285 import reference_segment,distance_to_primitive
base=ROOT/'server_rebuild';out=base/'continuous_head_v1';receipt=json.loads((out/'receipt.json').read_text());pred=np.load(out/'TRAIN_query_outputs.npy',mmap_mode='r');cursor=0;summary=[]
for key,bind in receipt['bindings'].items():
 z=np.load(out/(key+'.npz'));meta=z['candidate_positions'];pp=pred[cursor:cursor+len(meta)];cursor+=len(meta);cs=json.loads((base/'candidates_v8'/key/'candidates.json').read_text());targets=json.loads((base/'labels_v1'/key/'targets.json').read_text());cache=[]
 boundaries=np.r_[0,np.flatnonzero(np.diff(meta[:,0]))+1,len(meta)]
 for i,(begin,end) in enumerate(zip(boundaries[:-1],boundaries[1:])):
  ci=int(meta[begin,0]);assert i==ci
  c=cs[ci];length,point,tangent,arc=geometry(c['representation']);roles=[]
  for role in range(4):
   keep=pp[begin:end,role]>=0;s=np.clip(meta[begin:end,1][keep]+pp[begin:end,4+role][keep],0,length);roles.append((s,point(s)))
  cache.append((point,arc,roles))
 rows=[]
 for pi,parent in enumerate(targets['parents']):
  for ci,t in enumerate(parent['children']):
   length,point,_,arc=geometry(t['representation']);q=point(np.array([0,length]));typ=int(t['weld_type']=='vertical');best=None;ref=reference_segment(dict(world_representation=t['representation']))
   for k,(c,(cp,ca,roles)) in enumerate(zip(cs,cache)):
    if c['representation']['kind']!=t['original_kind']:continue
    s0,p0=roles[2*typ];s1,p1=roles[2*typ+1]
    if not len(s0) or not len(s1):continue
    d0=np.linalg.norm(p0-q[0],axis=1);d1=np.linalg.norm(p1-q[1],axis=1);a=int(d0.argmin());b=int(d1.argmin())
    if max(d0[a],d1[b])>3:continue
    lo,hi=sorted([s0[a],s1[b]])
    if hi-lo<=1e-6:continue
    rep=dict(kind='line',endpoints_mm=cp(np.array([lo,hi])).tolist()) if ca is None else dict(kind='arc',arc_parameters=ca.subarc(float(lo),float(hi)).to_parameters())
    p=reference_segment(dict(world_representation=rep));error=max(float(distance_to_primitive(ref['points'],p).max()),float(distance_to_primitive(p['points'],ref).max()))
    if error<=3 and (best is None or error<best['geometry_error_mm']):best=dict(candidate=k,geometry_error_mm=error)
   rows.append(dict(parent=pi,child=ci,has_predicted_ordered_endpoint_pair=best is not None,diagnostic_witness=best))
 result=dict(key=key,G=targets['original_children'],children_with_predicted_endpoint_pair=sum(r['has_predicted_ordered_endpoint_pair'] for r in rows),rows=rows,positive_query_predictions=int((pp[:,:4]>=0).sum()),final_weld_set_decoded=False,FP_FN_of_welds_not_measured=True,parent_material_attributes_not_validated=True)
 (out/(key+'_predicted_pair_audit.json')).write_text(json.dumps(result,indent=2));summary.append({k:v for k,v in result.items() if k!='rows'});print(summary[-1],flush=True)
assert cursor==len(pred)
(out/'predicted_pair_summary.json').write_text(json.dumps(summary,indent=2))
