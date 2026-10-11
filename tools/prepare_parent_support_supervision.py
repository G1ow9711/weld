"""Extend D3 supervision only within finite annotated geometry support.

A linear candidate whose endpoints lie inside a finite annotated line's convex
3 mm tube lies entirely inside that tube. This adds supervision for truncations
and erroneous cuts in an explicitly annotated span. It is not evidence of global
annotation completeness or material identity; outside-support candidates stay masked.
"""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.world_set_evaluation_285 import reference_segment
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/unified_parent_scoring_v1'
def run(key):
 f=BASE/('source_'+key);receipt=read(f/'receipt.json');prior=read(f/'d3_supervision.json');targetpath=ROOT/'server_rebuild/labels_v1'/key/'targets.json';assert digest(targetpath)==prior['target_sha256'];assert digest(f/'parents.json')==prior['parents_sha256']
 for name in ['child_segments.json','parent_children.json']:assert digest(f/name)==receipt['files'][name]
 truth=native_targets(read(targetpath));targets=[]
 for pi,p in enumerate(truth['parents']):
  for ci,c in enumerate(p['children']):
   ref=reference_segment(c)
   if ref['kind']=='line':targets.append((pi,ci,c['weld_type'],np.asarray(ref['points'])[[0,-1]]))
 bytype={}
 for wt in set(t[2] for t in targets):
  subset=[t for t in targets if t[2]==wt];ends=np.array([t[3] for t in subset]);v=ends[:,1]-ends[:,0];length=np.linalg.norm(v,axis=1);bytype[wt]=(subset,ends[:,0],v/length[:,None],length)
 segments=read(f/'child_segments.json');support=[]
 for s in segments:
  matches=[]
  if s['kind']=='line' and s['weld_type'] in bytype:
   subset,origin,direction,length=bytype[s['weld_type']];e=np.asarray(s['points'])[[0,-1]];delta=e[None,:,:]-origin[:,None,:];q=(delta*direction[:,None,:]).sum(2);closest=origin[:,None,:]+np.clip(q,0,length[:,None])[:,:,None]*direction[:,None,:];distance=np.linalg.norm(e[None,:,:]-closest,axis=2).max(1)
   for i in np.flatnonzero(distance<=3.):matches.append(dict(target_parent=subset[i][0],target_child=subset[i][1],max_finite_support_distance_mm=float(distance[i])))
  support.append(matches)
 children=read(f/'parent_children.json');rows=[];new=0
 for old,ids in zip(prior['rows'],children):
  row=dict(old);contained=all(bool(support[i]) for i in ids)
  row['prior_parent_loss_valid']=old['parent_loss_valid'];row['finite_annotated_line_support']=contained;row['parent_loss_valid']=old['parent_loss_valid'] or contained;new+=int(row['parent_loss_valid'] and not old['parent_loss_valid']);rows.append(row)
 report=dict(prior,rows=rows,prior_supervision_sha256=digest(f/'d3_supervision.json'),child_support=support,newly_supervised_parents=new,annotation_scope='Union of individually explicit finite annotated line spans within original D3 tolerance; no assertion outside that support',no_GT_mask_at_inference=True,material_identity_proved=False,source_sha256=digest(__file__))
 write(f/'d3_supervision_support_v2.json',report);print(dict(source=key,newly_supervised=new,still_unknown=sum(not r['parent_loss_valid'] for r in rows)),flush=True)
if __name__=='__main__':
 for k in ['21','BM79A_native_local','10','25','36']:run(k)
