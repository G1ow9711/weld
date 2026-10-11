import json,pathlib,sys,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scipy.optimize import linear_sum_assignment
from weld_pipeline.server_parent_chains import assemble
from weld_pipeline.server_geometry_events import geometry
base=ROOT/'server_rebuild';summary=[]
for key in ['21','BM79A_native_local','10','25','36']:
 src=base/'span_head_v1';rows=json.loads((src/(key+'_proposals.json')).read_text());matches=json.loads((src/(key+'_supervision.json')).read_text());t=json.loads((base/'labels_v1'/key/'targets.json').read_text());cs=json.loads((base/'candidates_v8'/key/'candidates.json').read_text());ids=[(pi,ci) for pi,p in enumerate(t['parents']) for ci,_ in enumerate(p['children'])];index={v:k for k,v in enumerate(ids)};cost=np.full((len(ids),len(rows)),1e6)
 for k,ms in enumerate(matches):
  for pi,ci,error in ms:cost[index[pi,ci],k]=error
 ii,jj=linear_sum_assignment(cost);assignment={ids[i]:int(j) for i,j in zip(ii,jj) if cost[i,j]<3.000001};gold=sorted(assignment.values());decoded=assemble([rows[i] for i in gold],cs);actual={(gold[e['source']],gold[e['target']]) for e in decoded['links']};expected=set();miss=[]
 for (pi,ci),a in assignment.items():
  b=assignment.get((pi,ci+1))
  if b is None:continue
  expected.add((a,b))
  if (a,b) not in actual:
   La,pa,ta,_=geometry(rows[a]['representation']);Lb,pb,tb,_=geometry(rows[b]['representation']);fa=cs[rows[a]['candidate']]['face_indices'];fb=cs[rows[b]['candidate']]['face_indices'];miss.append(dict(parent=pi,child=ci,distance=float(np.linalg.norm(pa(La)-pb(0))),tangent_dot=float(ta(La)@tb(0)),common_faces=sorted(set(fa)&set(fb)),face_pair_a=fa,face_pair_b=fb))
 value=dict(key=key,expected=len(expected),correct=len(expected&actual),missing=miss,wrong_links=len(actual-expected));summary.append(value);print(value,flush=True)
 (base/'interval_set_v1'/(key+'_canonical_assignment.json')).write_text(json.dumps([dict(parent=pi,child=ci,proposal=j) for (pi,ci),j in assignment.items()],indent=2))
(base/'interval_set_v1'/'parent_link_diagnosis.json').write_text(json.dumps(summary,indent=2))
