"""Group line proposals on the same measured plane pair, retaining support gaps."""
import json,pathlib,sys,hashlib,time,shutil
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for key in ['21','BM79A_native_local','10','25','36']:
 start=time.perf_counter();base=ROOT/'server_rebuild';src=base/'candidates_v7'/key;out=base/'candidates_v8'/key;out.mkdir(parents=True,exist_ok=False)
 receipt=json.loads((src/'receipt.json').read_text())
 for n,h in receipt['files'].items():assert sha(src/n)==h
 faces=json.loads((base/'candidates_v5'/key/'faces.json').read_text());cs=json.loads((src/'candidates.json').read_text());groups={};result=[]
 for i,c in enumerate(cs):
  if c['representation']['kind']!='line':result.append(c);continue
  groups.setdefault(tuple(sorted(c['face_indices'])),[]).append((i,c))
 for pair,rows in groups.items():
  ns=np.array([faces[i]['normal'] for i in pair]);off=np.array([faces[i]['offset'] for i in pair]);v=np.cross(*ns);v/=np.linalg.norm(v)
  if v[np.argmax(abs(v))]<0:v=-v
  origin=ns.T@np.linalg.solve(ns@ns.T,off);spans=sorted([sorted((np.array(c['representation']['endpoints_mm'])-origin)@v) for _,c in rows]);merged=[]
  for lo,hi in spans:
   if merged and lo<=merged[-1][1]+1e-5:merged[-1][1]=max(merged[-1][1],float(hi))
   else:merged.append([float(lo),float(hi)])
  low,high=merged[0][0],merged[-1][1]
  result.append(dict(representation=dict(kind='line',endpoints_mm=[(origin+low*v).tolist(),(origin+high*v).tolist()]),face_indices=list(pair),length_mm=high-low,support_intervals_mm=[[a-low,b-low] for a,b in merged],unobserved_gaps_mm=[[merged[i][1]-low,merged[i+1][0]-low] for i in range(len(merged)-1)],source_candidates=[i for i,_ in rows],material_pair=None,weld_type=None,status='UNCERTAIN_CARRIER_WITH_EXPLICIT_SUPPORT_GAPS',requires_material_contact_classification=True))
 (out/'candidates.json').write_text(json.dumps(result,indent=2));shutil.copyfile(__file__,out/'builder_snapshot.py')
 r=dict(status='CARRIERS_COMPLETE',source_receipt_sha256=sha(src/'receipt.json'),input_sha256=receipt['input_sha256'],files={n:sha(out/n) for n in ['candidates.json','builder_snapshot.py']},GT_reads=0,STEP_reads=0,candidates=len(result),line_carriers=len(groups),gap_intervals=sum(len(c.get('unobserved_gaps_mm',[])) for c in result),seconds=time.perf_counter()-start,material_contact_not_proven=True)
 (out/'receipt.json').write_text(json.dumps(r,indent=2));print(key,len(result),'lines',len(groups),flush=True)
