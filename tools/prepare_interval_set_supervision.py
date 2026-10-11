"""Training-only canonical geometry assignment and feasible-set audit."""
import json,pathlib,sys,hashlib,shutil,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scipy.optimize import linear_sum_assignment
from weld_pipeline.server_interval_set import compile_plan,select,controls
from weld_pipeline.server_parent_chains import assemble
from weld_pipeline.world_set_evaluation_285 import score_source,aggregate_sources
base=ROOT/'server_rebuild';source=base/'span_head_v1';dest=base/'interval_set_v1';dest.mkdir(exist_ok=True);assert not (dest/'supervision_receipt.json').exists();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();receipt=json.loads((source/'receipt.json').read_text());summary=[];strict=[]
shutil.copyfile(__file__,dest/'supervision_snapshot.py');shutil.copyfile(ROOT/'weld_pipeline/server_interval_set.py',dest/'selector_snapshot.py')
for key,b in receipt['bindings'].items():
 rows=json.loads((source/(key+'_proposals.json')).read_text());matches=json.loads((source/(key+'_supervision.json')).read_text());t=json.loads((base/'labels_v1'/key/'targets.json').read_text());ids=[(pi,ci) for pi,p in enumerate(t['parents']) for ci,_ in enumerate(p['children'])];index={v:k for k,v in enumerate(ids)};cost=np.full((len(ids),len(rows)),1e6)
 for k,ms in enumerate(matches):
  for pi,ci,error in ms:cost[index[pi,ci],k]=error
 ii,jj=linear_sum_assignment(cost);gold=np.array([int(j) for i,j in zip(ii,jj) if cost[i,j]<3.000001],dtype=np.int64);utility=np.full(len(rows),-1.);utility[gold]=1.;picked=select(utility,compile_plan(rows));missing=set(gold)-set(picked);assert not missing,(key,'Gold not feasible',missing)
 np.savez_compressed(dest/(key+'_gold.npz'),indices=gold);cs=json.loads((base/'candidates_v8'/key/'candidates.json').read_text());decoded=assemble([rows[i] for i in picked],cs);targets=dict(t,parents=[dict(p,children=[dict(c,world_representation=c['representation']) for c in p['children']]) for p in t['parents']]);r=score_source(decoded,targets);strict.append(r);value=dict(key=key,G=len(ids),reachable_gold=len(gold),strict_parent_oracle=r['categories']['overall'],supervision_only=True,model_predictions=False,proposals_sha256=sha(source/(key+'_proposals.json')),matches_sha256=sha(source/(key+'_supervision.json')),features_sha256=sha(source/(key+'.npz')));summary.append(value);print(value,flush=True)
(dest/'supervision_receipt.json').write_text(json.dumps(dict(sources=summary,controls=controls(),strict_oracle=aggregate_sources(strict),model_predictions=False),indent=2))
