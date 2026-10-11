"""TRAIN-only link-head probe, never an inference or acceptance result."""
import argparse,pickle
from pathlib import Path
import numpy as np
from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.contract_geometry_d3_409 import child_error
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.sjg_child_parent_supervision_1255 import singleton,state_geometry,link_features
from weld_pipeline.sjg_parent_set_features_1160 import ParentRelations

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/best_baseline_restore'
def run(key,child_pool=False):
 case=BASE/'legacy_runtime'/('source_'+key);folder=case/'run379'/('test_'+key)
 decoded=read(folder/'decoded.json');metadata=read(folder/'parent_metadata.json');parents=[];md=[];origins=[]
 for pi,(p,m) in enumerate(zip(decoded['parents'],metadata)):
  for ci,s in enumerate(p['segments']):
   parents.append(singleton(s));md.append(dict(m,length_mm=s['length_mm'],line_fraction=float(s['kind']=='line')));origins.append((pi,ci))
 if child_pool:
  source=ROOT/'server_rebuild/child_residual_v1'/('source_'+key);receipt=read(source/'receipt.json')
  for name,sha in receipt['files'].items():assert digest(source/name)==sha
  parents=read(source/'parents.json')['parents'];md=read(source/'metadata.json');origins=[(r['parent'],r['child']) for r in read(source/'origins.json')]
 state=state_geometry(ParentRelations(parents,md))
 modelpath=ROOT/'weld_pipeline/reports/goal_contract_training_20261005/sjg_pilot_776/run446/models.pkl'
 assert digest(modelpath)==read(modelpath.parent/'training_receipt.json')['model_sha256']
 with modelpath.open('rb') as f:model=pickle.load(f)['models']['link']
 targetpath=ROOT/'server_rebuild/labels_v1'/key/'targets.json'
 assert digest(targetpath)==read(targetpath.parent/'receipt.json')['targets_sha256']
 truth=native_targets(read(targetpath));ev=read(case/'run446'/('test_'+key)/'evaluation.json')
 matched={(x['original_parent_index'],x['child_index']) for x in ev['matches']};results=[]
 for pi,t in enumerate(truth['parents']):
  if all((pi,ci) in matched for ci in range(len(t['children']))):continue
  if t['is_closed']:results.append(dict(parent=pi,status='CLOSED_NOT_PROBED'));continue
  options=[]
  for c in t['children']:
   ref=reference_segment(c);ends=np.asarray(ref['points'])[[0,-1]]
   error=np.linalg.norm(state['ends']-ends,axis=2).max(axis=1)
   options.append([int(i) for i in np.flatnonzero(error<=3.) if child_error(state['segments'][i],c,ref,False,False) is not None])
  admissible=[];steps=[]
  for ci,(aa,bb) in enumerate(zip(options,options[1:])):
   pairs=[(a,b) for a in aa for b in bb if a//2!=b//2];kind=1 if t['gaps'][ci]=='SHARED' else 2
   assert t['gaps'][ci] in ('SHARED','POSITIVE_GAP')
   probs=np.zeros((len(pairs),3))
   if pairs:probs[:,model.classes_.astype(int)]=model.predict_proba(link_features(state,pairs))
   edges={};legal_count=0;margins=[]
   for j,(a,b) in enumerate(pairs):
    distance=float(np.linalg.norm(state['ends'][a,1]-state['ends'][b,0]));legal=distance<=6.00001 if kind==1 else distance>1e-9
    if not legal:continue
    legal_count+=1;margin=float(np.log(max(probs[j,kind],1e-6))-np.log(max(probs[j,0],1e-6)));margins.append(margin)
    if margin>0:edges[a,b]=margin
   admissible.append(edges);steps.append(dict(child=ci,kind=kind,pairs=len(pairs),legal_pairs=legal_count,positive_pairs=len(edges),maximum_margin=max(margins,default=None)))
  witness=[]
  def visit(path):
   if len(path)==len(options):return path
   for b in options[len(path)]:
    if any(a//2==b//2 for a in path):continue
    if path and (path[-1],b) not in admissible[len(path)-1]:continue
    found=visit(path+[b])
    if found:return found
   return []
  if options:witness=visit([])
  results.append(dict(parent=pi,children=len(options),geometry_options=list(map(len,options)),link_steps=steps,positive_link_path_exists=bool(witness),witness=[dict(source_parent=origins[i//2][0],source_child=origins[i//2][1],reversed=bool(i%2)) for i in witness]))
 report=dict(source=key,role='TRAIN',model_sha256=digest(modelpath),target_sha256=digest(targetpath),results=results,GT_used_to_probe_only=True,global_optimum_and_material_contact_not_proven=True,original_parent_metadata_semantics_used=not child_pool,actual_child_fragment_metadata_used=child_pool,fit_calls=0,DEV_reads=0)
 dest=BASE/'capacity_diagnostics'/('source_'+key);write(dest/('residual_link_probe_child_metadata.json' if child_pool else 'residual_link_probe.json'),report);print(report,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--key',required=True,choices=['10','25','36']);ap.add_argument('--child-pool',action='store_true');args=ap.parse_args();run(args.key,args.child_pool)
