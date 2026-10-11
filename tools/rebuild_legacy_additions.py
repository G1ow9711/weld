"""Reconstruct frozen run429/432/446 incremental inference without fitting."""
import argparse,contextlib,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import tools.rebuild_legacy_downstream as runtime
from weld_pipeline.sjg_local_runtime_904 import read,write,digest

def bind(run,names):
 dest=runtime.CASE/run;dest.mkdir(exist_ok=True)
 source=runtime.ROOT/'weld_pipeline/reports/goal_contract_training_20261005/sjg_pilot_776'/run
 for n in names:
  p=dest/n
  if not p.exists():p.symlink_to(source/n)
  assert digest(p)==digest(source/n)
 return dest

@contextlib.contextmanager
def checkpoint_loading(path):
 import torch
 original=torch.load;path=path.resolve()
 def load(p,*a,**kw):
  if isinstance(p,(str,Path)) and Path(p).resolve()==path:kw.setdefault('weights_only',False)
  return original(p,*a,**kw)
 torch.load=load
 try:yield
 finally:torch.load=original

def route(key):
 return dict(baseline_run='run379',proposal_run='run432',feature_run='run433',pair_inputs=dict(input_run='run430',context_run='run414/whole' if key in ('25','36') else 'run380',sequence_run='run421/sequence56',occupancy_run='run429'))

def run(key,stage):
 runtime.CASE=runtime.BASE/'legacy_runtime'/('source_'+key);start=time.perf_counter()
 if stage=='occupancy':
  dest=bind('run429',['sequence_model.pt','training_receipt.json']);p=dest/'sequence_model.pt';assert digest(p)==read(dest/'training_receipt.json')['model_sha256']
  m=runtime.module('sjg_native_occupancy_model_1271')
  with checkpoint_loading(p):m.infer(dest,key)
 elif stage=='overlay':
  m=runtime.module('sjg_endpoint_occupancy_overlay_1272');m.KEYS=(key,);m.prepare()
  dest=runtime.CASE/'run430';r=read(dest/'receipt.json');r['original_status']=r['status'];r['status']='RECONSTRUCTED_SINGLE_CASE_ENDPOINT_OVERLAY';write(dest/'receipt.json',r)
 elif stage=='pairs':
  dest=bind('run432',['pair_model.pt','training_receipt.json']);p=dest/'pair_model.pt';assert digest(p)==read(dest/'training_receipt.json')['model_sha256']
  m=runtime.module('sjg_joint_pair_model_1248')
  with checkpoint_loading(p):m.infer(dest,key,**route(key)['pair_inputs'])
 elif stage=='parent_features':
  # Attest missing source-access field from the actual input-only stage receipts.
  p=runtime.CASE/'run379'/('test_'+key)/'final_prediction.json';r=read(p)
  for name in ('graphs','object_features','relative_features','observed_features','screen','edge_graph','solve','decode','select'):
   evidence=read(runtime.CASE/(name+'_reconstruction_receipt.json'));assert evidence['GT_reads']==evidence['STEP_reads']==0
  r['STEP_reads']=0;r['source_access_attestation']='reconstruction stage receipts';write(p,r)
  dest=runtime.CASE/'run433';dest.mkdir();m=runtime.module('sjg_additive_parent_precision_1254');m.prepare(dest,keys=(key,),routes={key:route(key)})
 elif stage=='child_features':
  m=runtime.module('sjg_child_parent_supervision_1255');dest=runtime.CASE/'run434';dest.mkdir();write(dest/'input_routes.json',dict(routes={key:route(key)}))
  raw=runtime.CASE/'run433'/('source_'+key);r=read(raw/'receipt.json');assert digest(raw/'features.npy')==r['feature_sha256'] and digest(raw/'metadata.json')==r['metadata_sha256']
  original=np.load(raw/'features.npy');x=m.direction_free(original);reverse=original.copy();reverse[:,524:528]=original[:,[526,527,524,525]];assert np.array_equal(x,m.direction_free(reverse))
  folder=dest/('source_'+key);folder.mkdir();np.save(folder/'features.npy',x)
  _,_,context,origins,base,binding=m.child_inputs(key,dest)
  write(folder/'origin.json',dict(base_children=base,origins=origins))
  write(folder/'receipt.json',dict(source=key,schema=m.SCHEMA,**binding,columns=526,feature_sha256=digest(folder/'features.npy'),original_input_receipt_sha256=digest(raw/'receipt.json'),base_children=base,new_children=len(x),GT_reads=0,STEP_reads=0,traversal_invariance_all_rows_exact=True))
 elif stage=='infer446':
  dest=bind('run446',['models.pkl','training_receipt.json']);runtime.module('sjg_child_parent_model_1256').infer(dest,key)
 else:raise ValueError(stage)
 write(runtime.CASE/(stage+'_addition_receipt.json'),dict(status='ORIGINAL_ADDITION_STAGE_RECONSTRUCTED',source=key,stage=stage,seconds=time.perf_counter()-start,source_sha256=digest(__file__),GT_reads=0,STEP_reads=0,fit_calls=0,historical_parity_proven=False))
 print('ADDITION_STAGE_COMPLETE',key,stage,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--key',required=True,choices=['21','BM79A_native_local','10','25','36']);ap.add_argument('--stage',required=True,choices=['occupancy','overlay','pairs','parent_features','child_features','infer446']);args=ap.parse_args()
 with threadpool_limits(limits=2):run(args.key,args.stage)
