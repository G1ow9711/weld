"""Original 1261 input stages under isolated reconstruction roots."""
import argparse,time
from pathlib import Path
from threadpoolctl import threadpool_limits
import tools.rebuild_legacy_downstream as runtime
from weld_pipeline.sjg_local_runtime_904 import read,write,digest

STAGES={
 'sequence28':('sjg_dense_joint_sequence_1237','prepare'),
 'sequence52':('sjg_global_neighborhood_sequence_1239','prepare'),
 'sequence56':('sjg_finite_extent_sequence_1240','prepare'),
 'occupancy':('sjg_structured_sequence_1241','infer'),
 'eventsplane':('sjg_geometric_endpoint_events_1242','prepare'),
 'eventssupport':('sjg_fast_observed_boundaries_1266','prepare_support'),
 'eventsmultiscale':('sjg_fast_observed_boundaries_1266','prepare_multiscale'),
 'eventslegacy':('sjg_legacy_endpoint_neighborhood_1250','prepare'),
 'endpoints':('sjg_endpoint_role_features_1245','prepare')}

def run(key,stage):
 runtime.CASE=runtime.BASE/'legacy_runtime'/('source_'+key)
 whole='run414/whole' if key in ('25','36') else 'run380'
 legacy='run413' if key in ('25','36') else 'run379'
 if legacy=='run413' and not (runtime.CASE/legacy).exists():(runtime.CASE/legacy).symlink_to(runtime.CASE/'run379',target_is_directory=True)
 kwargs={
 'sequence28':dict(admission_run='run420',whole_run=whole),
 'sequence52':dict(base_run='run421/sequence28'),
 'sequence56':dict(base_run='run421/sequence52'),
 'occupancy':dict(input_run='run421/sequence56',whole_run=whole),
 'eventsplane':dict(sequence_run='run421/sequence56'),
 'eventssupport':dict(base_run='run421/eventsplane'),
 'eventsmultiscale':dict(base_run='run421/eventssupport',legacy_run=legacy),
 'eventslegacy':dict(base_run='run421/eventsmultiscale',legacy_run=legacy),
 'endpoints':dict(event_run='run421/eventslegacy',sequence_run='run421/sequence56',occupancy_run='run421/occupancy')}[stage]
 if stage=='endpoints':
  alias=runtime.CASE/'run392';origin=runtime.CASE/'run421/occupancy'
  receipt=read(origin/'training_receipt.json');assert digest(origin/'sequence_model.pt')==receipt['model_sha256']
  if not alias.exists():alias.symlink_to(origin,target_is_directory=True)
 dest=runtime.CASE/'run421'/stage;dest.mkdir(parents=True,exist_ok=True)
 name,method=STAGES[stage];m=runtime.module(name);start=time.perf_counter()
 if stage=='occupancy':
  source=runtime.ROOT/'weld_pipeline/reports/goal_contract_training_20261005/sjg_pilot_776/run421/occupancy'
  for n in ['sequence_model.pt','training_receipt.json']:
   link=dest/n
   if not link.exists():link.symlink_to(source/n)
   assert digest(link)==digest(source/n)
  import torch
  checkpoint=(dest/'sequence_model.pt').resolve();assert digest(checkpoint)==read(dest/'training_receipt.json')['model_sha256']
  original_load=torch.load
  def verified_load(path,*a,**kw):
   if isinstance(path,(str,Path)) and Path(path).resolve()==checkpoint:kw.setdefault('weights_only',False)
   return original_load(path,*a,**kw)
  torch.load=verified_load
  try:m.infer(dest,key,**kwargs)
  finally:torch.load=original_load
 else:
  control=getattr(m,'controls',None)
  if control is not None:control(dest)
  getattr(m,method)(key,dest,**kwargs)
 out=dest/(('proposal_' if stage=='occupancy' else 'source_')+key)/('prediction_receipt.json' if stage=='occupancy' else 'receipt.json')
 receipt=read(out);assert receipt['GT_reads']==receipt['STEP_reads']==0
 write(runtime.CASE/'run421'/('stage_'+stage+'_'+key+'.json'),dict(status='COMPLETE',stage=stage,source=key,kwargs=kwargs,output_receipts={str(out.relative_to(runtime.CASE/'run421')):digest(out)},module_sha256=digest(m.__file__),source_sha256=digest(__file__),seconds=time.perf_counter()-start,fit_calls=0,GT_reads=0,STEP_reads=0,historical_parity_proven=False))
 print('REBUILT_SEQUENCE_STAGE',key,stage,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--key',required=True,choices=['21','BM79A_native_local','10','25','36']);ap.add_argument('--stage',required=True,choices=list(STAGES));args=ap.parse_args()
 with threadpool_limits(limits=2):run(args.key,args.stage)
