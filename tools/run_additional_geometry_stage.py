"""Route unchanged legacy geometry stages to registered expansion TRAIN inputs."""
import argparse,importlib,time,traceback
from pathlib import Path
from threadpoolctl import threadpool_limits
from weld_pipeline.server_legacy_runtime import install
ADAPTER=install()
from weld_pipeline.sjg_local_runtime_904 import read,write,digest,check_resources
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1/frozen_features'
STAGES={'merge162':('merge_expansion_point_features','merge'),'merge26':('merge_expansion_run26_point_features','merge'),'instances':('rebuild_legacy_instances','prepare'),'finite':('rebuild_legacy_finite_support','prepare'),'surfaces':('rebuild_run26_learned_surfaces','run'),'unified':('rebuild_legacy_unified_candidates','run'),'union':('rebuild_legacy_joint_union','run')}
def run(key,stage):
 protocol=read(BASE.parent/'additional_batch01_protocol.json');assert key in {r['key'] for r in protocol['cases']};marks=BASE/'stage_receipts'/key;marks.mkdir(parents=True,exist_ok=True);marker=marks/(stage+'.json');assert not marker.exists();start=time.perf_counter()
 with threadpool_limits(limits=2):
  if stage in STAGES:
   name,func=STAGES[stage];m=importlib.import_module('tools.'+name);m.BASE=BASE;getattr(m,func)(key)
  elif stage=='rank2':
   m=importlib.import_module('tools.rebuild_legacy_rank2_candidates');m.BASE=BASE;case=read(BASE/'run26_merged_point_features'/('source_'+key)/'receipt.json');case['role']='TRAIN';folder=BASE/'rank2_candidates'/('source_'+key);folder.mkdir(parents=True,exist_ok=False)
   write(folder/'protocol.json',dict(source_sha256=digest(__file__),original_worker_sha256=digest(m.__file__),adapter=ADAPTER,GT_reads=0,STEP_reads=0,historical_parity_proven=False))
   def progress(*v):check_resources();write(folder/'progress.json',dict(event=v));print(key,*v,flush=True)
   write(folder/'controls.json',m.controls());profiles,row=m.generate(case,folder,progress);write(folder/'receipt.json',dict(status='ORIGINAL_RANK2_CANDIDATES_RECONSTRUCTED',historical_parity_proven=False,weld_accuracy_measured=False,**row))
  else:raise ValueError(stage)
 write(marker,dict(status='COMPLETE',key=key,stage=stage,seconds=time.perf_counter()-start,module_sha256=digest(m.__file__),runner_sha256=digest(__file__),GT_reads=0,STEP_reads=0))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--key',required=True);ap.add_argument('--stage',required=True,choices=list(STAGES)+['rank2']);a=ap.parse_args();run(a.key,a.stage)
