"""Matched training6 versus training6+original10 after active resource pipeline."""
import sys,time,subprocess,traceback
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'boundary_provenance_mixed_comparison_followup'
def main():
 OUT.mkdir(exist_ok=False);assert (BASE/'original10_all_boundary_training_preparation/receipt.json').exists();proc=Path('/proc/649890');identity=(proc/'stat').read_text().split()[21] if proc.exists() else None;ready=BASE/'additional_batch02_queue/receipt.json';deadline=time.monotonic()+14400
 write(OUT/'protocol.json',dict(source_model_sha256=read(BASE/'boundary_event_provenance_selector_v1/receipt.json')['model_sha256'],arms=['control6','mixed7'],extra_optimizer_steps_per_arm=16000,same_source_normalization=True,old_detector_frozen=True,raw_original10_DATA_ROLE='TRAIN',new_batch02_not_in_this_comparison=True,DEV_reads=0,no_automatic_promotion=True))
 while not ready.exists():
  if identity is None or not proc.exists() or (proc/'stat').read_text().split()[21]!=identity:raise RuntimeError('Active batch02 producer terminal without receipt; inspect resource/failure before starting')
  if time.monotonic()>deadline:raise TimeoutError('Observation expired; producer not stopped')
  time.sleep(5)
 for arm,flags in [('control6',[]),('mixed7',['--include-original10'])]:
  jobs=[('train','train_boundary_provenance_mixed_comparison.py',flags),('fit','evaluate_boundary_provenance_mixed_fit.py',flags)]
  jobs.extend((key,'infer_boundary_provenance_mixed_selector.py',['--key',key,*flags]) for key in ('new_cfc732e5f068','new_ceeae49d6777'))
  for stage,script,args in jobs:
   print('START',arm,stage,flush=True)
   with (OUT/(arm+'_'+stage+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools'/script),*args],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
   print('COMPLETE',arm,stage,flush=True)
  model=BASE/('boundary_provenance_'+arm+'_v1');cases=[]
  for key in ('new_cfc732e5f068','new_ceeae49d6777'):
   folder=BASE/('boundary_provenance_'+arm+'_full_output_v1')/key;cases.append(dict(key=key,rows=read(folder/'receipt.json')['rows'],prediction_freeze_sha256=digest(folder/'prediction_freeze.json')))
  write(model/'decision.json',dict(status='EXPERIMENTAL_NOT_PROMOTED',cases=cases,reason='Matched TRAIN structure experiment only; complete contract gates and protected DEV still required',DEV_reads=0))
 write(OUT/'receipt.json',dict(status='BOTH_ARMS_TRAIN_FIT_AND_FULL_OUTPUTS_COMPLETE',no_automatic_promotion=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
