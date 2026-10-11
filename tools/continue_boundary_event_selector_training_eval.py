"""Observe this continuation, then score unchanged protocol; never promote automatically."""
import os,sys,time,subprocess,traceback
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'boundary_event_selector_continuation_followup';MODEL=BASE/'boundary_event_selector_continuation_v1'
def main():
 OUT.mkdir(exist_ok=False);proc=Path('/proc/640452');identity=(proc/'stat').read_text().split()[21] if proc.exists() else None;deadline=time.monotonic()+3600
 while not (MODEL/'receipt.json').exists():
  if identity is None or not proc.exists() or (proc/'stat').read_text().split()[21]!=identity:raise RuntimeError('Training producer exited without receipt')
  if time.monotonic()>deadline:raise TimeoutError('Observation expired; producer not terminated')
  time.sleep(5)
 receipt=read(MODEL/'receipt.json');assert digest(MODEL/'model.pt')==receipt['model_sha256']
 jobs=[('training_fit','evaluate_boundary_selector_continuation_fit.py',[])]
 jobs.extend((key,'infer_boundary_event_selector_continuation.py',['--key',key]) for key in ('new_cfc732e5f068','new_ceeae49d6777'))
 for name,script,args in jobs:
  print('START',name,flush=True)
  with (OUT/(name+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools'/script),*args],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
  print('COMPLETE',name,flush=True)
 cases=[]
 for key in ('new_cfc732e5f068','new_ceeae49d6777'):
  result=BASE/'boundary_event_selector_continuation_full_output_v1'/key
  cases.append({'key':key,'rows':read(result/'receipt.json')['rows'],'prediction_freeze_sha256':digest(result/'prediction_freeze.json')})
 write(MODEL/'decision.json',dict(status='EXPERIMENTAL_NOT_PROMOTED',cases=cases,reason='Full-output evidence retained; no automatic acceptance or promotion; all contract gates and protected DEV remain required',DEV_reads=0))
 write(OUT/'receipt.json',dict(status='TRAIN_FIT_AND_TWO_FULL_OUTPUTS_COMPLETE',model_sha256=receipt['model_sha256'],no_automatic_promotion=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
