"""Prepare all-compatible TRAIN supervision once the existing raw pipeline finishes."""
import time,sys,subprocess,traceback
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import read,write
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'batch02_supervision_followup'
def main():
 OUT.mkdir(exist_ok=False);proc=Path('/proc/649890');identity=(proc/'stat').read_text().split()[21] if proc.exists() else None;ready=BASE/'additional_batch02_queue/receipt.json';deadline=time.monotonic()+7200
 while not ready.exists():
  if identity is None or not proc.exists() or (proc/'stat').read_text().split()[21]!=identity:raise RuntimeError('Raw pipeline producer terminal without completion receipt')
  if time.monotonic()>deadline:raise TimeoutError('Observation expired; producer not stopped')
  time.sleep(5)
 for case in read(BASE/'additional_batch02_protocol.json')['cases']:
  key=case['key']
  for stage,script in [('coverage','diagnose_all_compatible_batch02_carriers.py'),('intervals','prepare_all_compatible_batch02_intervals.py'),('queries','prepare_all_compatible_batch02_queries.py')]:
   print('START',key,stage,flush=True)
   with (OUT/(key+'_'+stage+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools'/script),'--key',key],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
   print('COMPLETE',key,stage,flush=True)
 with (OUT/'reference_task.log').open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools/prepare_reference_query_task_batch02.py')],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
 write(OUT/'receipt.json',dict(status='TWO_BATCH02_TRAIN_SUPERVISION_AND_REFERENCE_TASKS_COMPLETE',fit_calls=0,DEV_reads=0))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
