"""Actual TRAIN endpoint predictions, then post-freeze reference child quality."""
import sys,time,subprocess,traceback
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import read,write
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'balanced_training_endpoint_output_followup'
def main():
 OUT.mkdir(exist_ok=False);proc=Path('/proc/689663');identity=(proc/'stat').read_text().split()[21] if proc.exists() else None;deadline=time.monotonic()+10800;keys=read(BASE/'reference_query_task_v1/protocol.json')['training_keys']
 for key in keys:
  ready=BASE/'reference_detector_balanced_event_cache_v1'/key/'receipt.json'
  while not ready.exists():
   if identity is None or not proc.exists() or (proc/'stat').read_text().split()[21]!=identity:raise RuntimeError('Training input cache producer terminal without receipt')
   if time.monotonic()>deadline:raise TimeoutError('Observation expired; cache producer not stopped')
   time.sleep(5)
  for stage,script in [('output','infer_reference_detector_endpoint_output.py'),('quality','label_training_endpoint_reference_quality.py')]:
   print('START',key,stage,flush=True)
   with (OUT/(key+'_'+stage+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools'/script),'--arm','balanced','--key',key],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
   print('COMPLETE',key,stage,flush=True)
 write(OUT/'receipt.json',dict(status='SIX_TRAIN_ACTUAL_OUTPUTS_AND_REFERENCE_CHILD_QUALITY_READY',fit_calls=0,DEV_reads=0,scope='In-sample upstream predictions, not cross-fitted quality training evidence; no selector fitted or promoted'))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
