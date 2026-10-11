"""Label frozen input events on all eight registered TRAIN families."""
import sys,time,subprocess,traceback
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import read,write
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'bounded_raw_training_supervision_followup'
def main():
 OUT.mkdir(exist_ok=False);proc=Path('/proc/747804');identity=(proc/'stat').read_text().split()[21] if proc.exists() else None;deadline=time.monotonic()+7200;rows=[]
 for key in read(BASE/'bounded_raw_training_inputs_v1/protocol.json')['keys']:
  while not (BASE/'bounded_raw_training_inputs_v1'/key/'receipt.json').exists():
   if identity is None or not proc.exists() or (proc/'stat').read_text().split()[21]!=identity:raise RuntimeError('Input producer terminal without case receipt')
   if time.monotonic()>deadline:raise TimeoutError('Observation expired; producer not stopped')
   time.sleep(5)
  print('START',key,flush=True)
  with (OUT/(key+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools/prepare_bounded_raw_training_supervision.py'),'--key',key],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  r=read(BASE/'bounded_raw_training_supervision_v1'/key/'receipt.json');rows.append(dict(key=key,children=r['line_reference_children'],both_available=r['children_with_both_endpoints_available'],queries=r['line_queries']));print('COMPLETE',rows[-1],flush=True)
 write(OUT/'receipt.json',dict(status='EIGHT_TRAIN_BOUNDED_RAW_EVENT_SUPERVISION_READY',rows=rows,fit_calls=0,DEV_reads=0))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
