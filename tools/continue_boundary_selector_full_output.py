"""Wait for the live input cache producer, then evaluate actual selector output."""
import time,sys,subprocess,traceback
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import read,write
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'boundary_selector_full_output_followup';PID=637886

def main():
 OUT.mkdir(exist_ok=False);proc=Path('/proc')/str(PID);identity=(proc/'stat').read_text().split()[21] if proc.exists() else None;assert (BASE/'boundary_event_selector_v1/receipt.json').exists(),'Verify training completion before launching';deadline=time.monotonic()+3600
 for key in ('new_cfc732e5f068','new_ceeae49d6777'):
  ready=BASE/'runtime_boundary_anchor_cache_v1'/key/'receipt.json'
  while not ready.exists():
   if identity is None or not proc.exists() or (proc/'stat').read_text().split()[21]!=identity:raise RuntimeError('Input event producer terminal without receipt')
   if time.monotonic()>deadline:raise TimeoutError('Observation expired, not producer terminal')
   time.sleep(10)
  print('START',key,flush=True)
  with (OUT/(key+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools/infer_boundary_event_selector.py'),'--key',key],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
  print('COMPLETE',key,flush=True)
 write(OUT/'receipt.json',dict(status='BOTH_FULL_OUTPUTS_SCORED',no_automatic_promotion=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
