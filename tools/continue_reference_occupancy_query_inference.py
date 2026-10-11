"""Wait for two TRAIN fits, then freeze two-family query predictions for both arms."""
import time,sys,subprocess,traceback
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import write
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'reference_detector_query_inference_followup'
def main():
 OUT.mkdir(exist_ok=False);proc=Path('/proc/662446');identity=(proc/'stat').read_text().split()[21] if proc.exists() else None;ready=BASE/'reference_occupancy_only_comparison_followup/receipt.json';deadline=time.monotonic()+3600
 while not ready.exists():
  if identity is None or not proc.exists() or (proc/'stat').read_text().split()[21]!=identity:raise RuntimeError('Training controller terminal without receipt')
  if time.monotonic()>deadline:raise TimeoutError('Observation expired; producer not stopped')
  time.sleep(5)
 for arm in ['empirical','balanced']:
  for key in ['new_cfc732e5f068','new_ceeae49d6777']:
   print('START',arm,key,flush=True)
   with (OUT/(arm+'_'+key+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools/infer_reference_occupancy_only_queries.py'),'--arm',arm,'--key',key],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
   print('COMPLETE',arm,key,flush=True)
 write(OUT/'receipt.json',dict(status='BOTH_DETECTORS_TWO_INPUT_QUERY_POOLS_FROZEN',target_labels_read=False,not_full_extraction=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
