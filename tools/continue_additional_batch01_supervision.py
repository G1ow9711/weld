"""Offline TRAIN scoring/positive supervision only after each prediction freeze."""
import time,subprocess,sys,traceback
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import read,write
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'additional_batch01_supervision';PRODUCER=Path('/proc/578565');STAGES=['evaluate_additional_native_capacity.py','diagnose_additional_carrier_coverage.py','prepare_additional_positive_intervals.py','prepare_additional_query_interval_training.py']
def main():
 OUT.mkdir(exist_ok=False);deadline=time.monotonic()+3600;start=(PRODUCER/'stat').read_text().split()[21]
 for case in read(BASE/'additional_batch01_protocol.json')['cases']:
  key=case['key'];ready=BASE/'frozen_features/legacy_runtime'/('source_'+key)/'select_reconstruction_receipt.json'
  while not ready.exists():
   live=PRODUCER.exists() and (PRODUCER/'stat').read_text().split()[21]==start
   if not live and not ready.exists():raise RuntimeError('Producer terminal before select receipt; inspect batch failure, no automatic restart')
   if time.monotonic()>deadline:raise TimeoutError('Observer budget expired; producer is not thereby terminal')
   time.sleep(10)
  for script in STAGES:
   print('START',key,script,flush=True)
   with (OUT/(key+'_'+script+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools'/script),'--key',key],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  write(OUT/(key+'_complete.json'),dict(key=key,status='OFFLINE_TRAIN_SUPERVISION_COMPLETE',new_fit_calls=0,DEV_reads=0));print('COMPLETE',key,flush=True)
 write(OUT/'receipt.json',dict(status='FOUR_CASE_SUPERVISION_COMPLETE',new_fit_calls=0,DEV_reads=0))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
