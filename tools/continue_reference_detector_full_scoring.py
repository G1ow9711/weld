"""Score every frozen detector-selected query through the unchanged endpoint model."""
import sys,time,subprocess,traceback
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'reference_detector_full_scoring_followup'
def main():
 OUT.mkdir(exist_ok=False);assert read(BASE/'compact_line_output_controls_v1/receipt.json')['status']=='PASS';proc=Path('/proc/667050');identity=(proc/'stat').read_text().split()[21] if proc.exists() else None;deadline=time.monotonic()+7200;reports=[]
 for key in ['new_cfc732e5f068','new_ceeae49d6777']:
  for arm in ['empirical','balanced']:
   ready=BASE/('reference_detector_'+arm+'_event_cache_v1')/key/'receipt.json'
   while not ready.exists():
    if identity is None or not proc.exists() or (proc/'stat').read_text().split()[21]!=identity:raise RuntimeError('Input cache producer terminal without expected receipt')
    if time.monotonic()>deadline:raise TimeoutError('Observation expired; producer not stopped')
    time.sleep(5)
   print('START',arm,key,flush=True)
   with (OUT/(arm+'_'+key+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools/infer_reference_detector_endpoint_output.py'),'--arm',arm,'--key',key],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
   output=BASE/('reference_detector_'+arm+'_endpoint_full_output_v1')/key;reports.append(dict(arm=arm,key=key,receipt=read(output/'receipt.json'),prediction_freeze_sha256=digest(output/'prediction_freeze.json')));print('COMPLETE',arm,key,flush=True)
 write(OUT/'receipt.json',dict(status='BOTH_REFERENCE_DETECTORS_FULL_ENDPOINT_D3_OUTPUTS_COMPLETE',cases=reports,no_automatic_promotion=True,DEV_reads=0))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
