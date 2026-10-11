"""Post-freeze distinguish detector coverage from endpoint/parent errors."""
import sys,time,subprocess,traceback
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import write
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'reference_detector_diagnostics_followup'
def main():
 OUT.mkdir(exist_ok=False);proc=Path('/proc/673458');identity=(proc/'stat').read_text().split()[21] if proc.exists() else None;deadline=time.monotonic()+7200
 for key in ['new_cfc732e5f068','new_ceeae49d6777']:
  for arm in ['empirical','balanced']:
   ready=BASE/('reference_detector_'+arm+'_endpoint_full_output_v1')/key/'prediction_freeze.json'
   while not ready.exists():
    if identity is None or not proc.exists() or (proc/'stat').read_text().split()[21]!=identity:raise RuntimeError('Inference controller terminal without freeze')
    if time.monotonic()>deadline:raise TimeoutError('Observation expired; producer not stopped')
    time.sleep(5)
   for stage,script in [('coverage','audit_new_reference_detector_child_recall.py'),('endpoints','evaluate_new_detector_endpoint_child_capacity.py')]:
    print('START',arm,key,stage,flush=True)
    with (OUT/(arm+'_'+key+'_'+stage+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools'/script),'--arm',arm,'--key',key],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
    print('COMPLETE',arm,key,stage,flush=True)
 write(OUT/'receipt.json',dict(status='FOUR_POST_FREEZE_DETECTOR_AND_ENDPOINT_DIAGNOSTICS_COMPLETE',fit_calls=0,threshold_changes=0,DEV_reads=0))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
