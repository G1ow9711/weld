"""Input-only feature extraction on frozen outputs; held labels never read."""
import sys,time,subprocess,traceback
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'endpoint_candidate_quality_feature_followup'
def main():
 OUT.mkdir(exist_ok=False);controls=read(BASE/'endpoint_candidate_quality_features_v1/controls.json');assert controls['status']=='PASS' and controls['module_sha256']==digest(ROOT/'weld_pipeline/server_endpoint_candidate_quality_features.py');proc=Path('/proc/691475');identity=(proc/'stat').read_text().split()[21] if proc.exists() else None;deadline=time.monotonic()+10800;keys=['new_cfc732e5f068','new_ceeae49d6777']+read(BASE/'reference_query_task_v1/protocol.json')['training_keys']
 for key in keys:
  freeze=BASE/'reference_detector_balanced_endpoint_full_output_v1'/key/'prediction_freeze.json'
  while not freeze.exists():
   if identity is None or not proc.exists() or (proc/'stat').read_text().split()[21]!=identity:raise RuntimeError('TRAIN output controller terminal without freeze')
   if time.monotonic()>deadline:raise TimeoutError('Observation expired; producer not stopped')
   time.sleep(5)
  dest=BASE/'endpoint_candidate_quality_features_v1'/key
  if (dest/'receipt.json').exists():
   r=read(dest/'receipt.json');assert r['prediction_freeze_sha256']==digest(freeze);assert r['module_sha256']==controls['module_sha256'];assert digest(dest/'features.npy')==r['features_sha256'];print('REUSE_COMPLETE',key,flush=True);continue
  print('START',key,flush=True)
  with (OUT/(key+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools/prepare_endpoint_candidate_quality_features.py'),'--key',key],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
  print('COMPLETE',key,flush=True)
 write(OUT/'receipt.json',dict(status='SIX_TRAIN_AND_TWO_HELD_INPUT_QUALITY_FEATURES_COMPLETE',GT_reads=0,fit_calls=0,DEV_reads=0,no_selector_fitted=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
