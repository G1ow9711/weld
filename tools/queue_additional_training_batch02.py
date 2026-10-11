"""Serialize additional TRAIN preparation after the running provenance experiment."""
import sys,time,subprocess,traceback
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import write
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'additional_batch02_queue'
def main():
 OUT.mkdir(exist_ok=False);proc=Path('/proc/645479');identity=(proc/'stat').read_text().split()[21] if proc.exists() else None;ready=BASE/'boundary_provenance_experiment_followup/receipt.json';deadline=time.monotonic()+7200
 while not ready.exists():
  if identity is None or not proc.exists() or (proc/'stat').read_text().split()[21]!=identity:raise RuntimeError('Provenance producer terminal without receipt; inspect failure before starting new resource use')
  if time.monotonic()>deadline:raise TimeoutError('Observation expired; original producer not stopped')
  time.sleep(5)
 print('START additional_batch02',flush=True)
 with (OUT/'pipeline.log').open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools/run_additional_training_batch02.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 write(OUT/'receipt.json',dict(status='TWO_TRAIN_PIPELINES_COMPLETED',new_model_fit_calls=0,DEV_reads=0))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
