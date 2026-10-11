"""Prepare TRAIN endpoint targets only after each immutable input cache is ready."""
import time,subprocess,sys,traceback
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'boundary_anchor_supervision_followup';PID=630402

def main():
 OUT.mkdir(exist_ok=False);proc=Path('/proc')/str(PID);identity=(proc/'stat').read_text().split()[21] if proc.exists() else None;deadline=time.monotonic()+3600;keys=read(BASE/'sampled_boundary_anchor_training_v1/protocol.json')['keys']
 for key in keys:
  ready=BASE/'sampled_boundary_anchor_training_v1'/key/'receipt.json'
  while not ready.exists():
   if identity is None or not proc.exists() or (proc/'stat').read_text().split()[21]!=identity:raise RuntimeError('Input event producer ended without this cache receipt; inspect failure, no restart')
   if time.monotonic()>deadline:raise TimeoutError('Observer deadline only; does not stop producer')
   time.sleep(10)
  done=BASE/'boundary_anchor_supervision_v1'/key/'receipt.json'
  if done.exists():
   r=read(done);assert r['events_receipt_sha256']==digest(ready);assert digest(done.parent/'targets.npz')==r['supervision_sha256']
  else:
   with (OUT/(key+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools/prepare_boundary_anchor_supervision.py'),'--key',key],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
  print('COMPLETE',key,flush=True)
 write(OUT/'receipt.json',dict(status='SIX_TRAIN_ENDPOINT_SUPERVISION_TABLES_COMPLETE',fit_calls=0,DEV_reads=0))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
