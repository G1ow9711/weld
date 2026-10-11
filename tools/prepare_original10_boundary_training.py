"""Recover original TRAIN vertical multi-child examples in new input-event schema."""
import subprocess,sys,traceback
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import write
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'server_rebuild/expansion_v1/original10_boundary_training_preparation'
def main():
 OUT.mkdir(exist_ok=False)
 jobs=[('events','prepare_original10_boundary_anchor_cache.py',[]),('supervision','prepare_original10_boundary_anchor_supervision.py',['--key','10']),('provenance','cache_original10_boundary_event_provenance.py',[])]
 for name,script,args in jobs:
  print('START',name,flush=True)
  with (OUT/(name+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools'/script),*args],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
  print('COMPLETE',name,flush=True)
 write(OUT/'receipt.json',dict(status='ORIGINAL10_TRAIN_INPUT_EVENTS_AND_ENDPOINT_SUPERVISION_READY',fit_calls=0,DEV_reads=0,no_synthetic_or_GT_created_events=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
