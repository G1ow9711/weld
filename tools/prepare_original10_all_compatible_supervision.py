"""Expand original10 TRAIN carrier witnesses; preserve previous closest-three data."""
import subprocess,sys,traceback
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import write
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'server_rebuild/expansion_v1/original10_all_compatible_followup'
def main():
 OUT.mkdir(exist_ok=False)
 for stage,script in [('coverage','diagnose_all_compatible_original10_carriers.py'),('intervals','prepare_all_compatible_original10_intervals.py'),('queries','prepare_all_compatible_original10_queries.py')]:
  print('START',stage,flush=True)
  with (OUT/(stage+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools'/script),'--key','10'],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
  print('COMPLETE',stage,flush=True)
 write(OUT/'receipt.json',dict(status='ORIGINAL10_ALL_COMPATIBLE_TRAIN_SUPERVISION_COMPLETE',fit_calls=0,DEV_reads=0))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
