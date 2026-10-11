"""Small GPU reference-only comparison while independent CPU geometry continues."""
import sys,subprocess,traceback
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import write
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'server_rebuild/expansion_v1/reference_occupancy_only_comparison_followup'
def main():
 OUT.mkdir(exist_ok=False)
 for arm in ['empirical','balanced']:
  for stage,script in [('train','train_reference_occupancy_only_comparison.py'),('fit','evaluate_reference_occupancy_only_training.py')]:
   print('START',arm,stage,flush=True)
   with (OUT/(arm+'_'+stage+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools'/script),'--arm',arm],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
   print('COMPLETE',arm,stage,flush=True)
 write(OUT/'receipt.json',dict(status='TWO_REFERENCE_ONLY_TRAIN_FITS_COMPLETE',full_extraction_evaluated=False,no_automatic_promotion=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
