"""Complete matched six/eight TRAIN endpoint fits, preserve all other experiments."""
import sys,time,subprocess,traceback
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import read,write
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'boundary_provenance_batch02_comparison_followup'
def wait_for(path,pid):
 proc=Path('/proc')/str(pid);identity=(proc/'stat').read_text().split()[21] if proc.exists() else None;deadline=time.monotonic()+14400
 while not path.exists():
  if identity is None or not proc.exists() or (proc/'stat').read_text().split()[21]!=identity:raise RuntimeError('Producer terminal without receipt: '+str(pid))
  if time.monotonic()>deadline:raise TimeoutError('Observation expired without stopping producer')
  time.sleep(5)
def main():
 OUT.mkdir(exist_ok=False);write(OUT/'protocol.json',dict(arms=['control6','expanded8'],warm_model=read(BASE/'boundary_event_provenance_selector_v1/receipt.json')['model_sha256'],extra_steps_each=16000,source_normalization_frozen=True,original10_excluded=True,HELD_reads=0,DEV_reads=0,full_output_scoring_pending=True))
 wait_for(BASE/'boundary_provenance_batch02_control6_v1/receipt.json',723967)
 jobs=[('control6_fit','evaluate_boundary_provenance_batch02_fit.py',[]),('expanded8_train','train_boundary_provenance_batch02_comparison.py',['--include-batch02']),('expanded8_fit','evaluate_boundary_provenance_batch02_fit.py',['--include-batch02'])]
 for name,script,args in jobs:
  if name=='expanded8_train':wait_for(BASE/'batch02_boundary_training_preparation/receipt.json',715181)
  print('START',name,flush=True)
  with (OUT/(name+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools'/script),*args],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
  print('COMPLETE',name,flush=True)
 write(OUT/'receipt.json',dict(status='MATCHED_SIX_EIGHT_TRAIN_FITS_COMPLETE_NOT_PROMOTED',full_output_scoring_pending=True,DEV_reads=0,HELD_reads=0))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
