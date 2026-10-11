"""Follow the verified training process, then score complete frozen query output."""
import time,sys,subprocess,json,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'ordered_context_full_output';PID=617604

def main():
 OUT.mkdir(exist_ok=False)
 proc=Path('/proc')/str(PID);identity=(proc/'stat').read_text().split()[21]
 receipt=BASE/'positive_query_head_v1/ordered_context_v1/receipt.json';deadline=time.monotonic()+3600
 while not receipt.exists():
  if not proc.exists() or (proc/'stat').read_text().split()[21]!=identity:raise RuntimeError('Training process ended without receipt; inspect log before any restart')
  if time.monotonic()>deadline:raise TimeoutError('Observer expired; does not terminate training')
  time.sleep(10)
 stages=[('evaluate_ordered_profile_intervals.py',[])]
 for key in ('new_cfc732e5f068','new_ceeae49d6777'):
  stages.extend([('infer_ordered_profile_full_grid.py',['--key',key]),('evaluate_ordered_profile_full_grid.py',['--key',key])])
 stages.append(('train_query_reference_ordered_probe.py',[]))
 for i,(script,args) in enumerate(stages):
  print('START',script,args,flush=True)
  with (OUT/(str(i)+'_'+script+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools'/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  print('COMPLETE',script,args,flush=True)
 (OUT/'receipt.json').write_text(json.dumps({'status':'FULL_OUTPUT_EVALUATION_COMPLETE','not_automatic_promotion':True}))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);(OUT/'failure.json').write_text(json.dumps({'error':traceback.format_exc()}));raise
