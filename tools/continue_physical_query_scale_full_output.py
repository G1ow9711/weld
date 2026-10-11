"""Follow the verified training process, then score complete frozen query output."""
import time,sys,subprocess,json,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'physical_query_scale_followup';PID=622211

def main():
 OUT.mkdir(exist_ok=False)
 proc=Path('/proc')/str(PID);identity=(proc/'stat').read_text().split()[21]
 receipt=BASE/'physical_query_scale_comparison_v1/metric460/receipt.json';deadline=time.monotonic()+3600
 while not receipt.exists():
  if not proc.exists() or (proc/'stat').read_text().split()[21]!=identity:raise RuntimeError('Training process ended without receipt; inspect log before any restart')
  if time.monotonic()>deadline:raise TimeoutError('Observer expired; does not terminate training')
  time.sleep(10)
 stages=[]
 for arm in ('control457','metric460'):
  stages.append(('evaluate_physical_query_scale_intervals.py',['--arm',arm]))
  for key in ('new_cfc732e5f068','new_ceeae49d6777'):
   stages.append(('infer_physical_query_scale_full_output.py',['--key',key,'--arm',arm]))
 for i,(script,args) in enumerate(stages):
  print('START',script,args,flush=True)
  with (OUT/(str(i)+'_'+script+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools'/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  print('COMPLETE',script,args,flush=True)
 (OUT/'receipt.json').write_text(json.dumps({'status':'FULL_OUTPUT_EVALUATION_COMPLETE','not_automatic_promotion':True}))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);(OUT/'failure.json').write_text(json.dumps({'error':traceback.format_exc()}));raise
