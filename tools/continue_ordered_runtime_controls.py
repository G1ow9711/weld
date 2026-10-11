"""Verify each completed full-grid artifact while the known producer is alive."""
import time,subprocess,sys,json,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'ordered_runtime_controls';PID=617672

def main():
 OUT.mkdir(exist_ok=False);proc=Path('/proc')/str(PID);identity=(proc/'stat').read_text().split()[21];deadline=time.monotonic()+3600
 for key in ('new_cfc732e5f068','new_ceeae49d6777'):
  ready=BASE/'positive_query_ordered_context'/key/'receipt.json'
  while not ready.exists():
   if not proc.exists() or (proc/'stat').read_text().split()[21]!=identity:raise RuntimeError('Known full-grid producer ended before receipt; inspect its failure')
   if time.monotonic()>deadline:raise TimeoutError('Observer expired, not producer terminal')
   time.sleep(10)
  with (OUT/(key+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools/verify_ordered_context_runtime_equivalence.py'),'--key',key],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
  print('PASS',key,flush=True)
 (OUT/'receipt.json').write_text(json.dumps({'status':'BOTH_RUNTIME_CONTROLS_PASS'}))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);(OUT/'failure.json').write_text(json.dumps({'error':traceback.format_exc()}));raise
