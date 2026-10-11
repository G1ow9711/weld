"""Sequential equal-budget provenance experiment, no automatic promotion."""
import sys,time,subprocess,traceback
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'boundary_provenance_experiment_followup'
def main():
 OUT.mkdir(exist_ok=False);waiting=[(640627,BASE/'boundary_event_provenance_v1/receipt.json'),(640519,BASE/'boundary_event_selector_continuation_followup/receipt.json')];identities={pid:((Path('/proc')/str(pid)/'stat').read_text().split()[21] if (Path('/proc')/str(pid)).exists() else None) for pid,_ in waiting};deadline=time.monotonic()+7200
 for pid,receipt in waiting:
  proc=Path('/proc')/str(pid)
  while not receipt.exists():
   if identities[pid] is None or not proc.exists() or (proc/'stat').read_text().split()[21]!=identities[pid]:raise RuntimeError(f'Producer {pid} terminal without {receipt}')
   if time.monotonic()>deadline:raise TimeoutError('Observation expired; producers not stopped')
   time.sleep(5)
 jobs=[('runtime_cache','cache_runtime_boundary_event_provenance.py',[]),('train','train_boundary_event_provenance_selector.py',[]),('train_fit','evaluate_boundary_provenance_selector_fit.py',[])];jobs.extend((key,'infer_boundary_event_provenance_selector.py',['--key',key]) for key in ('new_cfc732e5f068','new_ceeae49d6777'))
 for name,script,args in jobs:
  print('START',name,flush=True)
  with (OUT/(name+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools'/script),*args],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
  print('COMPLETE',name,flush=True)
 rows=[]
 for key in ('new_cfc732e5f068','new_ceeae49d6777'):
  folder=BASE/'boundary_event_provenance_selector_full_output_v1'/key;rows.append(dict(key=key,rows=read(folder/'receipt.json')['rows'],prediction_freeze_sha256=digest(folder/'prediction_freeze.json')))
 write(BASE/'boundary_event_provenance_selector_v1/decision.json',dict(status='EXPERIMENTAL_NOT_PROMOTED',cases=rows,reason='Requires review of matched baseline and full contract gates; no automatic promotion',DEV_reads=0))
 write(OUT/'receipt.json',dict(status='PROVENANCE_TRAIN_FIT_AND_TWO_FULL_OUTPUTS_COMPLETE',no_automatic_promotion=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
