"""Wait for matched fits, score both frozen models on identical balanced inputs."""
import sys,time,subprocess,traceback
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'boundary_provenance_batch02_full_scoring_followup'
def main():
 OUT.mkdir(exist_ok=False);ready=BASE/'boundary_provenance_batch02_comparison_followup/receipt.json';proc=Path('/proc/725085');identity=(proc/'stat').read_text().split()[21] if proc.exists() else None;deadline=time.monotonic()+14400
 write(OUT/'protocol.json',dict(arms=['control6','expanded8'],detector='Frozen balanced reference detector at0.5',keys=['new_cfc732e5f068','new_ceeae49d6777'],same_input_events_and_provenance=True,quality_filter=False,no_parameter_search=True,DEV_reads=0))
 while not ready.exists():
  if identity is None or not proc.exists() or (proc/'stat').read_text().split()[21]!=identity:raise RuntimeError('Fit controller terminal without receipt')
  if time.monotonic()>deadline:raise TimeoutError('Observation expired; producer not stopped')
  time.sleep(5)
 rows=[]
 for arm in ['control6','expanded8']:
  model=BASE/('boundary_provenance_batch02_'+arm+'_v1');assert (model/'training_fit_summary.json').exists()
  for key in ['new_cfc732e5f068','new_ceeae49d6777']:
   print('START',arm,key,flush=True)
   with (OUT/(arm+'_'+key+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools/infer_boundary_provenance_batch02_output.py'),'--comparison',arm,'--key',key],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
   folder=BASE/('boundary_provenance_batch02_'+arm+'_full_output_v1')/key;rows.append(dict(arm=arm,key=key,rows=read(folder/'receipt.json')['rows'],prediction_freeze_sha256=digest(folder/'prediction_freeze.json')));print('COMPLETE',rows[-1],flush=True)
  write(model/'decision.json',dict(status='EXPERIMENTAL_NOT_PROMOTED',rows=[x for x in rows if x['arm']==arm],reason='No automatic promotion; full contract validation still outstanding',DEV_reads=0))
 write(OUT/'receipt.json',dict(status='MATCHED_ENDPOINT_MODELS_FULL_OUTPUT_SCORED',rows=rows,DEV_reads=0,no_automatic_promotion=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
