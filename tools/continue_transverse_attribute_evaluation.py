"""Freeze both matched attribute arms before unchanged reference correspondence scoring."""
import sys,time,subprocess,traceback
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'transverse_attribute_evaluation_followup'
def main():
 OUT.mkdir(exist_ok=False);proc=Path('/proc/784011');identity=(proc/'stat').read_text().split()[21] if proc.exists() else None;ready=BASE/'transverse_attribute_training_followup/receipt.json';deadline=time.monotonic()+14400
 while not ready.exists():
  if identity is None or not proc.exists() or (proc/'stat').read_text().split()[21]!=identity:raise RuntimeError('Training/context producer terminal without receipt')
  if time.monotonic()>deadline:raise TimeoutError('Observation expired; producer not stopped')
  time.sleep(5)
 rows=[]
 for stage,script in [('freeze','infer_transverse_endpoint_attributes.py'),('score','evaluate_transverse_endpoint_attributes.py')]:
  for arm,flags in [('control',[]),('context',['--transverse-context'])]:
   print('START',stage,arm,flush=True)
   with (OUT/(stage+'_'+arm+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools'/script),*flags],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
   print('COMPLETE',stage,arm,flush=True)
   if stage=='score':
    report=BASE/('transverse_attribute_'+arm+'_evaluation_v1/receipt.json');r=read(report);rows.append(dict(arm=arm,cases=r['cases'],evaluation_sha256=digest(report)));model=BASE/('transverse_attribute_head_v1' if arm=='context' else 'transverse_attribute_control_v1');write(model/'decision.json',dict(status='EXPERIMENTAL_NOT_PROMOTED',evaluation_sha256=digest(report),scope='Conditional attributes on unchanged geometry, full contract still outstanding',DEV_reads=0))
 write(OUT/'receipt.json',dict(status='MATCHED_TRANSVERSE_ATTRIBUTE_EVALUATION_COMPLETE',arms=rows,DEV_reads=0,no_automatic_promotion=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
