import sys,time,subprocess,traceback
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import read,write
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'raw_tube_supervision_probe_followup'
def main():
 OUT.mkdir(exist_ok=False);proc=Path('/proc/730353');identity=(proc/'stat').read_text().split()[21] if proc.exists() else None;ready=BASE/'raw_tube_anchor_training_probe_v1/receipt.json';deadline=time.monotonic()+3600
 while not ready.exists():
  if identity is None or not proc.exists() or (proc/'stat').read_text().split()[21]!=identity:raise RuntimeError('Input event producer terminal without receipt')
  if time.monotonic()>deadline:raise TimeoutError('Observation expired; producer not stopped')
  time.sleep(5)
 rows=[]
 for key in read(BASE/'raw_tube_anchor_training_probe_v1/protocol.json')['keys']:
  with (OUT/(key+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools/prepare_raw_tube_anchor_supervision_probe.py'),'--key',key],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
  r=read(BASE/'raw_tube_anchor_supervision_probe_v1'/key/'receipt.json');old=read(BASE/'boundary_anchor_supervision_v1'/key/'receipt.json');rows.append(dict(key=key,line_children=r['line_reference_children'],old_available=old['children_with_both_endpoints_available'],union_available=r['children_with_both_endpoints_available'],query_event_quantiles=r['events_per_query_quantiles']));print(rows[-1],flush=True)
 write(OUT/'receipt.json',dict(status='TRAIN_CANDIDATE_CAPACITY_PROBE_COMPLETE',rows=rows,fit_calls=0,DEV_reads=0,not_runtime_profile_selection=True,not_full_D3=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
