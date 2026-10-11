"""Prepare registered new TRAIN event pools, labels and provenance; no model fitting."""
import subprocess,sys,traceback,shutil
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'batch02_boundary_training_preparation'
def main():
 OUT.mkdir(exist_ok=False);assert (BASE/'batch02_supervision_followup/receipt.json').exists();assert shutil.disk_usage(ROOT).free>12*1024**3
 protocol=read(BASE/'additional_batch02_protocol.json');keys=[x['key'] for x in protocol['cases']];assert set(keys).isdisjoint(read(BASE/'reference_query_task_v1/protocol.json')['training_keys'])
 write(OUT/'protocol.json',dict(keys=keys,parent_protocol_sha256=digest(BASE/'additional_batch02_protocol.json'),scope='Additional TRAIN data preparation, not runtime or acceptance; no model fit',heldout_reads=0,DEV_reads=0))
 jobs=[('events','prepare_batch02_boundary_anchor_cache.py',[])]+[(key+'_supervision','prepare_batch02_boundary_anchor_supervision.py',['--key',key]) for key in keys]+[('provenance','cache_batch02_boundary_event_provenance.py',[])]
 for name,script,args in jobs:
  print('START',name,flush=True)
  with (OUT/(name+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools'/script),*args],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
  print('COMPLETE',name,flush=True)
 write(OUT/'receipt.json',dict(status='BATCH02_TRAIN_EVENTS_LABELS_PROVENANCE_READY',fit_calls=0,DEV_reads=0,heldout_reads=0))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
