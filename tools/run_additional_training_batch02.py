"""Serialized additional TRAIN feature/geometry batch; original weights frozen."""
import fcntl,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from tools.continue_expansion_geometry_features import GEOMETRY,FRAGMENTS,marker
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';FROZEN=BASE/'frozen_features';OUT=BASE/'additional_batch02'
def main():
 OUT.mkdir(exist_ok=False);protocol=read(BASE/'additional_batch02_protocol.json');assert digest(BASE/'group_manifest.json')==protocol['group_manifest_sha256'];groups={g['key']:g for g in read(BASE/'group_manifest.json')['groups']}
 with (BASE/'geometry_features_continuation.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for case in protocol['cases']:
   key=case['key'];assert groups[key]['role']=='EXPANSION_TRAIN';assert digest(BASE/'inputs'/key/'receipt.json')==case['input_receipt_sha256']
   stages=[('features',None)]+[(s,True) for s in GEOMETRY]+[(s,False) for s in FRAGMENTS]
   for stage,geometry in stages:
    assert shutil.disk_usage(BASE).free>=12*1024**3,'Stage admission disk headroom below 12GiB; inspect before resuming'
    target=FROZEN/('complete_'+key+'.json') if stage=='features' else marker(key,stage,geometry)
    assert not target.exists(),'Unexpected preexisting artifact in new batch; inspect, do not overwrite'
    runner='run_batch02_frozen_feature_case.py' if stage=='features' else ('run_batch02_geometry_stage.py' if geometry else 'run_batch02_fragment_stage.py');args=[sys.executable,str(ROOT/'tools'/runner),'--key',key]
    if stage!='features':args+=['--stage',stage]
    write(OUT/'progress.json',dict(key=key,stage=stage,status='RUNNING',started_unix=time.time()));print('START',key,stage,flush=True)
    with (OUT/(key+'_'+stage+'.log')).open('w') as log:subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert target.exists();print('COMPLETE',key,stage,flush=True)
   # Only this task's completed case artifacts, no global cache drop or deletion.
   released=0
   for parent in FROZEN.iterdir():
    folder=parent/('source_'+key)
    if not folder.is_dir():continue
    for path in folder.rglob('*'):
     if path.is_file() and not path.is_symlink():
      with path.open('rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
      released+=1
   write(OUT/(key+'_complete.json'),dict(status='COMPLETE',key=key,cache_advised_files=released,files_deleted=0,GT_reads=0,STEP_reads=0,new_fit_calls=0))
  write(OUT/'receipt.json',dict(status='ADDITIONAL_TWO_TRAIN_FROZEN_PIPELINES_COMPLETE',keys=[c['key'] for c in protocol['cases']],new_fit_calls=0,DEV_reads=0))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc(),time_unix=time.time()));raise
