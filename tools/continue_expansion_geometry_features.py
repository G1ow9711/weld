"""Serial continuation of frozen expansion geometry and fragment features.

Existing first-case geometry is allowed to finish; markers are checked before
continuation. This worker never trains, reads targets, or accesses DEV inputs.
"""
import argparse,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'server_rebuild/expansion_v1'
FROZEN=BASE/'frozen_features'
GEOMETRY=['merge162','merge26','instances','finite','surfaces','rank2','unified','union']
FRAGMENTS=['graphs','object_features','relative_features','observed_features','bind_models','screen','edge_graph','solve','decode','select']
def marker(key,stage,geometry):
 return FROZEN/'stage_receipts'/key/(stage+'.json') if geometry else FROZEN/'legacy_runtime'/('source_'+key)/(stage+'_reconstruction_receipt.json')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--wait-pid',type=int,required=True);a=ap.parse_args()
 with (BASE/'geometry_features_continuation.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  cases=json.loads((BASE/'coverage_cohort_protocol.json').read_text())['cases'];first=cases[0]['key']
  p=Path('/proc')/str(a.wait_pid);start=(p/'stat').read_text().split()[21] if p.exists() else None
  if start is not None:
   assert b'run_expansion_geometry_stage.py' in (p/'cmdline').read_bytes(), 'Unexpected process; do not wait on unrelated work'
  while p.exists() and (p/'stat').read_text().split()[21]==start:
   if marker(first,'union',True).exists():break
   time.sleep(10)
  assert marker(first,'union',True).exists(),'First geometry worker did not complete; inspect its original log'
  for case in cases:
   key=case['key'];assert (FROZEN/('complete_'+key+'.json')).exists()
   for geometry,stages in [(True,GEOMETRY),(False,FRAGMENTS)]:
    for stage in stages:
     target=marker(key,stage,geometry)
     if target.exists():
      receipt=json.loads(target.read_text());assert receipt['status'] in ('COMPLETE','RECONSTRUCTED_STAGE_COMPLETE')
      assert receipt.get('key',receipt.get('source_key'))==key and receipt['stage']==stage
      print('REUSE_COMPLETE',key,stage,flush=True);continue
     runner='run_expansion_geometry_stage.py' if geometry else 'run_expansion_fragment_stage.py'
     print('START',key,stage,flush=True)
     subprocess.run([sys.executable,str(ROOT/'tools'/runner),'--key',key,'--stage',stage],cwd=ROOT,check=True)
     assert target.exists();print('COMPLETE',key,stage,flush=True)
  print('EXPANSION_GEOMETRY_FRAGMENT_PIPELINE_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  (BASE/'geometry_features_continuation_failure.json').write_text(json.dumps({'error':traceback.format_exc(),'time_unix':time.time()},indent=2));raise
