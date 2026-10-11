"""Continue the original best composite once each original baseline is frozen."""
import json,os,subprocess,sys,time
from pathlib import Path
from tools.rebuild_legacy_sequences import STAGES
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/best_baseline_restore'

def save(value):
 p=BASE/'incremental_jobs.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)

def command(key,stage,script,extra=()):
 log=BASE/('incremental_'+key+'_'+stage+'.log')
 with log.open('w') as stream:
  p=subprocess.Popen([sys.executable,'-u','tools/'+script+'.py','--key',key,*extra],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
  save(dict(status='RUNNING',controller_pid=os.getpid(),key=key,stage=stage,pid=p.pid,log=str(log)));code=p.wait()
 if code:raise RuntimeError(f'{key}/{stage}: exit {code}, log {log}')
 print('INCREMENTAL_COMPLETE',key,stage,flush=True)

def wait_baseline(key):
 r=BASE/'legacy_runtime'/('source_'+key)/'run379'/('test_'+key)/'evaluation.json'
 while not r.exists():
  job=json.loads((BASE/'downstream_jobs.json').read_text());pid=job.get('controller_pid')
  if job.get('status')=='FAILED' or pid is None or not Path('/proc',str(pid)).exists():raise RuntimeError('Baseline worker stopped before '+key)
  save(dict(status='WAITING_FOR_LIVE_BASELINE',controller_pid=os.getpid(),key=key,baseline_pid=pid));time.sleep(5)

if __name__=='__main__':
 try:
  for key in ['21','BM79A_native_local','10','25','36']:
   wait_baseline(key);case=BASE/'legacy_runtime'/('source_'+key)
   if (case/'run446'/('test_'+key)/'evaluation.json').exists():continue
   for stage in ['whole','local','admission']:
    receipt=case/(stage+'_incremental_receipt.json') if stage!='admission' else case/'run420'/('proposal_'+key)/'prediction_receipt.json'
    if not receipt.exists():command(key,stage,'rebuild_legacy_incremental_inputs',['--stage',stage])
   for stage in STAGES:
    receipt=case/'run421'/('stage_'+stage+'_'+key+'.json')
    if not receipt.exists():command(key,stage,'rebuild_legacy_sequences',['--stage',stage])
   for stage in ['occupancy','overlay','pairs','parent_features','child_features','infer446']:
    if not (case/(stage+'_addition_receipt.json')).exists():command(key,'addition_'+stage,'rebuild_legacy_additions',['--stage',stage])
   command(key,'evaluate446','evaluate_restored_legacy',['--model','run446'])
  save(dict(status='ALL_FIVE_ORIGINAL_COMPOSITE_EVALUATIONS_COMPLETE',full_contract_pass=False))
 except BaseException as exc:
  save(dict(status='FAILED',error=repr(exc),controller_pid=os.getpid()));raise
