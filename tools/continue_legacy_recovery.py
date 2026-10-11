"""Advance each rebuilt case only after both live upstream branches complete."""
import os,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'server_rebuild/best_baseline_restore'
STAGES=['graphs','object_features','relative_features','observed_features','bind_models','screen','edge_graph','solve','decode','select']

def save(value):
    p=BASE/'downstream_jobs.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)

def alive(pid):
    try:
        state=Path('/proc',str(pid),'stat').read_text().split(') ')[1].split()[0]
        return state!='Z'
    except FileNotFoundError:return False

def wait_input(key,folder,jobs):
    receipt=BASE/folder/('source_'+key)/'receipt.json'
    while not receipt.exists():
        status=json.loads((BASE/jobs).read_text())
        pid=status.get('controller_pid')
        if pid is None or not alive(pid):raise RuntimeError('Upstream stopped before required artifact: '+str(receipt))
        save(dict(status='WAITING_FOR_VERIFIED_LIVE_UPSTREAM',key=key,dependency=str(receipt),upstream_controller_pid=pid,controller_pid=os.getpid()))
        time.sleep(5)
    return receipt

def command(key,stage,args):
    log=BASE/('downstream_'+key+'_'+stage+'.log')
    with log.open('w') as out:
        p=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
        save(dict(status='RUNNING',key=key,stage=stage,pid=p.pid,controller_pid=os.getpid(),log=str(log)))
        code=p.wait()
    if code:raise RuntimeError(f'{key} {stage} exited {code}; see {log}')
    print('DOWNSTREAM_STAGE_COMPLETE',key,stage,flush=True)

if __name__=='__main__':
    try:
        for key in ['21','BM79A_native_local','10','25','36']:
            wait_input(key,'unified_xyz_candidates','xyz_recovery_jobs.json')
            wait_input(key,'learned_finite_support','active_recovery_jobs.json')
            root=BASE/'legacy_runtime'/('source_'+key)
            if not (root/'run185'/('source_'+key)/'receipt.json').exists():command(key,'union',['tools/rebuild_legacy_joint_union.py','--key',key])
            for stage in STAGES:
                receipt=root/(stage+'_reconstruction_receipt.json')
                if receipt.exists():
                    assert json.loads(receipt.read_text())['status']=='RECONSTRUCTED_STAGE_COMPLETE';continue
                command(key,stage,['tools/rebuild_legacy_downstream.py','--key',key,'--stage',stage])
            if not (root/'run379'/('test_'+key)/'evaluation.json').exists():command(key,'evaluate',['tools/evaluate_restored_legacy.py','--key',key])
        save(dict(status='FIVE_RESTORED_BASELINE_EVALUATIONS_COMPLETE',full_contract_pass=False))
    except BaseException as exc:
        save(dict(status='FAILED',error=repr(exc),controller_pid=os.getpid()));raise
