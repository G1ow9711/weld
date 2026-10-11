"""Pro358: one new 900-second continuation; keep run358 failure permanent."""
import argparse,hashlib,os,shutil,time,traceback
from datetime import datetime,timezone
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,resources,utc
from weld_pipeline.sjg_owned_watchdog_1101 import supervise


def main():
    started=time.perf_counter();deadline=started+900.
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;bootstrap=dest/'bootstrap_failure.json'
    if bootstrap.exists():
        previous=read(bootstrap)
        assert not (dest/'registered_protocol.json').exists() and not (OUT/'single_execution_recovery_Pro358.json').exists()
        # PowerShell emits seven fractional digits; Python 3.10 accepts six.
        original_start=datetime.strptime(previous['original_launcher_start_utc'][:26],'%Y-%m-%dT%H:%M:%S.%f').replace(tzinfo=timezone.utc)
        elapsed=(datetime.now(timezone.utc)-original_start).total_seconds()
        assert 0<=elapsed<900
        started-=elapsed;deadline=started+900.
    else:dest.mkdir(exist_ok=False)
    old=OUT/'run358'
    pro=read(OUT/'pro_review_run358/receipt.json')
    assert pro['assistant_sentinel_verified'] and pro['generation_stopped']
    # Collector hashes normalized assistant text before Windows newline output.
    assert hashlib.sha256((OUT/'pro_review_run358/answer.md').read_text(encoding='utf-8').encode()).hexdigest()==pro['answer_sha256']
    failure=read(old/'failure.json');assert failure['status']=='JOINT_STAGE_EXECUTION_FAILED'
    prior=float(failure['cumulative_new_seconds'])
    write(dest/'registered_protocol.json',dict(status='PRO358_ONE_EXECUTION_ONLY_RECOVERY',new_window_seconds=900,
        measured_prior_execution_seconds=prior,diagnostic_budget_reservation_seconds=60,
        diagnostic_external_actual_seconds=None,reservation_is_not_measured_or_proven_upper_bound=True,
        total_budget_occupancy_ceiling_seconds=prior+60+900,original_900_protocol_resource_failure_preserved=True,
        forest_fit_calls_allowed=0,selector_updates=112,selector_epochs=16,selector_sublimit_seconds=180,
        candidate_order=['10','4','25','36','42'],final_order=['21','BM79A_native_local','10','64'],
        TRAIN_G=363,pilot_G=500,all_legal_edges_required=True,full_contract_pass=False,utc=utc()))
    checks=[]
    for i in range(3):
        if i:time.sleep(1)
        r=resources();r['disk_free_bytes']=shutil.disk_usage(ROOT).free
        r['passed']=r['available_physical_bytes']>=3758096384 and r['remaining_commit_bytes']>=5368709120 and r['disk_free_bytes']>=5637144576
        checks.append(r);print('STARTUP_HEADROOM',i+1,r,flush=True)
    write(dest/'startup_headroom.json',dict(checks=checks,passed=all(r['passed'] for r in checks)))
    if not all(r['passed'] for r in checks):
        write(dest/'receipt.json',dict(status='NO_START_RESOURCE_HEADROOM',forest_fit_calls=0,seconds=time.perf_counter()-started,full_contract_pass=False));return
    with (OUT/'single_execution_recovery_Pro358.json').open('x',encoding='utf-8') as f:f.write(args.run+'\n'+utc())
    completed=[]
    def stage(name,key='',limit=None):
        remaining=deadline-time.perf_counter()
        if remaining<=0:raise TimeoutError('Pro358 shared 900-second recovery exhausted')
        if limit is not None:remaining=min(remaining,limit)
        directory=dest/f'job_{name}_{key}';directory.mkdir(exist_ok=False)
        module='weld_pipeline.sjg_recall_recovery_worker_1165' if name in ('control','screen','graph','test_graph') else 'weld_pipeline.sjg_recall_set_worker_1162'
        command=[str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m',module,'--run',args.run,'--stage',name,'--source',key]
        print('RECOVERY_STAGE',name,key,'REMAINING',round(remaining,3),flush=True)
        result=supervise(command,directory,remaining)
        if result['termination_reason'] or result['exit_code']!=0:
            write(dest/'failure.json',dict(status='RECOVERY_STAGE_FAILED',stage=name,source=key,watchdog=result,
                completed_stages=completed,recovery_seconds=time.perf_counter()-started,
                measured_cumulative_execution_seconds=prior+time.perf_counter()-started,diagnostic_external_actual_unknown=True,
                automatic_retry=False,full_contract_pass=False))
            raise RuntimeError('Stop after failed execution; no automatic third continuation')
        assert read(dest/f'stage_{name}_{key}_complete.json')['status']=='STAGE_COMPLETE'
        completed.append(dict(stage=name,source=key))
        write(dest/'progress.json',dict(completed_stages=completed,recovery_seconds=time.perf_counter()-started))
    try:
        cleanup=read(old/'duplicate_C_tables_cleanup.json')
        for item in cleanup['files']:
            from pathlib import Path
            assert not Path(item['delete_path']).exists()
            assert digest(item['retained_path'])==item['retained_sha256']
        write(dest/'retained_layout_verified.json',dict(cleanup_receipt_sha256=digest(old/'duplicate_C_tables_cleanup.json'),
            no_cleanup_repeated=True,no_forest_refit=True))
        binding=read(old/'upstream_verified.json');assert digest(old/'model.pkl')==binding['model_sha256']
        os.link(old/'model.pkl',dest/'model.pkl');shutil.copy2(old/'upstream_verified.json',dest/'upstream_verified.json')
        for key in ('21','BM79A_native_local'):
            source=old/f'source_{key}';target=dest/f'source_{key}';target.mkdir()
            r=read(source/'receipt.json');assert r['upstream_model_sha256']==binding['model_sha256']
            assert digest(source/'decoded.json')==r['decoded_sha256'] and digest(source/'unary.npy')==r['unary_sha256']
            assert read(source/'compatibility.json')['all_parents_coverable']
            for path in source.iterdir():
                if path.is_file():os.link(path,target/path.name)
            shutil.copy2(old/f'stage_match_{key}_complete.json',dest/f'stage_match_{key}_complete.json')
        screen=old/'memory_diagnosis/frozen_screen.json';sr=read(old/'memory_diagnosis/receipt.json')
        assert digest(screen)==sr['screen_sha256'];os.link(screen,dest/'screen_10.json')
        value=read(screen);assert value['model_sha256']==binding['model_sha256'] and len(value['nodes'])==4795
        write(dest/'screen_10_receipt.json',dict(screen_sha256=sr['screen_sha256'],model_sha256=value['model_sha256'],
            feature_receipt_sha256=value['feature_receipt_sha256'],reuse_original_frozen_screen=True))
        snap=dest/'source_snapshot';snap.mkdir()
        for number in (1104,1107,1111,1129,1139,1150,1158,1159,1160,1161,1162,1164,1165,1166,1167):
            for path in (ROOT/'weld_pipeline').glob(f'*_{number}.py'):shutil.copy2(path,snap/path.name)
        stage('control')
        for key in ('10','4','25','36','42'):
            if key!='10':stage('screen',key)
            stage('graph',key);stage('match',key)
            cap=read(dest/f'source_{key}/compatibility.json')
            if key=='10' and not cap['all_parents_coverable']:
                write(dest/'receipt.json',dict(status='UPSTREAM_COMPLETE_PARENT_CAPACITY_FAILED',source=key,candidate_coverage=cap,
                    forest_fit_calls=0,selector_updates=0,recovery_seconds=time.perf_counter()-started,full_contract_pass=False));return
        stage('train_selector',limit=180.)
        results=[]
        for key in ('21','BM79A_native_local','10','64'):
            if key=='64':stage('screen',key);stage('test_graph',key)
            else:stage('test_upstream',key)
            stage('select',key,limit=60.);stage('score',key)
            r=read(dest/f'test_{key}/evaluation.json')['report'];results.append(r)
            passed=r['strict_TRAIN_gate_pass'] if key!='64' else r['TP']/r['G']>=.5 and r['FP']/r['G']<=.5
            if not passed:
                write(dest/'receipt.json',dict(status='RECOVERY_FIRST_FAILED_TEST',source=key,results=results,forest_fit_calls=0,
                    selector_updates=112,recovery_seconds=time.perf_counter()-started,full_contract_pass=False));return
        assert time.perf_counter()<=deadline
        write(dest/'receipt.json',dict(status='RECOVERY_PILOT_CONTINUATION_GATE_PASSED',results=results,forest_fit_calls=0,
            selector_updates=112,recovery_seconds=time.perf_counter()-started,full_contract_pass=False))
    except BaseException:
        if not (dest/'failure.json').exists():write(dest/'failure.json',dict(status='RECOVERY_CONTROLLER_FAILED',
            error=traceback.format_exc(),completed_stages=completed,recovery_seconds=time.perf_counter()-started,automatic_retry=False))
        raise


if __name__=='__main__':main()
