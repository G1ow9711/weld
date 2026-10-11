"""Pro355: one shared 900-second window, one object fit, one final set model."""
import argparse, shutil, time
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,resources,utc
from weld_pipeline.sjg_owned_watchdog_1101 import supervise


def main():
    started=time.perf_counter()
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False)
    prior=sum(read(OUT/name/'prior_execution_cost.json')['measured_external_exec_seconds'] for name in ('run356','run357'))
    extra=OUT/'run357/interface_execution_cost.json'
    if extra.exists():prior+=read(extra)['measured_external_exec_seconds']
    deadline=started+900.-prior
    assert read(OUT/'pro_review_run355/receipt.json')['assistant_sentinel_verified']
    assert read(OUT/'run356/receipt.json')['status']=='DYNAMIC_ORACLE_MECHANISM_CONTROL_PASSED'
    assert read(OUT/'run357/receipt.json')['status']=='NUMPY_PARENT_SET_SCORER_CONTROL_PASSED'
    assert read(OUT/'run357/interface_control.json')['status']=='FROZEN_PARENT_FEATURE_COMPATIBILITY_PARITY_PASSED'
    write(dest/'registered_protocol.json',dict(status='PRO355_ONE_JOINT_RECALL_SET_TRIAL',new_total_seconds=900,
        prior_software_control_seconds=prior,old_run349_seconds=829.285514,old_execution_preserved_separately=True,
        object_real_fit_calls_allowed=1,endpoint_fit_calls_allowed=0,connection_fit_calls_allowed=0,
        negative_weight_multiplier=363./1122223.,object_presence_threshold=.5,
        selector_updates=112,selector_epochs=16,selector_training_sublimit=180,selector_inference_sublimit=60,
        train_order=['21','BM79A_native_local','10','4','25','36','42'],final_test_order=['21','BM79A_native_local','10','64'],
        no_teacher_candidates=True,all_legal_edges_required=True,TRAIN_G=363,full_pilot_G=500,full_contract_pass=False,utc=utc()))
    # Existing F layout is reused; reserve 256 MiB beyond the running disk guard.
    checks=[]
    for i in range(3):
        if i:time.sleep(1)
        r=resources();r['disk_free_bytes']=shutil.disk_usage(ROOT).free
        r['passed']=r['available_physical_bytes']>=3758096384 and r['remaining_commit_bytes']>=5368709120 and r['disk_free_bytes']>=5637144576
        checks.append(r);print('STARTUP_HEADROOM',i+1,r,flush=True)
    write(dest/'startup_headroom.json',dict(checks=checks,passed=all(r['passed'] for r in checks)))
    if not all(r['passed'] for r in checks):
        write(dest/'receipt.json',dict(status='NO_START_RESOURCE_HEADROOM',real_fit_calls=0,seconds=time.perf_counter()-started,
            cumulative_new_seconds=prior+time.perf_counter()-started,full_contract_pass=False))
        return
    with (OUT/'single_joint_recall_set_Pro355.json').open('x',encoding='utf-8') as f:f.write(args.run+'\n'+utc())
    snap=dest/'source_snapshot';snap.mkdir()
    for number in (904,1100,1101,1111,1116,1129,1139,1146,1149,1150,1154,1158,1159,1160,1161,1162,1163):
        for path in (ROOT/'weld_pipeline').glob(f'*_{number}.py'):shutil.copy2(path,snap/path.name)
    completed=[]
    def stage(name,key='',limit=None):
        remaining=deadline-time.perf_counter()
        if remaining<=0:raise TimeoutError('Shared new 900-second execution window exhausted')
        if limit is not None:remaining=min(remaining,limit)
        directory=dest/f'job_{name}_{key}';directory.mkdir(exist_ok=False)
        print('JOINT_STAGE',name,key,'REMAINING_NEW_SECONDS',round(deadline-time.perf_counter(),3),flush=True)
        command=[str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m',
            'weld_pipeline.sjg_recall_set_worker_1162','--run',args.run,'--stage',name,'--source',key]
        result=supervise(command,directory,remaining)
        if result['termination_reason'] or result['exit_code']!=0:
            write(dest/'failure.json',dict(status='JOINT_STAGE_EXECUTION_FAILED',stage=name,source=key,watchdog=result,
                completed_stages=completed,cumulative_new_seconds=prior+time.perf_counter()-started,automatic_retry=False))
            raise RuntimeError('Stage failed; no automatic retry')
        assert read(dest/f'stage_{name}_{key}_complete.json')['status']=='STAGE_COMPLETE'
        completed.append(dict(stage=name,source=key))
        write(dest/'progress.json',dict(completed_stages=completed,cumulative_new_seconds=prior+time.perf_counter()-started))
    try:
        for name in ('object','package','verify'):stage(name)
        for key in ('21','BM79A_native_local','10','4','25','36','42'):
            stage('upstream',key);stage('match',key)
            cap=read(dest/f'source_{key}/compatibility.json')
            if key in ('21','BM79A_native_local','10') and not cap['all_parents_coverable']:
                write(dest/'receipt.json',dict(status='UPSTREAM_COMPLETE_PARENT_CAPACITY_FAILED',failed_source=key,
                    candidate_coverage=cap,new_forest_fit_calls=1,selector_updates=0,completed_stages=completed,
                    cumulative_new_seconds=prior+time.perf_counter()-started,full_contract_pass=False))
                print('STOP_UPSTREAM_CANNOT_SUPPORT_REQUIRED_REGRESSION',key,flush=True);return
        stage('train_selector',limit=180.)
        results=[]
        for key in ('21','BM79A_native_local','10','64'):
            stage('test_upstream',key);stage('select',key,limit=60.);stage('score',key)
            report=read(dest/f'test_{key}/evaluation.json')['report'];results.append(report)
            passed=report['strict_TRAIN_gate_pass'] if key!='64' else report['TP']/report['G']>=.5 and report['FP']/report['G']<=.5
            if not passed:
                write(dest/'receipt.json',dict(status='JOINT_ROUND_FIRST_FAILED_TEST',failed_source=key,results=results,
                    new_forest_fit_calls=1,selector_updates=112,cumulative_new_seconds=prior+time.perf_counter()-started,full_contract_pass=False))
                return
        assert time.perf_counter()<=deadline
        write(dest/'receipt.json',dict(status='JOINT_PILOT_CONTINUATION_GATE_PASSED',results=results,
            new_forest_fit_calls=1,selector_updates=112,cumulative_new_seconds=prior+time.perf_counter()-started,full_contract_pass=False))
    except BaseException:
        if not (dest/'failure.json').exists():
            import traceback
            write(dest/'failure.json',dict(status='JOINT_CONTROLLER_FAILED',error=traceback.format_exc(),completed_stages=completed,
                cumulative_new_seconds=prior+time.perf_counter()-started,automatic_retry=False))
        raise


if __name__=='__main__':main()
