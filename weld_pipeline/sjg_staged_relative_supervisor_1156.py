"""Light controller: one 900-second deadline, original running resource guards."""
import argparse,os,shutil,time
from datetime import datetime,timezone
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,resources,utc
from weld_pipeline.sjg_owned_watchdog_1101 import supervise


def main():
    started=time.perf_counter();deadline=started+900
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run346');ap.add_argument('--check-only',action='store_true');ap.add_argument('--resume',default='')
    args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    previous_seconds=0.;pause_seconds=0.
    if args.resume:
        assert args.resume=='run347', 'Only the audited zero-real-fit software failure may resume here'
        assert read(OUT/'pro_review_run347/receipt.json')['assistant_sentinel_verified']
        decision=read(OUT/'pro_review_run347/resume_decision.json');assert decision['resume_approved']
        previous=OUT/args.resume;failure=read(previous/'failure.json')
        assert failure['stage']=='software_control' and failure['completed_stages']==['layout']
        assert not (previous/'object_saved.json').exists() and not (previous/'job_object').exists()
        assert (OUT/'single_staged_CPU_trial_Pro345.json').read_text(encoding='utf-8').splitlines()[0]==args.resume
        previous_seconds=float(failure['seconds']);deadline-=previous_seconds
        for run in decision.get('additional_no_start_runs',[]):
            preflight=read(OUT/run/'receipt.json')
            assert preflight['status']=='NO_START_RESOURCE_HEADROOM' and preflight['real_data_fit_calls']==0
            previous_seconds+=float(preflight['seconds']);deadline-=float(preflight['seconds'])
        pause_seconds=(datetime.now(timezone.utc)-datetime.fromisoformat(failure['watchdog']['exit_observed_utc'])).total_seconds()
    assert read(OUT/'pro_review_run345/receipt.json')['assistant_sentinel_verified']
    assert read(OUT/'run345/saved_tables_exact_control.json')['status']=='ALL_SEVEN_SAVED_TABLES_EXACT'
    layout_bytes=sum(read(OUT/'run345/saved_tables_exact_control.json')['files'][name]['bytes'] for name in ('object_x','endpoint_x'))+4096
    write(dest/'registered_protocol.json',dict(status='PRO345_STAGED_RELATIVE_FULL_MODEL',wall_seconds=900,n_jobs=1,
        startup_RAM_bytes=3758096384,startup_remaining_commit_bytes=5368709120,running_original_guards_unchanged=True,
        startup_disk_bytes=5368709120+layout_bytes,TRAIN_G=363,full_pilot_G=500,K=12,features=[152,153,32],
        old_run345_failure_preserved=True,contract_uncached_inference_10s_unchanged=True,
        resume_from=args.resume or None,prior_execution_seconds=previous_seconds,cumulative_budget_seconds=900,
        diagnostic_and_consult_pause_seconds=pause_seconds,continuous_wall_clock_claim=False,utc=utc()))
    checks=[]
    for i in range(3):
        if i:time.sleep(1)
        r=resources();r['disk_free_bytes']=shutil.disk_usage(ROOT).free
        r['passed']=r['available_physical_bytes']>=3758096384 and r['remaining_commit_bytes']>=5368709120 and r['disk_free_bytes']>=5368709120+layout_bytes
        checks.append(r);print('STARTUP_HEADROOM',i+1,r,flush=True)
    write(dest/'startup_headroom.json',dict(checks=checks,passed=all(r['passed'] for r in checks)))
    if not all(r['passed'] for r in checks):
        write(dest/'receipt.json',dict(status='NO_START_RESOURCE_HEADROOM',real_data_fit_calls=0,layout_started=False,
            full_contract_pass=False,seconds=time.perf_counter()-started))
        print('NO_START_RESOURCE_HEADROOM',flush=True);return
    if args.check_only:
        write(dest/'receipt.json',dict(status='STARTUP_RESOURCES_PASS_CHECK_ONLY',real_data_fit_calls=0,layout_started=False,
            full_contract_pass=False,seconds=time.perf_counter()-started))
        print('STARTUP_RESOURCES_PASS_CHECK_ONLY',flush=True);return
    snap=dest/'source_snapshot';snap.mkdir()
    for number in (904,1100,1101,1116,1129,1130,1139,1146,1149,1150,1154,1155,1156,1157):
        for path in (ROOT/'weld_pipeline').glob(f'*_{number}.py'):shutil.copy2(path,snap/path.name)
    ledger='single_staged_resume_Pro347.json' if args.resume else 'single_staged_CPU_trial_Pro345.json'
    with (OUT/ledger).open('x',encoding='utf-8') as f:f.write(args.run+'\n'+utc())
    stages=['layout','software_control','object','verify_object','endpoint','verify_endpoint','connection_tables','connection','verify_connection','package','verify_package']
    receipts=[]
    if args.resume:
        layout=read(previous/'layout.json');assert layout['status']=='F_ORDER_LOGICAL_PARITY_PASSED'
        assert layout['original']==read(OUT/'run345/saved_tables_exact_control.json')
        for entry in layout['features'].values():
            source=previous/entry['file'];assert source.name==entry['file'] and digest(source)==entry['sha256']
            os.link(source,dest/entry['file'])
        shutil.copy2(previous/'layout.json',dest/'layout.json')
        write(dest/'stage_layout_complete.json',dict(status='STAGE_COMPLETE',stage='layout',reused_from=args.resume,layout_sha256=digest(dest/'layout.json'),no_conversion_repeated=True))
        stages.remove('layout');receipts.append('layout')
    for stage in stages:
        remaining=deadline-time.perf_counter()
        if remaining<=0:
            write(dest/'failure.json',dict(status='SHARED_DEADLINE_EXPIRED',stage=stage,completed_stages=receipts))
            raise RuntimeError('Shared deadline expired')
        directory=dest/('job_'+stage);directory.mkdir()
        print('STAGED_EXECUTION',stage,'REMAINING_SECONDS',round(remaining,2),flush=True)
        result=supervise([str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m',
            'weld_pipeline.sjg_staged_relative_worker_1154','--run',args.run,'--stage',stage],directory,remaining)
        if result['termination_reason'] or result['exit_code']!=0:
            write(dest/'failure.json',dict(status='STAGED_TRAINING_INCOMPLETE',stage=stage,watchdog=result,
                completed_stages=receipts,automatic_retry=False,seconds=time.perf_counter()-started,
                cumulative_execution_seconds=previous_seconds+time.perf_counter()-started))
            raise RuntimeError('Stage failed; no automatic retry')
        assert read(dest/('stage_'+stage+'_complete.json'))['status']=='STAGE_COMPLETE';receipts.append(stage)
    package=read(dest/'package_verified.json');assert package['status']=='PACKAGE_VERIFIED' and time.perf_counter()<=deadline
    write(dest/'receipt.json',dict(package,status='THREE_RELATIVE_ORDINAL_FORESTS_TRAINING_COMPLETE',external_wall_seconds=time.perf_counter()-started,
        prior_execution_seconds=previous_seconds,cumulative_execution_seconds=previous_seconds+time.perf_counter()-started,
        completed_stages=receipts,full_contract_pass=False,extraction_correct_rate=None,extraction_precision_mm=None))
    print('STAGED_FULL_MODEL_COMPLETE',time.perf_counter()-started,flush=True)


if __name__=='__main__':main()
