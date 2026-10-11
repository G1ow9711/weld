"""Pro362: finish updates 66..112 within the original window balance."""
import argparse,ast,hashlib,os,shutil,time,traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,resources,utc
from weld_pipeline.sjg_owned_watchdog_1101 import supervise


def main():
    started=time.perf_counter();ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);completed=[]
    pro=read(OUT/'pro_review_run362/receipt.json')
    assert pro['assistant_sentinel_verified'] and pro['generation_stopped']
    assert hashlib.sha256((OUT/'pro_review_run362/answer.md').read_text(encoding='utf-8').encode()).hexdigest()==pro['answer_sha256']
    assert digest(OUT/'pro_review_run362/packet.md')==pro['packet_sha256']
    budget=283.8590928;deadline=started+budget
    for number in (1159,1162,1173,1174,1178,1179,1180):
        for path in (ROOT/'weld_pipeline').glob(f'*_{number}.py'):ast.parse(path.read_text(encoding='utf-8-sig'))
    write(dest/'registered_protocol.json',dict(status='PRO362_COMPLETE_ORIGINAL_112_FROM_UPDATE_65',
        original_window_seconds=600.,prior_failed_window_seconds=313.8154848,prior_audit_external_seconds=2.3254224,
        execution_budget_seconds=budget,consultation_wait_separately_accounted=True,
        forest_fit_calls_allowed=0,selector_updates=112,completed_updates_restored=65,new_updates_required=47,
        selector_epochs=16,selector_180_second_sublimit_removed=True,
        original_TRAIN_sources=['21','4','10','25','36','42','BM79A_native_local'],TRAIN_G=363,pilot_G=500,
        extra_100_percent_candidate_gate_removed=True,baseline_preserved=True,missing_GT_not_removed=True,
        final_test_order=['21','BM79A_native_local','10','64'],formal_raw_PCD_time_seconds=10,
        path_development_seconds=60,full_contract_pass=False,utc=utc()))
    checks=[]
    for i in range(3):
        if i:time.sleep(1)
        r=resources();r['disk_free_bytes']=shutil.disk_usage(ROOT).free
        r['passed']=r['available_physical_bytes']>=3758096384 and r['remaining_commit_bytes']>=5368709120 and r['disk_free_bytes']>=5637144576
        checks.append(r);print('STARTUP_HEADROOM',i+1,r,flush=True)
    write(dest/'startup_headroom.json',dict(checks=checks,passed=all(v['passed'] for v in checks)))
    if not all(v['passed'] for v in checks):
        write(dest/'receipt.json',dict(status='NO_START_RESOURCE_HEADROOM',updates=0,seconds=time.perf_counter()-started,full_contract_pass=False));return
    with (OUT/'single_B_trial_Pro362.json').open('x',encoding='utf-8') as stream:stream.write(args.run+'\n'+utc())
    def stage(name,key='',limit=None):
        remaining=deadline-time.perf_counter()
        if remaining<=0:raise TimeoutError('Shared Pro362 B trial balance exhausted')
        if limit is not None:remaining=min(remaining,limit)
        if name in ('resume_control','resume_train'):module='weld_pipeline.sjg_selector_resume_worker_1179'
        elif name in ('aggregate','score'):module='weld_pipeline.sjg_selector_trial_worker_1174'
        elif name in ('graph','solve','decode','test_graph','test_solve','test_decode'):module='weld_pipeline.sjg_sparse_recovery_worker_1170'
        elif name=='screen':module='weld_pipeline.sjg_recall_recovery_worker_1165'
        else:module='weld_pipeline.sjg_recall_set_worker_1162'
        directory=dest/f'job_{name}_{key}';directory.mkdir(exist_ok=False)
        command=[str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m',module,'--run',args.run,'--stage',name,'--source',key]
        if name in ('resume_control','resume_train'):command.extend(['--limit-seconds',str(remaining)])
        print('B_TRIAL_STAGE',name,key,'SECONDS_AVAILABLE',remaining,flush=True)
        result=supervise(command,directory,remaining)
        if result['termination_reason'] or result['exit_code']!=0:
            write(dest/'failure.json',dict(status='B_TRIAL_STAGE_FAILED',stage=name,source=key,watchdog=result,
                completed_stages=completed,seconds=time.perf_counter()-started,execution_budget_seconds=budget,
                automatic_retry=False,training_status='TRAIN_COMPLETE_EVALUATION_INCOMPLETE' if (dest/'selector_training.json').exists() else 'TRAIN_INCOMPLETE',full_contract_pass=False))
            raise RuntimeError('B trial stopped; preserve complete artifacts, no automatic extension')
        assert read(dest/f'stage_{name}_{key}_complete.json')['status']=='STAGE_COMPLETE'
        completed.append(dict(stage=name,source=key));write(dest/'progress.json',dict(completed_stages=completed,seconds=time.perf_counter()-started))
    try:
        old=OUT/'run362';binding=read(old/'upstream_verified.json');assert digest(old/'model.pkl')==binding['model_sha256']
        os.link(old/'model.pkl',dest/'model.pkl');shutil.copy2(old/'upstream_verified.json',dest/'upstream_verified.json')
        reused={}
        for key in ('21','4','10','25','36','42','BM79A_native_local'):
            source=old/f'source_{key}';target=dest/f'source_{key}';target.mkdir()
            r=read(source/'receipt.json');c=read(source/'compatibility.json')
            assert r['upstream_model_sha256']==binding['model_sha256']
            assert digest(source/'decoded.json')==r['decoded_sha256']==c['prediction_sha256']
            assert digest(source/'unary.npy')==r['unary_sha256']
            assert read(old/f'stage_match_{key}_complete.json')['status']=='STAGE_COMPLETE'
            files={}
            for path in source.iterdir():
                if path.is_file():os.link(path,target/path.name);files[path.name]=digest(path)
            os.link(old/f'screen_{key}.json',dest/f'screen_{key}.json')
            shutil.copy2(old/f'stage_match_{key}_complete.json',dest/f'stage_match_{key}_complete.json')
            reused[key]=files
        write(dest/'frozen_sources_reused.json',reused)
        shutil.copy2(old/'all_train_candidate_coverage.json',dest/'all_train_candidate_coverage.json')
        snapshot=dest/'source_snapshot';snapshot.mkdir()
        for number in (1158,1159,1160,1161,1162,1164,1165,1168,1169,1170,1173,1174,1176,1178,1179,1180):
            for path in (ROOT/'weld_pipeline').glob(f'*_{number}.py'):shutil.copy2(path,snapshot/path.name)
        stage('resume_control')
        stage('resume_train')
        trained=read(dest/'selector_training.json')
        assert trained['updates']==112 and trained['new_updates']==47 and trained['completed_before_resume']==65
        assert digest(dest/'selector.npz')==trained['selector_sha256']
        for key in ('21','BM79A_native_local','10'):
            stage('test_upstream',key);stage('select',key,60.);stage('score',key)
        stage('aggregate')
        result=read(dest/'three_case_final_summary.json')
        if not result['geometric_continuation_pass']:
            write(dest/'receipt.json',dict(status='B_FINAL_THREE_CASE_GEOMETRY_FAILED',summary=result,
                selector_updates=112,seconds=time.perf_counter()-started,full_contract_pass=False));return
        key='64';stage('screen',key);stage('test_graph',key);stage('test_solve',key,60.);stage('test_decode',key)
        stage('select',key,60.);stage('score',key)
        write(dest/'receipt.json',dict(status='B_PILOT_AND_DEV_TESTS_COMPLETED',summary=result,
            DEV_report=read(dest/'test_64/evaluation.json')['report'],selector_updates=112,
            seconds=time.perf_counter()-started,full_contract_pass=False))
    except BaseException:
        if not (dest/'failure.json').exists():write(dest/'failure.json',dict(status='B_TRIAL_CONTROLLER_FAILED',error=traceback.format_exc(),
            completed_stages=completed,seconds=time.perf_counter()-started,automatic_retry=False,full_contract_pass=False))
        raise


if __name__=='__main__':main()
