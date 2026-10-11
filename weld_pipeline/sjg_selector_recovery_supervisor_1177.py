"""Pro361 numerical-certificate recovery; one 600s window, full original B."""
import argparse,ast,hashlib,os,shutil,time,traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,resources,utc
from weld_pipeline.sjg_owned_watchdog_1101 import supervise


def main():
    started=time.perf_counter();ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);completed=[]
    pro=read(OUT/'pro_review_run361/receipt.json')
    assert pro['assistant_sentinel_verified'] and pro['generation_stopped']
    assert hashlib.sha256((OUT/'pro_review_run361/answer.md').read_text(encoding='utf-8').encode()).hexdigest()==pro['answer_sha256']
    assert digest(OUT/'pro_review_run361/packet.md')==pro['packet_sha256']
    budget=600.;deadline=started+budget
    for number in (1159,1162,1168,1173,1174,1176,1177):
        for path in (ROOT/'weld_pipeline').glob(f'*_{number}.py'):ast.parse(path.read_text(encoding='utf-8-sig'))
    write(dest/'registered_protocol.json',dict(status='PRO361_FIRST_FULL_B_TRIAL',approved_new_window_seconds=600.,replaced_unspent_balance_seconds=311.5696832,
        prior_failed_window_seconds=92.558299,prior_diagnostic_external_seconds=1.0685577,execution_budget_seconds=budget,
        forest_fit_calls_allowed=0,selector_updates=112,selector_epochs=16,selector_sublimit_seconds=180,
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
    with (OUT/'single_B_trial_Pro361.json').open('x',encoding='utf-8') as stream:stream.write(args.run+'\n'+utc())
    def stage(name,key='',limit=None):
        remaining=deadline-time.perf_counter()
        if remaining<=0:raise TimeoutError('Shared Pro361 B trial balance exhausted')
        if limit is not None:remaining=min(remaining,limit)
        if name in ('certificate_control','cache_control','coverage','train_selector','aggregate','score'):module='weld_pipeline.sjg_selector_trial_worker_1174'
        elif name in ('graph','solve','decode','test_graph','test_solve','test_decode'):module='weld_pipeline.sjg_sparse_recovery_worker_1170'
        elif name=='screen':module='weld_pipeline.sjg_recall_recovery_worker_1165'
        else:module='weld_pipeline.sjg_recall_set_worker_1162'
        directory=dest/f'job_{name}_{key}';directory.mkdir(exist_ok=False)
        command=[str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m',module,'--run',args.run,'--stage',name,'--source',key]
        print('B_TRIAL_STAGE',name,key,'SECONDS_AVAILABLE',remaining,flush=True)
        result=supervise(command,directory,remaining)
        if result['termination_reason'] or result['exit_code']!=0:
            write(dest/'failure.json',dict(status='B_TRIAL_STAGE_FAILED',stage=name,source=key,watchdog=result,
                completed_stages=completed,seconds=time.perf_counter()-started,execution_budget_seconds=budget,
                automatic_retry=False,full_contract_pass=False))
            raise RuntimeError('B trial stopped; preserve complete artifacts, no automatic extension')
        assert read(dest/f'stage_{name}_{key}_complete.json')['status']=='STAGE_COMPLETE'
        completed.append(dict(stage=name,source=key));write(dest/'progress.json',dict(completed_stages=completed,seconds=time.perf_counter()-started))
    try:
        old=OUT/'run361';binding=read(old/'upstream_verified.json');assert digest(old/'model.pkl')==binding['model_sha256']
        os.link(old/'model.pkl',dest/'model.pkl');shutil.copy2(old/'upstream_verified.json',dest/'upstream_verified.json')
        reused={}
        for key in ('21','BM79A_native_local','10','4'):
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
        # 25: only immutable graph inputs. Do not hardlink mutable phase logs.
        source=old/'source_25';target=dest/'source_25';target.mkdir()
        manifest=read(source/'edge_manifest.json');files={}
        for name,b in manifest['files'].items():
            assert digest(source/name)==b['sha256'];os.link(source/name,target/name);files[name]=b['sha256']
        os.link(source/'edge_manifest.json',target/'edge_manifest.json')
        assert digest(old/'screen_25.json')==manifest['screen_sha256']
        for name in ('screen_25.json','screen_25_receipt.json'):
            os.link(old/name,dest/name)
        reused['25_frozen_graph']=files
        assert read(old/'cache_equivalence.json')['status']=='EXACT_RELATION_COLUMN_CACHE_PASSED'
        oldcache=old/'source_snapshot/sjg_relation_column_cache_1173.py'
        oldfeatures=old/'source_snapshot/sjg_parent_set_features_1160.py'
        assert digest(oldcache)==digest(ROOT/'weld_pipeline/sjg_relation_column_cache_1173.py')
        assert digest(oldfeatures)==digest(ROOT/'weld_pipeline/sjg_parent_set_features_1160.py')
        shutil.copy2(old/'cache_equivalence.json',dest/'cache_equivalence.json')
        write(dest/'frozen_sources_reused.json',reused)
        snapshot=dest/'source_snapshot';snapshot.mkdir()
        for number in (1158,1159,1160,1161,1162,1164,1165,1168,1169,1170,1173,1174,1176,1177):
            for path in (ROOT/'weld_pipeline').glob(f'*_{number}.py'):shutil.copy2(path,snapshot/path.name)
        stage('certificate_control')
        # Keep 25's previous failed solve + diagnostic within its development allowance.
        stage('solve','25',60.-1.198569-1.0685577);stage('decode','25');stage('match','25')
        for key in ('36','42'):
            stage('screen',key);stage('graph',key);stage('solve',key,60.);stage('decode',key);stage('match',key)
        stage('coverage')
        stage('train_selector',limit=180.)
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
