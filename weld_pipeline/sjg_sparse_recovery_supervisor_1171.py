"""Resume only Pro359's explicitly allotted balance of the failed 900s window."""
import argparse,ast,hashlib,os,shutil,time,traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,resources,utc
from weld_pipeline.sjg_owned_watchdog_1101 import supervise


def main():
    started=time.perf_counter()
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);completed=[]
    pro=read(OUT/'pro_review_run359/receipt.json')
    assert pro['assistant_sentinel_verified'] and pro['generation_stopped']
    assert hashlib.sha256((OUT/'pro_review_run359/answer.md').read_text(encoding='utf-8').encode()).hexdigest()==pro['answer_sha256']
    assert digest(OUT/'pro_review_run359/packet.md')==pro['packet_sha256']
    previous=read(OUT/'run359/failure.json');budget=900.-previous['recovery_seconds'];deadline=started+budget
    assert 620.<budget<621.
    for number in (1168,1169,1170,1171,1172):
        for path in (ROOT/'weld_pipeline').glob(f'*_{number}.py'):ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    write(dest/'registered_protocol.json',dict(status='PRO359_EXPLICIT_BALANCE_RECOVERY',window_seconds=budget,
        original_window=900.,run359_charged_wall_seconds=previous['recovery_seconds'],
        prior_charged_wall_seconds=previous['measured_cumulative_execution_seconds'],diagnostic_reservation_seconds=60,
        diagnostic_actual_external_seconds=None,forest_fit_calls_allowed=0,selector_updates=112,selector_sublimit=180,
        path_development_sublimit=60,formal_uncached_raw_PCD_limit=10,tied_optimum_changes_must_be_proved=True,
        all_legal_edges_scored_and_persisted=True,TRAIN_G=363,pilot_G=500,full_contract_pass=False,utc=utc()))
    checks=[]
    for i in range(3):
        if i:time.sleep(1)
        r=resources();r['disk_free_bytes']=shutil.disk_usage(ROOT).free
        r['passed']=r['available_physical_bytes']>=3758096384 and r['remaining_commit_bytes']>=5368709120 and r['disk_free_bytes']>=5637144576
        checks.append(r);print('STARTUP_HEADROOM',i+1,r,flush=True)
    write(dest/'startup_headroom.json',dict(checks=checks,passed=all(v['passed'] for v in checks)))
    if not all(v['passed'] for v in checks):
        write(dest/'receipt.json',dict(status='NO_START_RESOURCE_HEADROOM',forest_fit_calls=0,seconds=time.perf_counter()-started,full_contract_pass=False));return
    with (OUT/'single_balance_recovery_Pro359.json').open('x',encoding='utf-8') as stream:stream.write(args.run+'\n'+utc())
    def stage(name,key='',limit=None):
        remaining=deadline-time.perf_counter()
        if remaining<=0:raise TimeoutError('Pro359 balance exhausted')
        if limit is not None:remaining=min(remaining,limit)
        if name in ('mechanism','graph','solve','decode','path_control','test_graph','test_solve','test_decode'):module='weld_pipeline.sjg_sparse_recovery_worker_1170'
        elif name=='screen':module='weld_pipeline.sjg_recall_recovery_worker_1165'
        else:module='weld_pipeline.sjg_recall_set_worker_1162'
        directory=dest/f'job_{name}_{key}';directory.mkdir(exist_ok=False)
        command=[str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m',module,'--run',args.run,'--stage',name,'--source',key]
        print('BALANCE_STAGE',name,key,'SECONDS_AVAILABLE',remaining,flush=True)
        result=supervise(command,directory,remaining)
        if result['termination_reason'] or result['exit_code']!=0:
            write(dest/'failure.json',dict(status='BALANCE_STAGE_FAILED',stage=name,source=key,watchdog=result,
                completed_stages=completed,window_seconds=time.perf_counter()-started,window_limit=budget,automatic_retry=False,full_contract_pass=False))
            raise RuntimeError('Stage failed; explicit balance is stopped')
        assert read(dest/f'stage_{name}_{key}_complete.json')['status']=='STAGE_COMPLETE'
        completed.append(dict(stage=name,source=key));write(dest/'progress.json',dict(completed_stages=completed,seconds=time.perf_counter()-started))
    def capacity(key):
        value=read(dest/f'source_{key}/compatibility.json')
        if not value['all_parents_coverable']:
            write(dest/'receipt.json',dict(status='UPSTREAM_COMPLETE_PARENT_CAPACITY_FAILED',source=key,candidate_coverage=value,
                forest_fit_calls=0,selector_updates=0,seconds=time.perf_counter()-started,full_contract_pass=False));return False
        return True
    try:
        binding=read(OUT/'run358/upstream_verified.json');assert digest(OUT/'run358/model.pkl')==binding['model_sha256']
        os.link(OUT/'run358/model.pkl',dest/'model.pkl');shutil.copy2(OUT/'run358/upstream_verified.json',dest/'upstream_verified.json')
        for key in ('21','BM79A_native_local'):
            old=OUT/'run358'/f'source_{key}';r=read(old/'receipt.json');saved=read(old/'selected_graph.json')
            assert r['upstream_model_sha256']==binding['model_sha256'] and read(old/'screen.json')['invalid_nodes']==0
            screen=dict(nodes=saved['nodes'],invalid=[],model_sha256=binding['model_sha256'],
                feature_receipt_sha256=r['input_feature_receipt_sha256'],GT_reads=0,fit_calls=0)
            path=dest/f'screen_{key}.json';write(path,screen)
            write(dest/f'screen_{key}_receipt.json',dict(screen_sha256=digest(path),source_nodes_sha256=digest(old/'selected_graph.json')))
        old=OUT/'run358/memory_diagnosis';r=read(old/'receipt.json');assert digest(old/'frozen_screen.json')==r['screen_sha256']
        os.link(old/'frozen_screen.json',dest/'screen_10.json');write(dest/'screen_10_receipt.json',dict(screen_sha256=r['screen_sha256']))
        snapshot=dest/'source_snapshot';snapshot.mkdir()
        for number in (1104,1107,1111,1125,1129,1139,1150,1158,1159,1160,1161,1162,1164,1165,1168,1169,1170,1171,1172):
            for path in (ROOT/'weld_pipeline').glob(f'*_{number}.py'):shutil.copy2(path,snapshot/path.name)
        stage('mechanism')
        for key in ('21','BM79A_native_local'):
            stage('graph',key);stage('solve',key,60.);stage('decode',key);stage('path_control',key);stage('match',key)
            if not capacity(key):return
        for key in ('10','4','25','36','42'):
            if key!='10':stage('screen',key)
            stage('graph',key);stage('solve',key,60.);stage('decode',key);stage('match',key)
            if key=='10' and not capacity(key):return
        stage('train_selector',limit=180.)
        results=[]
        for key in ('21','BM79A_native_local','10','64'):
            if key=='64':
                stage('screen',key);stage('test_graph',key);stage('test_solve',key,60.);stage('test_decode',key)
            else:stage('test_upstream',key)
            stage('select',key,60.);stage('score',key)
            result=read(dest/f'test_{key}/evaluation.json')['report'];results.append(result)
            passed=result['strict_TRAIN_gate_pass'] if key!='64' else result['TP']/result['G']>=.5 and result['FP']/result['G']<=.5
            if not passed:
                write(dest/'receipt.json',dict(status='BALANCE_FIRST_FAILED_FINAL_TEST',source=key,results=results,
                    forest_fit_calls=0,selector_updates=112,seconds=time.perf_counter()-started,full_contract_pass=False));return
        write(dest/'receipt.json',dict(status='PILOT_CONTINUATION_GATE_PASSED',results=results,forest_fit_calls=0,
            selector_updates=112,seconds=time.perf_counter()-started,full_contract_pass=False))
    except BaseException:
        if not (dest/'failure.json').exists():write(dest/'failure.json',dict(status='BALANCE_CONTROLLER_FAILED',error=traceback.format_exc(),
            completed_stages=completed,seconds=time.perf_counter()-started,automatic_retry=False,full_contract_pass=False))
        raise


if __name__=='__main__':main()
