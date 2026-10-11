"""Single Pro301 endpoint refit under original external resource limits."""
import argparse,json,os,shutil,traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,digest,utc
from weld_pipeline.sjg_owned_watchdog_1101 import supervise,atomic_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run302');args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False)
    pro=read(OUT/'pro_review_run301/receipt.json');assert pro['assistant_sentinel_verified'] and pro['generation_stopped']
    assert read(OUT/'run298/receipt.json')['status']=='CONDITIONAL_MODEL_CONNECTION_REFIT_COMPLETE'
    assert read(OUT/'run283_verified/receipt.json')['watchdog_code_sha256']==digest(ROOT/'weld_pipeline/sjg_owned_watchdog_1101.py')
    with (OUT/'single_CPU_trial_Pro301.json').open('x',encoding='utf-8') as f:
        json.dump(dict(run=args.run,wall_seconds=180,endpoint_fit_calls=1,depth=32,utc=utc()),f)
    snapshot=dest/'source_snapshot';snapshot.mkdir();bindings={}
    for number in (904,907,1081,1100,1101,1104,1105,1106,1107,1108,1109,1111,1112,1116,1119,1120):
        for path in (ROOT/'weld_pipeline').glob(f'*_{number}.py'):
            shutil.copy2(path,snapshot/path.name);bindings[path.name]=digest(path)
    atomic_json(dest/'registered_protocol.json',dict(status='ONE_ENDPOINT_DEPTH32_REFIT_REGISTERED',
        base_model_run='run298',base_model_sha256=read(OUT/'run298/receipt.json')['model_sha256'],
        single_changed_parameter={'endpoint.max_depth':[16,32]},endpoint_fit_calls=1,object_fit_calls=0,connection_fit_calls=0,
        TRAIN=['21','BM79A_native_local','10'],TRAIN_G=213,positive_rows=782,DEV='64',full_pilot_G=500,
        presence_threshold=.5,shared_distance_mm=6.,rebuild_all_runtime_edges=True,all_positive_nodes_fixed_selected=True,
        wall_budget_seconds=180,includes_import_load_fit_save_reload_verify=True,code_bindings=bindings,
        review_receipt_sha256=digest(OUT/'pro_review_run301/receipt.json'),retry_on_failure=False,full_contract_pass=False,utc=utc()))
    os.environ['PYTHONPATH']=';'.join(map(str,(ROOT,ROOT/'.venv/Lib/site-packages')))
    command=[str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m',
        'weld_pipeline.sjg_endpoint_depth32_train_1119','--run',args.run]
    try:
        result=supervise(command,dest,180.)
        if result['termination_reason'] or result['exit_code']!=0 or result['wall_seconds']>180 or not (dest/'worker_receipt.json').exists():
            atomic_json(dest/'failure.json',dict(status='ENDPOINT_DEPTH32_REFIT_INCOMPLETE',watchdog=result,
                last_phase=read(dest/'worker_phase.json') if (dest/'worker_phase.json').exists() else None,retry_permitted=False,utc=utc()))
            raise RuntimeError('Single endpoint refit failed within original boundary')
        worker=read(dest/'worker_receipt.json');assert worker['forest_fit_calls']==1 and worker['frozen_components_verified_after_save_reload']
        assert digest(dest/'model.pkl')==worker['model_sha256']
        atomic_json(dest/'receipt.json',dict(worker,status='CONDITIONAL_MODEL_ENDPOINT_REFIT_COMPLETE',
            external_wall_seconds=result['wall_seconds'],external_budget_pass=True,full_pilot_G=500,full_contract_pass=False))
        print('ONE_ENDPOINT_FOREST_REFIT_COMPLETE_EXTERNAL_SECONDS',result['wall_seconds'],flush=True)
    except BaseException:
        atomic_json(dest/'controller_error.json',dict(error=traceback.format_exc(),utc=utc()));raise


if __name__=='__main__':main()
