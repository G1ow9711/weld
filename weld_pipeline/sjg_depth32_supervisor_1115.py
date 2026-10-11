"""Single registered depth-32 object refit with unchanged resource guards."""
import argparse
import json
import os
import shutil
import traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,digest,utc
from weld_pipeline.sjg_owned_watchdog_1101 import supervise,atomic_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run294');args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False)
    pro=read(OUT/'pro_review_run291/receipt.json');assert pro['assistant_sentinel_verified'] and pro['generation_stopped']
    assert read(OUT/'run289/receipt.json')['status']=='THREE_CPU_FORESTS_TRAINING_COMPLETE'
    control=read(OUT/'run283_verified/receipt.json')
    assert control['watchdog_code_sha256']==digest(ROOT/'weld_pipeline/sjg_owned_watchdog_1101.py')
    with (OUT/'single_CPU_trial_Pro291.json').open('x',encoding='utf-8') as f:
        json.dump(dict(run=args.run,wall_seconds=180,object_fit_calls=1,depth=32,utc=utc()),f)
    snapshot=dest/'source_snapshot';snapshot.mkdir();bindings={}
    for number in (904,907,1081,1100,1101,1104,1105,1106,1107,1108,1109,1111,1112,1114,1115):
        for path in (ROOT/'weld_pipeline').glob(f'*_{number}.py'):
            shutil.copy2(path,snapshot/path.name);bindings[path.name]=digest(path)
    atomic_json(dest/'registered_protocol.json',dict(status='ONE_OBJECT_DEPTH32_REFIT_REGISTERED',
        base_model_run='run289',base_model_sha256=read(OUT/'run289/receipt.json')['model_sha256'],
        single_changed_parameter={'object.max_depth':[16,32]},object_fit_calls=1,
        endpoint_fit_calls=0,connection_fit_calls=0,complete_original_object_rows=336344,
        TRAIN=['21','BM79A_native_local','10'],TRAIN_G=213,DEV='64',full_pilot_G=500,
        presence_threshold=.5,all_positive_nodes_fixed_selected=True,wall_budget_seconds=180,
        includes_import_load_object_fit_save_reload_verify=True,code_bindings=bindings,
        review_receipt_sha256=digest(OUT/'pro_review_run291/receipt.json'),retry_on_failure=False,
        raw_PCD_uncached_contract_runtime='NOT_MEASURED',full_contract_pass=False,utc=utc()))
    os.environ['PYTHONPATH']=';'.join(map(str,(ROOT,ROOT/'.venv/Lib/site-packages')))
    command=[str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m',
             'weld_pipeline.sjg_object_depth32_train_1114','--run',args.run]
    try:
        result=supervise(command,dest,180.)
        if result['termination_reason'] or result['exit_code']!=0 or result['wall_seconds']>180 or not (dest/'worker_receipt.json').exists():
            phase=read(dest/'worker_phase.json') if (dest/'worker_phase.json').exists() else {'phase':'NOT_OBSERVED'}
            atomic_json(dest/'failure.json',dict(status='DEPTH32_OBJECT_REFIT_INCOMPLETE',watchdog=result,
                last_phase=phase,retry_permitted=False,extraction_correct_rate=None,extraction_precision_mm=None,utc=utc()))
            raise RuntimeError('Single depth-32 refit failed within the original boundary')
        worker=read(dest/'worker_receipt.json');assert worker['forest_fit_calls']==1 and worker['frozen_components_verified_after_save_reload']
        assert digest(dest/'model.pkl')==worker['model_sha256']
        atomic_json(dest/'receipt.json',dict(worker,status='CONDITIONAL_MODEL_OBJECT_REFIT_COMPLETE',
            external_wall_seconds=result['wall_seconds'],external_budget_pass=True,full_pilot_G=500,full_contract_pass=False))
        print('ONE_OBJECT_FOREST_REFIT_COMPLETE_EXTERNAL_SECONDS',result['wall_seconds'],flush=True)
    except BaseException:
        atomic_json(dest/'controller_error.json',dict(error=traceback.format_exc(),utc=utc()));raise


if __name__=='__main__':main()
