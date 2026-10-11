"""Registered one-shot Pro314 seven-case training with owned 180 s watchdog."""
import argparse
import os
import shutil
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc
from weld_pipeline.sjg_owned_watchdog_1101 import supervise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run319');args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False)
    pro=read(OUT/'pro_review_run314/receipt.json');assert pro['assistant_sentinel_verified'] and pro['generation_stopped']
    target=read(OUT/'run318/receipt.json');assert target['status']=='ORDERED_INTERVAL_SUPERVISION_READY' and target['G']==363
    controls=read(OUT/'run318/runtime_controls.json');assert controls['passed']
    assert read(OUT/'run318/global_supervision_control.json')['passed']
    assert controls['runtime_sha256']==digest(ROOT/'weld_pipeline/sjg_interval_forest_runtime_1129.py')
    assert read(OUT/'run283_verified/receipt.json')['watchdog_code_sha256']==digest(ROOT/'weld_pipeline/sjg_owned_watchdog_1101.py')
    with (OUT/'single_interval_CPU_trial_Pro314.json').open('x',encoding='utf-8') as f:
        f.write(args.run+'\n'+utc())
    snapshot=dest/'source_snapshot';snapshot.mkdir();bindings={}
    for number in (904,907,1081,1100,1101,1104,1105,1106,1107,1108,1109,1111,1116,1125,1127,1128,1129,1130,1131):
        for path in (ROOT/'weld_pipeline').glob(f'*_{number}.py'):
            shutil.copy2(path,snapshot/path.name);bindings[path.name]=digest(path)
    write(dest/'registered_protocol.json',dict(status='ONE_SEVEN_TRAIN_INTERVAL_EXPERIMENT_REGISTERED',
        TRAIN=['21','4','10','25','36','42','BM79A_native_local'],TRAIN_G=363,DEV='64',full_pilot_G=500,
        K=target['K'],E=target['E'],wall_budget_seconds=180,fit_calls_each=1,forests=3,
        same_resource_guards=True,retry_on_failure=False,code_bindings=bindings,
        review_receipt_sha256=digest(OUT/'pro_review_run314/receipt.json'),
        runtime_controls_sha256=digest(OUT/'run318/runtime_controls.json'),
        global_supervision_controls_sha256=digest(OUT/'run318/global_supervision_control.json'),
        supervision_receipt_sha256=digest(OUT/'run318/receipt.json'),full_contract_pass=False))
    os.environ['PYTHONPATH']=';'.join(map(str,(ROOT,ROOT/'.venv/Lib/site-packages')))
    command=[str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m',
             'weld_pipeline.sjg_interval_forest_train_1130','--run',args.run]
    result=supervise(command,dest,180.)
    if result['termination_reason'] or result['exit_code']!=0 or result['wall_seconds']>180 or not (dest/'worker_receipt.json').exists():
        write(dest/'failure.json',dict(status='INTERVAL_FOREST_EXPERIMENT_INCOMPLETE',watchdog=result,retry_permitted=False,
              last_phase=read(dest/'worker_phase.json') if (dest/'worker_phase.json').exists() else None,full_contract_pass=False))
        raise RuntimeError('Registered interval experiment did not finish within original limits')
    worker=read(dest/'worker_receipt.json');assert worker['forest_fit_calls']==3
    assert digest(dest/'model.pkl')==worker['model_sha256']
    write(dest/'receipt.json',dict(worker,status='THREE_INTERVAL_FORESTS_TRAINING_COMPLETE',
          external_wall_seconds=result['wall_seconds'],external_budget_pass=True))
    print('SEVEN_TRAIN_INTERVAL_FORESTS_COMPLETE',result['wall_seconds'],flush=True)


if __name__=='__main__':main()
