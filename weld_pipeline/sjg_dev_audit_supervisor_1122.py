"""One 180s Pro306 diagnostic lease; preserve NOT_PROVEN on timeout."""
import argparse,json,os,shutil,traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,digest,utc
from weld_pipeline.sjg_owned_watchdog_1101 import supervise,atomic_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run307');args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False)
    pro=read(OUT/'pro_review_run306/receipt.json');assert pro['assistant_sentinel_verified'] and pro['generation_stopped']
    assert read(OUT/'run283_verified/receipt.json')['watchdog_code_sha256']==digest(ROOT/'weld_pipeline/sjg_owned_watchdog_1101.py')
    with (OUT/'single_diagnostic_Pro306.json').open('x',encoding='utf-8') as f:json.dump(dict(run=args.run,wall_seconds=180,new_fit_calls=0,utc=utc()),f)
    snapshot=dest/'source_snapshot';snapshot.mkdir();bindings={}
    for number in (904,907,915,1101,1104,1105,1107,1109,1111,1121,1122):
        for path in (ROOT/'weld_pipeline').glob(f'*_{number}.py'):
            shutil.copy2(path,snapshot/path.name);bindings[path.name]=digest(path)
    atomic_json(dest/'registered_protocol.json',dict(status='ONE_FROZEN_DEV_AUDIT_REGISTERED',
        model_run='run302',input_run='run293',prediction_run='run306',source='64',G=26,new_fit_calls=0,
        wall_budget_seconds=180,code_bindings=bindings,retry_on_timeout=False,
        unproved_status='NOT_PROVEN',full_pilot_G=500,full_contract_pass=False,utc=utc()))
    os.environ['PYTHONPATH']=';'.join(map(str,(ROOT,ROOT/'.venv/Lib/site-packages')))
    command=[str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m','weld_pipeline.sjg_dev_coverage_audit_1121','--run',args.run]
    result=supervise(command,dest,180.)
    if result['termination_reason'] or result['exit_code']!=0 or not (dest/'worker_receipt.json').exists():
        atomic_json(dest/'receipt.json',dict(status='DEV_DIAGNOSTIC_PARTIAL_NOT_PROVEN',watchdog=result,
            rows=read(dest/'parent_progress.json')['rows'] if (dest/'parent_progress.json').exists() else [],
            new_fit_calls=0,retry_permitted=False,not_extraction_accuracy=True,utc=utc()))
        raise RuntimeError('One diagnostic incomplete; preserve proofs, no budget extension')
    worker=read(dest/'worker_receipt.json')
    atomic_json(dest/'receipt.json',dict(worker,external_wall_seconds=result['wall_seconds'],external_budget_pass=True))
    print('DEV_DIAGNOSTIC_COMPLETE',result['wall_seconds'],worker['full_parent_coverage_counts'],flush=True)


if __name__=='__main__':main()
