"""Single Pro336 fit with the original 180-second owned-job budget."""
import argparse,shutil
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc
from weld_pipeline.sjg_owned_watchdog_1101 import supervise
from weld_pipeline.sjg_seven_relative_prepare_1146 import KEYS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run345');args=ap.parse_args();dest=OUT/args.run
    gate=read(OUT/'run342/receipt.json');assert gate['status']=='SEVEN_RELATIVE_K12_SUPERVISION_READY'
    assert read(OUT/'pro_review_run336/receipt.json')['assistant_sentinel_verified']
    controls=read(OUT/'run342/runtime_controls.json');assert controls['passed']
    assert all(digest(ROOT/'weld_pipeline'/name)==sha for name,sha in controls['code_bindings'].items())
    for data in (read(OUT/'run338/progress.json'),read(OUT/'run339/receipt.json')):
        for case in data['cases']:
            for change in case['transforms']:
                assert change['maximum_probability_difference']==0 and change['admission_changes']==0 and change['all_rows_endpoint_parameter_travel_max_mm']==0
    dest.mkdir(exist_ok=False)
    with (OUT/'single_relative_ordinal_CPU_trial_Pro336.json').open('x',encoding='utf-8') as stream:stream.write(args.run+'\n'+utc())
    snap=dest/'source_snapshot';snap.mkdir();bindings={}
    for number in (904,907,1081,1100,1101,1104,1107,1111,1112,1116,1125,1128,1129,1130,1139,1146,1147,1149,1150,1151,1152,1153):
        for file in (ROOT/'weld_pipeline').glob(f'*_{number}.py'):
            shutil.copy2(file,snap/file.name);bindings[file.name]=digest(file)
    preparation=sum(read(OUT/run/'failure.json')['watchdog']['wall_seconds'] for run in ('run341','run343'))
    preparation+=read(OUT/'run344/receipt.json')['external_wall_seconds']+gate['external_wall_seconds']
    write(dest/'registered_protocol.json',dict(status='PRO336_SEVEN_RELATIVE_K12_ONCE',TRAIN=list(KEYS),G=363,full_G=500,K=12,
        object_columns=152,endpoint_columns=153,connection_columns=32,trees=64,depths=[32,32,16],seed=1031,
        original_resource_guards=True,wall_seconds=180,fit_calls_each=1,compatible_weights_only=True,
        no_GT_endpoint_overrides=True,no_endpoint_sort_or_snap=True,no_DEV_training=True,
        preparation_including_failed_attempts_seconds=preparation,code_bindings=bindings,utc=utc()))
    result=supervise([str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m',
        'weld_pipeline.sjg_relative_ordinal_train_1151','--run',args.run],dest,180.)
    if result['termination_reason'] or result['exit_code']!=0:
        write(dest/'failure.json',dict(status='RELATIVE_ORDINAL_TRAINING_INCOMPLETE',watchdog=result,retry_permitted=False))
        raise RuntimeError('Relative ordinal training incomplete')
    worker=read(dest/'worker_receipt.json');assert worker['forest_fit_calls']==3 and digest(dest/'model.pkl')==worker['model_sha256']
    write(dest/'receipt.json',dict(worker,status='THREE_RELATIVE_ORDINAL_FORESTS_TRAINING_COMPLETE',external_wall_seconds=result['wall_seconds'],
        preparation_including_failed_attempts_seconds=preparation,total_current_attempts_seconds=preparation+result['wall_seconds']))
    print('RELATIVE_ORDINAL_TRAINING_COMPLETE',result['wall_seconds'],flush=True)


if __name__=='__main__':main()
