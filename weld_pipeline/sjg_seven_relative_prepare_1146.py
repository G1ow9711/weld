"""Pro336: apply the unchanged 152-channel factory to four existing TRAIN cases."""
import argparse,gc,shutil,time,traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources

KEYS=('21','4','10','25','36','42','BM79A_native_local')
EXTRA=('4','25','36','42')


def feature_path(key):
    run='run341' if key=='4' else 'run344' if key in ('25','36','42') else 'run335' if key=='64' else 'run329'
    return OUT/run/f'source_{key}'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');args=ap.parse_args();dest=OUT/'run341'
    if args.worker:
        from weld_pipeline.sjg_relative_geometry_batch_1136 import prepare as relative
        from weld_pipeline.sjg_observed_face_features_1138 import prepare as observed
        started=time.perf_counter();base=OUT/'run340';base.mkdir(exist_ok=False)
        try:
            for key in EXTRA:
                check_resources();relative(key,base);gc.collect()
                observed(key,dest,prior_run='run340');gc.collect()
            cases=[]
            for key in KEYS:
                folder=feature_path(key);r=read(folder/'receipt.json')
                assert r['columns']==152 and digest(folder/'features.npy')==r['features_sha256']
                cases.append(dict(source=key,feature_run=folder.parent.name,receipt_sha256=digest(folder/'receipt.json'),**{k:v for k,v in r.items() if k!='source'}))
            write(dest/'all_relative_inputs_frozen_before_labels.json',dict(cases=cases,GT_reads=0,STEP_reads=0))
            write(dest/'worker_receipt.json',dict(status='SEVEN_RELATIVE_INPUTS_FROZEN',cases=cases,fit_calls=0,GT_reads=0,STEP_reads=0,
                seconds=time.perf_counter()-started,feature_definition_changed=False))
        except BaseException:write(dest/'worker_failure.json',dict(error=traceback.format_exc(),fit_calls=0));raise
        return
    assert read(OUT/'pro_review_run336/receipt.json')['assistant_sentinel_verified']
    assert read(OUT/'run339/receipt.json')['status']=='REMAINING_FROZEN_RIGID_OUTPUT_AUDIT_COMPLETE'
    dest.mkdir(exist_ok=False);snap=dest/'source_snapshot';snap.mkdir()
    for name in ('sjg_seven_relative_prepare_1146.py','sjg_relative_geometry_batch_1136.py','sjg_observed_face_features_1138.py'):
        shutil.copy2(ROOT/'weld_pipeline'/name,snap)
    write(dest/'registered_protocol.json',dict(status='PRO336_FOUR_ADDITIONAL_TRAIN_INPUTS',sources=list(KEYS),new_sources=list(EXTRA),
        columns=152,K=12,G=363,fit_calls=0,GT_reads=0,wall_seconds=180,utc=utc()))
    from weld_pipeline.sjg_owned_watchdog_1101 import supervise
    result=supervise([str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m',
        'weld_pipeline.sjg_seven_relative_prepare_1146','--worker'],dest,180.)
    if result['termination_reason'] or result['exit_code']!=0:
        write(dest/'failure.json',dict(status='SEVEN_RELATIVE_INPUTS_INCOMPLETE',watchdog=result));raise RuntimeError('Input preparation incomplete')
    r=read(dest/'worker_receipt.json');r['external_wall_seconds']=result['wall_seconds'];write(dest/'receipt.json',r)
    print('SEVEN_RELATIVE_INPUTS_READY',result['wall_seconds'],flush=True)


if __name__=='__main__':main()
