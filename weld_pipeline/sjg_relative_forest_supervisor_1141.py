"""One registered relative-feature training trial with unchanged owned-job guards."""
import argparse
import shutil
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc
from weld_pipeline.sjg_owned_watchdog_1101 import supervise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run330');args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False)
    pro=read(OUT/'pro_review_run324/receipt.json');assert pro['assistant_sentinel_verified'] and pro['generation_stopped']
    gate=read(OUT/'run329/receipt.json');assert gate['status']=='RELATIVE_SUPERVISION_READY'
    assert gate['classification_conflict_rows']==0 and gate['endpoint_feasibility']=='PASSED'
    controls=read(OUT/'run329/runtime_controls.json');assert controls['passed']
    assert controls['runtime_sha256']==digest(ROOT/'weld_pipeline/sjg_relative_forest_runtime_1139.py')
    assert read(OUT/'run283_verified/receipt.json')['watchdog_code_sha256']==digest(ROOT/'weld_pipeline/sjg_owned_watchdog_1101.py')
    with (OUT/'single_relative_CPU_trial_Pro324.json').open('x',encoding='utf-8') as f:f.write(args.run+'\n'+utc())
    snapshot=dest/'source_snapshot';snapshot.mkdir();code={}
    for number in (904,907,1081,1100,1101,1104,1105,1106,1107,1108,1109,1111,1112,1116,1117,1129,1135,1138,1139,1140,1141,1142):
        for path in (ROOT/'weld_pipeline').glob(f'*_{number}.py'):
            shutil.copy2(path,snapshot/path.name);code[path.name]=digest(path)
    write(dest/'registered_protocol.json',dict(status='RELATIVE_OBSERVED_FACE_3TRAIN_V1',TRAIN=['21','BM79A_native_local','10'],G=213,
        full_pilot_G=500,DEV='64',object_endpoint_features=152,connection_features=32,forest_trees=64,
        depths=[32,32,16],seed=1031,min_leaf=1,max_features='sqrt',class_weight=None,bootstrap=False,n_jobs=1,
        warm_start=False,fit_calls_each=1,wall_budget_seconds=180,original_resource_guards=True,
        preparation_controls_seconds=gate['external_wall_seconds'],preparation_plus_training_reported=True,
        no_threshold_search=True,no_new_samples=True,no_DEV_labels=True,
        original_TRAIN_gate='TP213/G213, FP0, complete parent structure, all geometry <=3mm',
        stage_order=['21','BM79A_native_local','10','64'],stop_after_first_failed_gate=True,
        review_receipt_sha256=digest(OUT/'pro_review_run324/receipt.json'),
        local_representation_extension='known-face masking repair + relative observed support centroid/radius; TRAIN-only preflight',
        controls_sha256=digest(OUT/'run329/runtime_controls.json'),code_bindings=code,utc=utc()))
    result=supervise([str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m',
        'weld_pipeline.sjg_relative_forest_train_1140','--run',args.run],dest,180.)
    if result['termination_reason'] or result['exit_code']!=0:
        write(dest/'failure.json',dict(status='RELATIVE_TRAINING_INCOMPLETE',watchdog=result,retry_permitted=False));raise RuntimeError('Relative training incomplete')
    worker=read(dest/'worker_receipt.json');assert worker['forest_fit_calls']==3
    assert digest(dest/'model.pkl')==worker['model_sha256']
    write(dest/'receipt.json',dict(worker,status='THREE_RELATIVE_FORESTS_TRAINING_COMPLETE',external_wall_seconds=result['wall_seconds'],
        external_budget_pass=result['wall_seconds']<=180,preparation_controls_seconds=gate['external_wall_seconds'],
        current_preparation_plus_training_seconds=gate['external_wall_seconds']+result['wall_seconds']))
    print('RELATIVE_TRAINING_COMPLETE',result['wall_seconds'],flush=True)


if __name__=='__main__':main()
