"""Freeze all nine native predictions, then diagnose TRAIN surface learning."""
import argparse
import os
import subprocess
import traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    stages=[('weld_pipeline.sjg_native_material_features_970','run165','ptv3',[],'NATIVE_MATERIAL_SURFACE_FEATURES_FROZEN'),
        ('weld_pipeline.sjg_native_material_merge_971','run166','base',[],'NATIVE_MATERIAL_POINT_FEATURES_READY'),
        ('weld_pipeline.sjg_native_surface_baseline_969','run168','base',['--feature-run','run166'],'NATIVE_TRAIN_SURFACE_EVALUATED')]
    write(dest/'registered_protocol.json',dict(stage='NATIVE_NINE_FEATURE_FREEZE_AND_SURFACE_DIAGNOSIS',stages=stages,
        training_run='run162',all_nine_free_outputs_before_labels=True,original_eight_G=496,
        point_metrics_do_not_replace_weld_acceptance=True,source_sha256=digest(__file__),start_utc=utc()))
    completed=[]
    try:
        assert read(OUT/'run162/receipt.json')['status']=='NATIVE_PTV3_STRUCTURAL_TRAINING_COMPLETE'
        for module,run,runtime,extra,status in stages:
            assert not (OUT/run).exists();check_resources()
            executable=ROOT/('.venv-ptv3/Scripts/python.exe' if runtime=='ptv3' else 'tools/python/cpython-3.10-windows-x86_64-none/python.exe')
            env=os.environ.copy();paths=[str(ROOT)]
            if runtime=='ptv3':paths.append(str(ROOT/'.venv-ptv3/Lib/site-packages'))
            paths.append(str(ROOT/'.venv/Lib/site-packages'));env['PYTHONPATH']=';'.join(paths)
            print('NATIVE_FREE_VALIDATION_STAGE',run,module,flush=True);write(dest/'progress.json',dict(run=run,completed=completed,time_utc=utc()))
            subprocess.run([str(executable),'-u','-m',module,'--run',run,*extra],cwd=ROOT,env=env,check=True)
            assert read(OUT/run/'receipt.json')['status']==status;completed.append(dict(run=run,receipt_sha256=digest(OUT/run/'receipt.json')))
        old=read(OUT/'run164/receipt.json');new=read(OUT/'run168/receipt.json')
        write(dest/'receipt.json',dict(status='NATIVE_SURFACE_LEARNING_DIAGNOSED_WELD_VALIDATION_PENDING',completed=completed,
            baseline_boundary=old['boundary_total'],new_boundary=new['boundary_total'],
            formal_independence_not_claimed=True,material_global_partition_and_weld_acceptance_pending=True,
            full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',completed=completed,error=traceback.format_exc(),end_utc=utc()));raise

if __name__=='__main__':main()
