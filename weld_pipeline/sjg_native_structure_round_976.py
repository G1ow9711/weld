"""Single desktop stage: nine free partitions, then TRAIN structure diagnosis."""
import argparse
import os
import subprocess
import traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    stages=[('weld_pipeline.sjg_native_instance_freeze_974','run169','NATIVE_NINE_INSTANCE_HYPOTHESES_FROZEN'),
            ('weld_pipeline.sjg_native_instance_evaluate_975','run170','NATIVE_TRAIN_INSTANCE_DIAGNOSIS_COMPLETE')]
    write(dest/'registered_protocol.json',dict(stage='NATIVE_OVERLAP_STRUCTURE_DIAGNOSIS',stages=stages,
        features='run165',source_sha256=digest(__file__),all_nine_before_labels=True,
        original_eight_G=496,supplemental_BM_G=4,full_contract_pass=False,start_utc=utc()))
    completed=[]
    try:
        assert read(OUT/'run167/receipt.json')['status']=='NATIVE_SURFACE_LEARNING_DIAGNOSED_WELD_VALIDATION_PENDING'
        executable=ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'
        for module,run,status in stages:
            assert not (OUT/run).exists();check_resources()
            print('NATIVE_STRUCTURE_STAGE',run,module,flush=True)
            write(dest/'progress.json',dict(run=run,completed=completed,time_utc=utc()))
            subprocess.run([str(executable),'-u','-m',module,'--run',run],cwd=ROOT,env=os.environ.copy(),check=True)
            assert read(OUT/run/'receipt.json')['status']==status
            completed.append(dict(run=run,receipt_sha256=digest(OUT/run/'receipt.json')))
        write(dest/'receipt.json',dict(status='NATIVE_STRUCTURE_DIAGNOSIS_COMPLETE_WELD_VALIDATION_PENDING',
            completed=completed,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',completed=completed,error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
