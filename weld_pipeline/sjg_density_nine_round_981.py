"""Unchanged nine-case density decoding followed by the same TRAIN-only score."""
import argparse
import os
import subprocess
import traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    stages=[('weld_pipeline.sjg_density_partition_nine_980','run174',[],'NATIVE_NINE_INSTANCE_HYPOTHESES_FROZEN'),
            ('weld_pipeline.sjg_native_instance_evaluate_975','run175',['--prediction-run','run174'],'NATIVE_TRAIN_INSTANCE_DIAGNOSIS_COMPLETE')]
    write(dest/'registered_protocol.json',dict(stage='UNCHANGED_NINE_DENSITY_DECODER_VALIDATION',stages=stages,
        development_TRAIN_only=['21'],fixed_decoder_source_sha256=digest(ROOT/'weld_pipeline/sjg_density_modes_978.py'),
        baseline_prediction_run='run169',baseline_score_run='run170',checkpoint_unchanged='run162',
        feature_run_unchanged='run165',thresholds_unchanged=True,additional_optimizer_updates=0,
        all_nine_before_reference_reads=True,original_eight_G=496,supplemental_BM_G=4,
        full_contract_pass=False,start_utc=utc()))
    completed=[]
    try:
        executable=ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'
        for module,run,extra,status in stages:
            assert not (OUT/run).exists();check_resources()
            print('DENSITY_NINE_STAGE',run,module,flush=True)
            write(dest/'progress.json',dict(run=run,completed=completed,time_utc=utc()))
            subprocess.run([str(executable),'-u','-m',module,'--run',run,*extra],cwd=ROOT,env=os.environ.copy(),check=True)
            assert read(OUT/run/'receipt.json')['status']==status
            completed.append(dict(run=run,receipt_sha256=digest(OUT/run/'receipt.json')))
        write(dest/'receipt.json',dict(status='DENSITY_NINE_STRUCTURE_DIAGNOSIS_COMPLETE_WELD_PENDING',
            completed=completed,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',completed=completed,error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
