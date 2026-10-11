"""Sequential binding and native backbone training in one desktop terminal."""
import argparse
import subprocess
import traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    stages=[('weld_pipeline.sjg_native_surface_baseline_969','run164','base','NATIVE_TRAIN_SURFACE_BASELINE_MEASURED'),
            ('weld_pipeline.sjg_native_structural_blocks_965','run161','base','NATIVE_STRUCTURAL_TRAIN_BLOCKS_READY'),
            ('weld_pipeline.sjg_native_structural_train_966','run162','ptv3','NATIVE_PTV3_STRUCTURAL_TRAINING_COMPLETE')]
    write(dest/'registered_protocol.json',dict(stage='NATIVE_STRUCTURAL_BACKBONE_ROUND',stages=stages,
        teacher_run='run160',original_eight_G=496,BM_weld_TRAIN_kept=True,epochs=2,
        source_sha256=digest(__file__),full_contract_pass=False,start_utc=utc()))
    completed=[]
    try:
        assert read(OUT/'run160/receipt.json')['status']=='NATIVE_TRAIN_PLANAR_TEACHERS_READY'
        for module,run,runtime,expected in stages:
            assert not (OUT/run).exists();check_resources()
            executable=ROOT/('.venv-ptv3/Scripts/python.exe' if runtime=='ptv3' else 'tools/python/cpython-3.10-windows-x86_64-none/python.exe')
            import os
            env=os.environ.copy();paths=[str(ROOT)]
            if runtime=='ptv3':paths.append(str(ROOT/'.venv-ptv3/Lib/site-packages'))
            paths.append(str(ROOT/'.venv/Lib/site-packages'));env['PYTHONPATH']=';'.join(paths)
            print('NATIVE_STRUCTURE_STAGE',run,module,flush=True);write(dest/'progress.json',dict(stage=run,completed=completed,time_utc=utc()))
            subprocess.run([str(executable),'-u','-m',module,'--run',run],cwd=ROOT,env=env,check=True)
            assert read(OUT/run/'receipt.json')['status']==expected;completed.append(dict(run=run,receipt_sha256=digest(OUT/run/'receipt.json')))
        write(dest/'receipt.json',dict(status='NATIVE_STRUCTURAL_BACKBONE_TRAINED_FREE_INFERENCE_PENDING',completed=completed,
            free_XYZ_validation_required=True,new_extraction_accuracy_not_measured=True,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',completed=completed,error=traceback.format_exc(),end_utc=utc()));raise

if __name__=='__main__':main()
