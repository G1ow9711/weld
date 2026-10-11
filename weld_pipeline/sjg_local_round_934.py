"""One visible, sequential small-pilot data/train/free-test round."""
import argparse
import os
import subprocess
import traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    stages=[
        ('controls','weld_pipeline.sjg_local_event_controls_933',args.run+'/controls','ptv3','LOCAL_EVENT_CONTROLS_PASSED'),
        ('data','weld_pipeline.sjg_local_training_data_918','run116','base','LOCAL_TYPED_GRAPH_TRAINING_DATA_READY'),
        ('train','weld_pipeline.sjg_local_event_train_930','run117','ptv3','LOCAL_TYPED_EVENT_TRAINING_COMPLETE'),
        ('free_prediction','weld_pipeline.sjg_local_event_infer_931','run118','ptv3','LOCAL_TYPED_FREE_PREDICTIONS_FROZEN'),
        ('independent_score','weld_pipeline.sjg_local_event_score_932','run119','base','LOCAL_EVENT_GEOMETRY_DIAGNOSED')]
    write(dest/'registered_protocol.json',dict(stage='SEQUENTIAL_SMALL_PILOT_ROUND',stages=stages,
        original_G=496,additional_TRAIN_G=4,training_epochs=8,train_queries_per_case=2048,batch_queries=256,
        independent_stages_serial=True,source_sha256=digest(__file__),
        no_expansion_before_contract_gates=True,no_external_consultation=True,full_contract_pass=False,start_utc=utc()))
    completed=[]
    try:
        assert read(OUT/'run115/receipt.json')['status']=='LOCAL_RANK2_TYPED_CURVES_FROZEN'
        for name,module,run,runtime,expected in stages:
            if (OUT/run).exists():raise RuntimeError('Refusing to overwrite existing stage '+run)
            check_resources();env=os.environ.copy()
            python=ROOT/('.venv-ptv3/Scripts/python.exe' if runtime=='ptv3' else 'tools/python/cpython-3.10-windows-x86_64-none/python.exe')
            paths=[str(ROOT)]
            if runtime=='ptv3':paths.append(str(ROOT/'.venv-ptv3/Lib/site-packages'))
            paths.append(str(ROOT/'.venv/Lib/site-packages'));env['PYTHONPATH']=';'.join(paths)
            write(dest/'progress.json',dict(current_stage=name,current_run=run,completed=completed,start_utc=utc()))
            print('PILOT_STAGE_START',name,run,flush=True)
            subprocess.run([str(python),'-u','-m',module,'--run',run],cwd=str(ROOT),env=env,check=True)
            receipt=read(OUT/run/'receipt.json')
            if receipt['status']!=expected:raise RuntimeError('Unexpected terminal receipt '+run)
            completed.append(dict(stage=name,run=run,receipt_sha256=digest(OUT/run/'receipt.json')))
            print('PILOT_STAGE_COMPLETE',name,run,flush=True)
        score=read(OUT/'run119/receipt.json');training=read(OUT/'run117/receipt.json')
        write(dest/'local_round_review.json',dict(results=score['groups'],optimizer_updates=training['optimizer_updates'],
            completed_epochs=training['epochs'],full_contract_pass=False,
            unresolved_contract_items=score['unmet_or_not_run'],external_advice_sent=False,
            next_decision='Review missing carrier coverage, missed/false events, and parent/material schema before the next optimization.'))
        write(dest/'receipt.json',dict(status='SMALL_LOCAL_EVENT_ROUND_COMPLETE_CONTRACT_NOT_PASSED',completed=completed,
            optimizer_updates=training['optimizer_updates'],groups=score['groups'],full_contract_pass=False,end_utc=utc()))
        print('PILOT_ROUND_COMPLETE','CONTRACT_NOT_PASSED','LOCAL_REVIEW_SAVED',flush=True)
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',completed=completed,error=traceback.format_exc(),end_utc=utc()));raise

if __name__=='__main__':main()
