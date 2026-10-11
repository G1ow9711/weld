"""Paired TRAIN diagnostic: freeze both arms before reading the diagnostic target."""
import argparse
import os
import subprocess
import traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT, OUT, read, write, digest, utc, check_resources


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    stages=[('weld_pipeline.sjg_endpoint_loss_pair_958','run152',[],'ENDPOINT_PAIRED_TRAINING_COMPLETE'),
        ('weld_pipeline.sjg_local_event_infer_931','run153',['--data-run','run133','--training-run','run152/weighted_control','--proposal-run','run132','--sources','21'],'LOCAL_TYPED_DIAGNOSTIC_PREDICTIONS_FROZEN'),
        ('weld_pipeline.sjg_local_event_infer_931','run154',['--data-run','run133','--training-run','run152/unit_endpoint_weight','--proposal-run','run132','--sources','21'],'LOCAL_TYPED_DIAGNOSTIC_PREDICTIONS_FROZEN')]
    write(dest/'registered_protocol.json',dict(stage='PAIRED_TRAIN_ENDPOINT_DIAGNOSTIC',stages=stages,
        diagnostic_source='21',role='TRAIN',not_complete_cohort=True,all_arms_frozen_before_target_read=True,
        data_expansion=False,original_eight_G=496,full_contract_pass=False,source_sha256=digest(__file__),start_utc=utc()))
    done=[]
    try:
        for module,run,extra,status in stages:
            assert not (OUT/run).exists();check_resources()
            print('PAIR_STAGE_START',run,module,flush=True);write(dest/'progress.json',dict(run=run,completed=done,time_utc=utc()))
            subprocess.run([str(ROOT/'.venv-ptv3/Scripts/python.exe'),'-u','-m',module,'--run',run,*extra],cwd=ROOT,check=True)
            assert read(OUT/run/'receipt.json')['status']==status;done.append(run)
        for run in ('run153','run154'):
            folder=OUT/run/'source_21';freeze=read(folder/'input_output_freeze.json')
            for file,sha in freeze['files'].items():assert digest(folder/file)==sha
        from weld_pipeline.sjg_local_event_score_932 import geometric_match
        case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']=='21')
        assert digest(OUT/case['target_file'])==case['target_sha256'];target=read(OUT/case['target_file'])
        scores={arm:geometric_match(read(OUT/run/'source_21/predictions.json'),target) for arm,run in [('weighted_control','run153'),('unit_endpoint_weight','run154')]}
        for arm,score in scores.items():print('TRAIN_DIAGNOSTIC',arm,'TP',score['TP'],'/',score['G'],'FP',score['FP'],flush=True)
        write(dest/'local_review.json',dict(scores=scores,old_same_source=dict(TP=0,G=1,FP=242),
            whole_cohort_accuracy_NOT_MEASURED=True,external_consultation_sent=False,
            next_action='Use paired evidence to decide whether endpoint weighting or feature observability needs further repair; full cohort and contract gates remain required.'))
        write(dest/'receipt.json',dict(status='PAIRED_TRAIN_ENDPOINT_DIAGNOSIS_COMPLETE',completed=done,scores=scores,
            whole_cohort_accuracy_NOT_MEASURED=True,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',completed=done,error=traceback.format_exc(),end_utc=utc()));raise

if __name__=='__main__':main()
