"""Sequential small-pilot training and free validation on repaired native geometry."""
import argparse
import os
import subprocess
import traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources

def main(resume_after_data=False,resume_after_training=False):
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    stages=[
        ('controls','weld_pipeline.sjg_local_event_controls_933',args.run+'/controls','ptv3',[], 'LOCAL_EVENT_CONTROLS_PASSED'),
        ('data','weld_pipeline.sjg_local_training_data_918','run133','base',['--proposal-run','run132'],'LOCAL_TYPED_GRAPH_TRAINING_DATA_READY'),
        ('material','weld_pipeline.sjg_local_material_targets_952','run138','base',['--data-run','run133','--proposal-run','run132'],'LOCAL_MATERIAL_QUERY_TARGETS_COMPLETE'),
        ('train','weld_pipeline.sjg_local_event_train_930','run134','ptv3',['--data-run','run133','--relation-run','run138'],'LOCAL_TYPED_EVENT_TRAINING_COMPLETE'),
        ('free_prediction','weld_pipeline.sjg_local_event_infer_931','run135','ptv3',
            ['--data-run','run133','--training-run','run134','--proposal-run','run132'],'LOCAL_TYPED_FREE_PREDICTIONS_FROZEN'),
        ('independent_score','weld_pipeline.sjg_local_event_score_932','run136','base',
            ['--prediction-run','run135','--data-run','run133','--proposal-run','run132'],'LOCAL_EVENT_GEOMETRY_DIAGNOSED')]
    score_run='run136';training_run='run134'
    if resume_after_data:
        replacement={'run138':'run139','run134':'run140','run135':'run141','run136':'run142'}
        stages=[(name,module,replacement.get(run,run),runtime,[replacement.get(v,v) for v in extra],expected)
            for name,module,run,runtime,extra,expected in stages if name!='data']
        score_run='run142';training_run='run140'
    if resume_after_training:
        replacement={'run134':'run140','run135':'run144','run136':'run145'}
        stages=[(name,module,replacement.get(run,run),runtime,[replacement.get(v,v) for v in extra],expected)
            for name,module,run,runtime,extra,expected in stages if name in ('free_prediction','independent_score')]
        score_run='run145';training_run='run140'
    write(dest/'registered_protocol.json',dict(stage='REPAIRED_NATIVE_LOCAL_PILOT_ROUND',stages=stages,original_G=496,
        supplemental_TRAIN_G=4,training_epochs=8,proposal_run='run132',source_sha256=digest(__file__),
        no_data_expansion=True,reuse_verified_data=resume_after_data or resume_after_training,
        reuse_verified_training=resume_after_training,no_external_consultation=True,full_contract_pass=False,start_utc=utc()))
    completed=[]
    try:
        assert read(OUT/'run132/receipt.json')['status']=='UNIFIED_LOCAL_CANDIDATES_FROZEN'
        if resume_after_data or resume_after_training:
            assert read(OUT/'run133/receipt.json')['status']=='LOCAL_TYPED_GRAPH_TRAINING_DATA_READY'
            assert read(OUT/'run133/registered_protocol.json')['source_run']=='run132'
        if resume_after_training:assert read(OUT/'run140/receipt.json')['status']=='LOCAL_TYPED_EVENT_TRAINING_COMPLETE'
        for name,module,run,runtime,extra,expected in stages:
            if (OUT/run).exists():raise RuntimeError('Refusing to overwrite '+run)
            check_resources();env=os.environ.copy();paths=[str(ROOT)]
            executable=ROOT/('.venv-ptv3/Scripts/python.exe' if runtime=='ptv3' else 'tools/python/cpython-3.10-windows-x86_64-none/python.exe')
            if runtime=='ptv3':paths.append(str(ROOT/'.venv-ptv3/Lib/site-packages'))
            paths.append(str(ROOT/'.venv/Lib/site-packages'));env['PYTHONPATH']=';'.join(paths)
            write(dest/'progress.json',dict(current_stage=name,current_run=run,completed=completed,time_utc=utc()))
            print('REPAIRED_PILOT_STAGE_START',name,run,flush=True)
            subprocess.run([str(executable),'-u','-m',module,'--run',run,*extra],cwd=str(ROOT),env=env,check=True)
            receipt=read(OUT/run/'receipt.json');assert receipt['status']==expected
            completed.append(dict(stage=name,run=run,receipt_sha256=digest(OUT/run/'receipt.json')))
            print('REPAIRED_PILOT_STAGE_COMPLETE',name,run,flush=True)
        score=read(OUT/score_run/'receipt.json');train=read(OUT/training_run/'receipt.json')
        write(dest/'local_round_review.json',dict(groups=score['groups'],completed_epochs=train['epochs'],optimizer_updates=train['optimizer_updates'],
            unmet_or_not_run=score['unmet_or_not_run'],full_contract_pass=False,external_consultation_sent=False,
            next_action='Diagnose frozen missed and false curves, then complete parent/material/attribute heads under the unchanged contract.'))
        write(dest/'receipt.json',dict(status='REPAIRED_LOCAL_ROUND_COMPLETE_CONTRACT_PENDING',completed=completed,
            groups=score['groups'],optimizer_updates=train['optimizer_updates'],full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',completed=completed,error=traceback.format_exc(),end_utc=utc()));raise

if __name__=='__main__':main()
