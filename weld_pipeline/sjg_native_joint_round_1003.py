"""Serial native-material graph training and frozen-output small-pilot review.

Geometry scores remain auxiliary until parent structure, weld types, E16 and
timing are implemented and accepted. No automatic retry or next optimization.
"""
import argparse
import os
import subprocess
import traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);completed=[]
    stages=[
        ('data','weld_pipeline.sjg_native_joint_training_data_1000','run186','base','LOCAL_TYPED_GRAPH_TRAINING_DATA_READY',[]),
        ('train','weld_pipeline.sjg_native_joint_train_1001','run187','ptv3','LOCAL_TYPED_EVENT_TRAINING_COMPLETE',[]),
        ('free_prediction','weld_pipeline.sjg_native_joint_infer_1002','run188','ptv3','LOCAL_TYPED_FREE_PREDICTIONS_FROZEN',[]),
        ('geometry_score','weld_pipeline.sjg_local_event_score_932','run189','base','LOCAL_EVENT_GEOMETRY_DIAGNOSED',
            ['--prediction-run','run188','--data-run','run186','--proposal-run','run185'])]
    write(dest/'registered_protocol.json',dict(stage='NATIVE_SURFACE_MATERIAL_GRAPH_SMALL_PILOT_ROUND',
        source_sha256=digest(__file__),stages=stages,original_eight_G=496,supplemental_TRAIN_G=4,
        train_G=363,development=['64'],arc_diagnostic=['24'],epochs=8,batch_queries=256,
        features='run166',instances='run174',proposal_run='run185',model='surface_material_joint_985',
        fixed_last_epoch_no_development_selection=True,no_external_consultation=True,
        material_graph_enters_training_and_cached_free_inference=True,
        parent_type_E16_timing_remain_required=True,full_contract_pass=False,start_utc=utc()))
    try:
        assert read(OUT/'run185/receipt.json')['status']=='NATIVE_JOINT_NINE_CANDIDATES_FROZEN'
        assert read(OUT/'run184/real_native_material_graph_controls.json')['cached_inference_vs_forward_max_error']<1e-5
        for name,module,run,runtime,expected,extra in stages:
            if (OUT/run).exists():raise RuntimeError('Refusing to overwrite existing stage '+run)
            check_resources();env=os.environ.copy()
            python=ROOT/('.venv-ptv3/Scripts/python.exe' if runtime=='ptv3' else 'tools/python/cpython-3.10-windows-x86_64-none/python.exe')
            paths=[str(ROOT)]
            if runtime=='ptv3':paths.append(str(ROOT/'.venv-ptv3/Lib/site-packages'))
            paths.append(str(ROOT/'.venv/Lib/site-packages'));env['PYTHONPATH']=';'.join(paths)
            print('NATIVE_ROUND_STAGE_START',name,run,flush=True)
            child=subprocess.Popen([str(python),'-u','-m',module,'--run',run]+extra,cwd=str(ROOT),env=env)
            write(dest/'progress.json',dict(current_stage=name,current_run=run,child_pid=child.pid,completed=completed,start_utc=utc()))
            code=child.wait()
            if code:raise RuntimeError('Stage failed '+run+' exit='+str(code))
            receipt=read(OUT/run/'receipt.json')
            if receipt['status']!=expected:raise RuntimeError('Unexpected terminal receipt '+run)
            completed.append(dict(stage=name,run=run,receipt_sha256=digest(OUT/run/'receipt.json')))
            print('NATIVE_ROUND_STAGE_COMPLETE',name,run,flush=True)
        score=read(OUT/'run189/receipt.json');training=read(OUT/'run187/receipt.json')
        write(dest/'local_round_review.json',dict(groups=score['groups'],epochs=training['epochs'],optimizer_updates=training['optimizer_updates'],
            unresolved_contract_items=score['unmet_or_not_run'],external_consultation_sent=False,
            full_contract_pass=False,next_action='Inspect all nine misses and false predictions, then implement remaining contract fields; no automatic extra epochs.'))
        write(dest/'receipt.json',dict(status='NATIVE_MATERIAL_GRAPH_ROUND_DIAGNOSED_CONTRACT_PENDING',completed=completed,
            optimizer_updates=training['optimizer_updates'],geometry_groups=score['groups'],full_contract_pass=False,end_utc=utc()))
        print('NATIVE_ROUND_COMPLETE','FULL_CONTRACT_PENDING',flush=True)
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',completed=completed,error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
