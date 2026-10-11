"""One desktop session: bounded training, fresh inference, independent scoring."""
import argparse,subprocess,sys,time
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,utc

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();dest=OUT/a.run;dest.mkdir(exist_ok=False)
    stages=[('train','weld_pipeline.sjg_structured_train_1060',['--run','run235'],220),
            ('predict','weld_pipeline.sjg_structured_test_1061',['--run','run236','--training-run','run235'],90),
            ('score','weld_pipeline.sjg_structured_test_1061',['--run','run236','--score'],60)]
    write(dest/'registered_protocol.json',dict(source='21',train_run='run235',test_run='run236',data_run='run224',all_G=500,
        first_piece_failure_stops=True,remaining_eight_NOT_TESTED=True,training_updates=32,training_budget_seconds=180,utc=utc()))
    start=time.perf_counter();done=[];reason=None
    for name,module,args,timeout in stages:
        print('REAL_SMALL_TRIAL',name,flush=True)
        try:r=subprocess.run([sys.executable,'-u','-m',module,*args],timeout=timeout)
        except subprocess.TimeoutExpired:reason=name+'_timeout';break
        done.append(dict(stage=name,exit_code=r.returncode))
        if r.returncode:reason=name+'_error';break
        if name=='train' and read(OUT/'run235/receipt.json')['status']!='CONTACT_TRAIN_COMPLETE':reason='incomplete_training';break
    scorefile=OUT/'run236/source_21/geometry_score.json';s=read(scorefile) if scorefile.exists() else None
    if s and (s['TP']!=1 or s['FP']!=0):reason='first_piece_actual_extraction_failed'
    parentfile=OUT/'run236/source_21/parent_d3_score.json'
    if s and not reason:
        if not parentfile.exists():reason='parent_D3_not_scored'
        else:
            ps=read(parentfile)['categories']['overall']
            if ps['TP']!=1 or ps['FP']!=0:reason='parent_D3_failed'
    write(dest/'receipt.json',dict(status='QUICK_REAL_ROUND_STOPPED' if reason else 'SINGLE_TRAIN_FEASIBILITY_PASSED',stop_reason=reason,
        stages=done,score=s,not_tested=['BM79A_native_local','10','64','4','25','36','42','24'],all_G=500,
        requires_Pro_before_next_training=bool(reason),full_contract_pass=False,seconds=time.perf_counter()-start,end_utc=utc()))
    print('REAL_SMALL_TRIAL_END',reason or 'single TRAIN gate only',flush=True)
if __name__=='__main__':main()
