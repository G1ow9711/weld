"""One visible desktop run, last fixed local-potential trial."""
import subprocess,sys,time
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,utc

def main():
    dest=OUT/'run247';dest.mkdir(exist_ok=False);assert read(OUT/'run244/receipt.json')['status']=='DUAL_PREFLIGHT_PASS'
    stages=[('train','weld_pipeline.sjg_dual_train_1068',['--run','run245'],240),
        ('predict','weld_pipeline.sjg_dual_test_1069',['--run','run246'],90),
        ('score','weld_pipeline.sjg_dual_test_1069',['--run','run246','--score'],60)]
    write(dest/'registered_protocol.json',dict(train_run='run245',test_run='run246',third_local_potential_trial=True,total_updates=32,all_G=500,
        failure_retires_local_potential_main_predictor=True,utc=utc()))
    done=[];reason=None;start=time.perf_counter()
    for name,module,args,timeout in stages:
        print('TWO_MODEL_REAL_TRIAL',name,flush=True)
        try:r=subprocess.run([sys.executable,'-u','-m',module,*args],timeout=timeout)
        except subprocess.TimeoutExpired:reason=name+'_timeout';break
        done.append(dict(stage=name,exit_code=r.returncode))
        if r.returncode:reason=name+'_error';break
        if name=='train' and read(OUT/'run245/receipt.json')['status']!='CONTACT_TRAIN_COMPLETE':reason='incomplete_train';break
    scorefile=OUT/'run246/source_21/geometry_score.json';s=read(scorefile) if scorefile.exists() else None
    parentfile=OUT/'run246/source_21/parent_d3_score.json';ps=read(parentfile)['categories']['overall'] if parentfile.exists() else None
    if s and (s['TP']!=1 or s['FP']!=0):reason='first_piece_actual_extraction_failed'
    if not reason and (ps is None or ps['TP']!=1 or ps['FP']!=0):reason='parent_D3_failed_or_missing'
    write(dest/'receipt.json',dict(status='QUICK_REAL_ROUND_STOPPED' if reason else 'SINGLE_TRAIN_FEASIBILITY_PASSED',stop_reason=reason,
        stages=done,score=s,parent_score=ps,not_tested=['BM79A_native_local','10','64','4','25','36','42','24'],all_G=500,
        full_contract_pass=False,seconds=time.perf_counter()-start,utc=utc()))
    print('TWO_MODEL_REAL_TRIAL_END',reason or 'single TRAIN feasibility passed only',flush=True)
if __name__=='__main__':main()
