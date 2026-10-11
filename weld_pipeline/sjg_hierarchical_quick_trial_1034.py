"""Fixed-order, failure-stopping pilot trial; prediction and scoring are separate processes."""
import argparse,subprocess,sys,time,shutil
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,utc

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--training-run',default='run218');a=ap.parse_args()
    dest=OUT/a.run;dest.mkdir(exist_ok=False);(dest/'source_snapshot').mkdir()
    for n in (1023,1028,1029,1032,1033,1034,1043):
        for path in (ROOT/'weld_pipeline').glob('*_'+str(n)+'.py'):shutil.copy2(path,dest/'source_snapshot'/path.name)
    train_receipt=OUT/a.training_run/'receipt.json'
    training=read(train_receipt) if train_receipt.exists() else read(OUT/a.training_run/'failure.json')
    order=['21','BM79A_native_local','10','64','4','25','36','42','24']
    incomplete=training.get('status')!='HIERARCHICAL_TRAINING_COMPLETE'
    checkpoint='final.pt' if train_receipt.exists() else 'failure_checkpoint.pt'
    write(dest/'registered_protocol.json',dict(order=order,cumulative_wall_budget_seconds=180,per_piece_seconds=60,
        incomplete_training_policy='diagnose source21 only; no acceptance claim',all_G=500,full_contract_pass=False,start_utc=utc()))
    results=[];start=time.perf_counter();reason=None
    for key in order:
        remaining=180-(time.perf_counter()-start)
        if remaining<=0:reason='cumulative_budget';break
        print('QUICK_TRIAL_BEGIN',key,flush=True)
        try:
            p=subprocess.run([sys.executable,'-u','-m','weld_pipeline.sjg_hierarchical_infer_1032','--run',a.run,'--source',key,'--training-run',a.training_run,'--checkpoint',checkpoint],timeout=min(70,remaining))
        except subprocess.TimeoutExpired:reason='whole_piece_process_timeout_'+key;break
        if p.returncode:reason='inference_failed_'+key;break
        remaining=180-(time.perf_counter()-start)
        if remaining<=0:reason='cumulative_budget_before_score';break
        try:p=subprocess.run([sys.executable,'-u','-m','weld_pipeline.sjg_hierarchical_quick_score_1033','--run',a.run,'--source',key],timeout=remaining)
        except subprocess.TimeoutExpired:reason='scoring_timeout_'+key;break
        if p.returncode:reason='score_failed_'+key;break
        s=read(dest/('source_'+key)/'geometry_score.json');results.append(s)
        if s['elapsed_seconds']>60:reason='whole_piece_60sec_'+key;break
        if incomplete:reason='training_budget_incomplete_single_piece_diagnosis';break
        failed=(key=='21' and (s['TP']!=1 or s['FP']!=0)) or (key=='BM79A_native_local' and (s['TP']!=4 or s['FP']!=0 or not all(t['passed'] for t in s['type_checks']))) or (key=='10' and (s['TP_over_G']<.8 or s['FP_over_G']>.2)) or (key=='64' and (s['TP_over_G']<.5 or s['FP_over_G']>.5))
        if failed:reason='fixed_fast_gate_failed_'+key;break
    write(dest/'receipt.json',dict(status='QUICK_TRIAL_STOPPED' if reason else 'QUICK_TRIAL_COMPLETED',stop_reason=reason,
        scores=results,not_tested=[k for k in order if k not in [s['source'] for s in results]],all_G=500,
        elapsed_seconds=time.perf_counter()-start,full_contract_pass=False,requires_Pro_before_next_training=True,end_utc=utc()))
    print('QUICK_TRIAL_END',reason or 'all nine tested','SCORED',len(results),flush=True)

if __name__=='__main__':main()
