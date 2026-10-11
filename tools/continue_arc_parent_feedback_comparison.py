"""Complete matched conditional TRAIN feedback experiment and free-decode scoring."""
from pathlib import Path
import sys,time,subprocess,traceback
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'arc_parent_feedback_comparison_followup'

def main():
    OUT.mkdir(exist_ok=False);write(OUT/'protocol.json',dict(warm_model_sha256=read(BASE/'arc_parent_boundary_pointer_v1/receipt.json')['model_sha256'],additional_steps_each=4000,arms=['fixed','model_valid'],free_decode_evaluation=True,conditional_TRAIN_only=True,no_automatic_promotion=True,DEV_reads=0))
    ready=BASE/'arc_parent_feedback_fixed_v1/receipt.json';proc=Path('/proc/996099');identity=(proc/'stat').read_text().split()[21] if proc.exists() else None
    while not ready.exists():
        if identity is None or not proc.exists() or (proc/'stat').read_text().split()[21]!=identity:raise RuntimeError('Fixed trainer stopped without receipt')
        time.sleep(5)
    jobs=[('model_valid_train','train_arc_parent_feedback_comparison.py',['--feedback','model_valid'])]+[(arm+'_fit','evaluate_arc_parent_pointer_train_fit.py',['--model-folder','arc_parent_feedback_'+arm+'_v1']) for arm in ('fixed','model_valid')]
    for name,script,args in jobs:
        print('START',name,flush=True)
        with (OUT/(name+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools'/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',name,flush=True)
    rows=[]
    for arm in ('fixed','model_valid'):
        folder=BASE/('arc_parent_feedback_'+arm+'_v1');fit=read(folder/'conditional_train_geometry.json')
        rows.append(dict(arm=arm,complete_conditioned_carriers=fit['complete_conditioned_carriers'],conditioned_carriers=fit['conditioned_carriers'],all_conditioned_outputs=fit['all_conditioned_outputs_score']['categories']['overall'],fit_sha256=digest(folder/'conditional_train_geometry.json')))
        write(folder/'decision.json',dict(status='CONDITIONAL_TRAIN_NOT_PROMOTED',reason='No runtime carrier selection, reference rejection, heldout validation or complete contract verification.',full_contract_pass=False))
    write(OUT/'receipt.json',dict(status='MATCHED_CONDITIONAL_FEEDBACK_COMPARISON_COMPLETE',rows=rows,no_automatic_promotion=True,DEV_reads=0))

if __name__=='__main__':
    try:main()
    except BaseException:
        OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
