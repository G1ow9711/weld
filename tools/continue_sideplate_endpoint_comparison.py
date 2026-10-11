"""Matched eight/nine TRAIN continuation and unchanged complete-output scoring."""
import time,sys,subprocess,traceback
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import read,write,digest

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'sideplate_endpoint_comparison_followup'

def main():
    OUT.mkdir(exist_ok=False)
    write(OUT/'protocol.json',dict(arms=['control8','expanded9'],warm_model=read(BASE/'boundary_provenance_batch02_expanded8_v1/receipt.json')['model_sha256'],extra_steps_each=16000,normalization_frozen=True,sideplate_header_audit_sha256=digest(BASE/'excel_split_header_semantics_audit_v1/report.json'),keys=['new_cfc732e5f068','new_ceeae49d6777'],detector='Existing frozen balanced reference classifier at0.5',both_models_frozen_before_scoring=True,no_parameter_search=True,DEV_reads=0))
    ready=BASE/'sideplate_endpoint_control8_v1/receipt.json';proc=Path('/proc/808928')
    identity=(proc/'stat').read_text().split()[21] if proc.exists() else None
    while not ready.exists():
        if identity is None or not proc.exists() or (proc/'stat').read_text().split()[21]!=identity:raise RuntimeError('Control trainer stopped without receipt')
        time.sleep(5)
    jobs=[('expanded9_train','train_sideplate_endpoint_comparison.py',['--include-original10'])]
    jobs += [(arm+'_'+key,'infer_sideplate_endpoint_output.py',['--comparison',arm,'--key',key]) for arm in ['control8','expanded9'] for key in ['new_cfc732e5f068','new_ceeae49d6777']]
    for name,script,args in jobs:
        print('START',name,flush=True)
        with (OUT/(name+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'tools'/script),*args],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
        print('COMPLETE',name,flush=True)
    rows=[]
    for arm in ['control8','expanded9']:
        scores=[]
        for key in ['new_cfc732e5f068','new_ceeae49d6777']:
            folder=BASE/('sideplate_endpoint_'+arm+'_full_output_v1')/key
            scores.append(dict(key=key,rows=read(folder/'receipt.json')['rows'],prediction_freeze_sha256=digest(folder/'prediction_freeze.json')))
        write(BASE/('sideplate_endpoint_'+arm+'_v1')/'decision.json',dict(status='EXPERIMENTAL_NOT_PROMOTED',scores=scores,full_contract_pass=False))
        rows.append(dict(arm=arm,scores=scores))
    write(OUT/'receipt.json',dict(status='MATCHED_SIDEPLATE_ENDPOINT_COMPARISON_COMPLETE',rows=rows,DEV_reads=0,no_automatic_promotion=True))

if __name__=='__main__':
    try:main()
    except BaseException:
        OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
