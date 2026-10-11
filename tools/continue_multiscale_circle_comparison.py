"""Collect fixed-method union graph seed diagnostics after the live producer."""
from pathlib import Path
import os,sys,time,subprocess
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'multiscale_circle_comparison_followup'

def main():
    OUT.mkdir(exist_ok=False);key='new_fcaf2af1abee';source=BASE/'multiscale_arc_seeds_v1'/key
    write(OUT/'protocol.json',dict(producer_pid=23243,key=key,comparison='Same observed direction chains, old seeds retained; dyadic contiguous subpath circles fitted with original least-squares method, no reference-based windows.',DEV_reads=0))
    while not (source/'receipt.json').exists():
        os.kill(23243,0)
        if Path('/proc/23243/stat').read_text().split(') ',1)[1].split()[0]=='Z':raise RuntimeError('Producer exited without receipt')
        time.sleep(10)
    commands=[('extent',['tools/diagnose_local_graph_arc_extent.py','--key',key,'--graph','multiscale']),('capacity',['tools/evaluate_local_graph_arc_capacity.py','--key',key,'--folder-name','multiscale_arc_seeds_v1'])]
    for name,command in commands:
        print('START',name,flush=True)
        with (OUT/(name+'.log')).open('w') as log:subprocess.run([sys.executable,*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',name,flush=True)
    write(OUT/'receipt.json',dict(status='MULTISCALE_CIRCLE_COMPARISON_COMPLETE',capacity_sha256=digest(source/'d3_interval_capacity.json'),extent_sha256=digest(BASE/('multiscale_graph_arc_extent_'+key+'_v1')/'report.json'),no_automatic_promotion=True))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():write(OUT/'failure.json',dict(error=repr(error)))
        raise
