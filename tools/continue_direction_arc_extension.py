"""Collect fixed-method union graph seed diagnostics after the live producer."""
from pathlib import Path
import os,sys,time,subprocess
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'direction_arc_extension_followup'

def main():
    OUT.mkdir(exist_ok=False);key='new_fcaf2af1abee';source=BASE/'direction_union_arc_candidates_v1'/key
    write(OUT/'protocol.json',dict(producer_pid=13797,key=key,comparison='Frozen direction-chain circles extended only over observed annular union-graph edges. No refit or GT generation.',DEV_reads=0))
    while not (source/'receipt.json').exists():
        os.kill(13797,0)
        if Path('/proc/13797/stat').read_text().split(') ',1)[1].split()[0]=='Z':raise RuntimeError('Producer exited without receipt')
        time.sleep(10)
    commands=[('capacity',['tools/evaluate_local_graph_arc_capacity.py','--key',key,'--folder-name','direction_union_arc_candidates_v1'])]
    for name,command in commands:
        print('START',name,flush=True)
        with (OUT/(name+'.log')).open('w') as log:subprocess.run([sys.executable,*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',name,flush=True)
    write(OUT/'receipt.json',dict(status='DIRECTION_ARC_EXTENSION_CAPACITY_COMPLETE',capacity_sha256=digest(source/'d3_interval_capacity.json'),no_automatic_promotion=True))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():write(OUT/'failure.json',dict(error=repr(error)))
        raise
