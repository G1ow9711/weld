"""Continue the two existing producers; never restart on an observation timeout."""
from pathlib import Path
import os
import subprocess
import sys
import time
from weld_pipeline.sjg_local_runtime_904 import write, digest

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'server_rebuild/expansion_v1'
OUT=BASE/'second_arc_event_capacity_followup'


def main():
    OUT.mkdir(exist_ok=False)
    dependencies=[(996013, BASE/'union_nearest_arc_candidates_v1/new_fcaf2af1abee/d3_interval_capacity.json'), (996805, BASE/'training_arc_vertex_events_v1/new_fcaf2af1abee/receipt.json')]
    write(OUT/'protocol.json', dict(dependencies=[dict(pid=p,path=str(f)) for p,f in dependencies], scope='Existing TRAIN fc only, actual frozen input event capacity; no automatic model promotion or DEV reads'))
    while True:
        missing=[(p,f) for p,f in dependencies if not f.exists()]
        if not missing:break
        for pid,path in missing:
            os.kill(pid,0)
            stat=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1]
            if stat.split()[0]=='Z':raise RuntimeError(f'Producer {pid} exited without {path}')
        time.sleep(10)
    print('START actual_event_capacity',flush=True)
    with (OUT/'capacity.log').open('w') as log:
        subprocess.run([sys.executable,str(ROOT/'tools/evaluate_training_arc_event_capacity.py'),'--key','new_fcaf2af1abee'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    report=BASE/'training_arc_event_capacity_fc_v1/report.json'
    write(OUT/'receipt.json',dict(status='SECOND_TRAIN_INPUT_EVENT_CAPACITY_COMPLETE',report_sha256=digest(report),no_automatic_promotion=True))
    print('COMPLETE actual_event_capacity',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():write(OUT/'failure.json',dict(error=repr(error)))
        raise
